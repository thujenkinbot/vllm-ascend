from __future__ import annotations

import weakref
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from functools import cached_property
from multiprocessing.synchronize import Lock as LockType
from typing import Any

import vllm.v1.executor.multiproc_executor
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.logger import init_logger
from vllm.utils.network_utils import get_distributed_init_method, get_loopback_ip, get_open_port
from vllm.utils.system_utils import get_mp_context
from vllm.v1.executor.abstract import FailureCallback
from vllm.v1.executor.multiproc_executor import (
    FutureWrapper,
    MultiprocExecutor,
    UnreadyWorkerProcHandle,
    WorkerProc,
    set_multiprocessing_worker_envs,
)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.worker.multi_edge_control import (
    CloudRPCRequest,
    DeferCloudRPC,
    MultiEdgeCloudArbiter,
    MultiEdgeControlClient,
)

logger = init_logger(__name__)


class AscendMultiprocExecutor(MultiprocExecutor):
    @cached_property
    def max_concurrent_batches(self) -> int:
        """Keep the multi-edge MVP strictly single-batch.

        The logical edge-cloud topology reports PP=2, which would normally
        enable vLLM's PP batch queue even when async scheduling is disabled.
        The MVP cloud adapter deliberately supports only one in-flight batch
        per edge, so do not let the synthetic PP size enable that queue.
        """
        if self.parallel_config.enable_edge_cloud and self.parallel_config.num_edges > 1:
            return 1
        return super().max_concurrent_batches

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback: FailureCallback | None = None

        tensor_parallel_size, pp_parallel_size, pcp_parallel_size = self._get_parallel_sizes()
        if not self.parallel_config.enable_edge_cloud:
            assert self.world_size == tensor_parallel_size * pp_parallel_size * pcp_parallel_size, (
                f"world_size ({self.world_size}) must be equal to the "
                f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
                f"_parallel_size ({pp_parallel_size}) x prefill_context"
                f"_parallel_size ({pcp_parallel_size}). "
            )

        # Set multiprocessing envs
        set_multiprocessing_worker_envs()

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # get_loopback_ip() for communication.
        distributed_init_method = get_distributed_init_method(get_loopback_ip(), get_open_port())
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None
        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        multi_edge = self.parallel_config.enable_edge_cloud and self.parallel_config.num_edges > 1
        if multi_edge:
            logger.info(
                "Using AscendMultiprocExecutor for multi-edge-cloud: node_rank=%d, role=%s, local_world_size=%d",
                self.parallel_config.node_rank,
                "edge" if self.parallel_config.is_edge_node else "cloud",
                self.local_world_size,
            )
            if self.scheduler_config.async_scheduling:
                raise ValueError("multi-edge-cloud MVP requires async_scheduling=False")
            if self.speculative_config is not None:
                raise ValueError("multi-edge-cloud MVP does not support speculative decoding")
            if self.lora_config is not None:
                raise ValueError("multi-edge-cloud MVP does not support LoRA")
            if self.model_config.multimodal_config is not None:
                raise ValueError("multi-edge-cloud MVP currently supports text models only")
            if self.cache_config.enable_prefix_caching:
                raise ValueError("multi-edge-cloud MVP does not support prefix caching")
        if multi_edge and self.parallel_config.is_edge_node:
            edge_id = envs_ascend.VLLM_ASCEND_EDGE_CLOUD_EDGE_IDX
            if edge_id != self.parallel_config.node_rank:
                raise ValueError(
                    "VLLM_ASCEND_EDGE_CLOUD_EDGE_IDX must match node_rank in "
                    f"the static MVP topology: {edge_id} != "
                    f"{self.parallel_config.node_rank}"
                )
        if self.parallel_config.node_rank_within_dp == 0 or multi_edge:
            # For leader node within each dp rank,
            # each dp will have its own leader multiproc executor.
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            self.rpc_broadcast_mq = MessageQueue(
                self.local_world_size if multi_edge else self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=(get_loopback_ip() if multi_edge else self.parallel_config.master_addr),
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        # Create workers
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            if self.parallel_config.enable_edge_cloud:
                global_start_rank = self.parallel_config.edge_cloud_global_start_rank
            else:
                global_start_rank = self.local_world_size * self.parallel_config.node_rank_within_dp

            # When using fork, keep track of socket file descriptors that are
            # inherited by the worker, so that we can close them in subsequent
            # workers
            inherited_fds: list[int] | None = [] if context.get_start_method() == "fork" else None

            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                is_driver_worker = self._is_driver_worker(global_rank)
                unready_worker_handle = AscendWorkerProc.make_worker_process(
                    vllm_config=self.vllm_config,
                    local_rank=local_rank,
                    rank=global_rank,
                    distributed_init_method=distributed_init_method,
                    input_shm_handle=scheduler_output_handle,
                    shared_worker_lock=shared_worker_lock,
                    is_driver_worker=is_driver_worker,
                    inherited_fds=inherited_fds,
                )
                unready_workers.append(unready_worker_handle)
                if inherited_fds is not None:
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.

            # Wait for all local workers to be ready.
            self.workers = AscendWorkerProc.wait_for_ready(unready_workers)
            if multi_edge:
                self.workers.sort(key=lambda worker: worker.rank)

            # Start background thread to monitor worker health if not in headless mode.
            if self.monitor_workers:
                self.start_worker_monitor()

            self.response_mqs = []
            # Only leader node have remote response mqs
            if multi_edge:
                for worker in self.workers:
                    local_message_queue = worker.worker_response_mq
                    assert local_message_queue is not None
                    self.response_mqs.append(local_message_queue)
            elif self.parallel_config.node_rank_within_dp == 0 and (
                not self.parallel_config.enable_edge_cloud or self.parallel_config.is_edge_node
            ):
                for rank in range(self.world_size):
                    local_idx = rank - global_start_rank
                    if 0 <= local_idx < self.local_world_size:
                        local_message_queue = self.workers[local_idx].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[rank]
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.

            # Wait for all input mqs to be ready.
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # Wait for all remote response mqs to be ready.
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()
            self.futures_queue = deque[tuple[FutureWrapper, Callable]]()
            self._post_init_executor()

            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()
        self._multi_edge_control_client: MultiEdgeControlClient | None = None

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        self.world_size = self.parallel_config.world_size
        if not self.parallel_config.enable_edge_cloud:
            assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (
                f"global world_size ({self.parallel_config.world_size}) must be "
                f"divisible by nnodes_within_dp "
                f"({self.parallel_config.nnodes_within_dp}). "
            )
        self.local_world_size = self.parallel_config.local_world_size
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        return tp_size, pp_size, pcp_size

    def _post_init_executor(self) -> None:
        pass

    def _is_driver_worker(self, rank: int) -> bool:
        if self.parallel_config.enable_edge_cloud:
            return rank == self.parallel_config.edge_cloud_global_start_rank
        return rank % self.parallel_config.tensor_parallel_size == 0

    def _get_output_rank(self) -> int:
        if self.parallel_config.enable_edge_cloud:
            if self.parallel_config.num_edges > 1 and self.parallel_config.is_edge_node:
                return self.parallel_config.node_rank
            return 0
        return super()._get_output_rank()

    def _get_multi_edge_control_client(self) -> MultiEdgeControlClient:
        client = self._multi_edge_control_client
        if client is None:
            edge_id = self.parallel_config.node_rank
            client = MultiEdgeControlClient(
                edge_id=edge_id,
                cloud_addr=envs_ascend.VLLM_ASCEND_EDGE_CLOUD_CLOUD_ADDR,
                port=envs_ascend.VLLM_ASCEND_EDGE_CLOUD_CONTROL_PORT,
            )
            self._multi_edge_control_client = client
        return client

    def collective_rpc(
        self,
        method,
        timeout=None,
        args=(),
        kwargs=None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator=None,
    ):
        multi_edge = self.parallel_config.enable_edge_cloud and self.parallel_config.num_edges > 1
        if not multi_edge or not self.parallel_config.is_edge_node:
            return super().collective_rpc(
                method,
                timeout,
                args,
                kwargs,
                non_block,
                unique_reply_rank,
                kv_output_aggregator,
            )

        # EngineCore always submits execute_model with non_block=True, even
        # when async scheduling and the batch queue are disabled. Preserve the
        # Future-shaped executor API while completing the multi-edge RPC
        # synchronously. This is compatibility plumbing, not async scheduling:
        # max_concurrent_batches remains one and this call returns only after
        # both the local edge and shared cloud work have completed.
        if non_block:
            future: Future[Any] = Future()
            try:
                result = self._multi_edge_collective_rpc_sync(
                    method,
                    timeout,
                    args,
                    kwargs,
                    unique_reply_rank,
                    kv_output_aggregator,
                )
            except Exception as error:
                future.set_exception(error)
            else:
                future.set_result(result)
            return future

        return self._multi_edge_collective_rpc_sync(
            method,
            timeout,
            args,
            kwargs,
            unique_reply_rank,
            kv_output_aggregator,
        )

    def _multi_edge_collective_rpc_sync(
        self,
        method,
        timeout,
        args,
        kwargs,
        unique_reply_rank,
        kv_output_aggregator,
    ):
        """Execute one edge/cloud RPC using the MVP's serial contract."""

        edge_id = self.parallel_config.node_rank
        method_name = method if isinstance(method, str) else None
        edge_only_methods = {"sample_tokens", "take_draft_token_ids"}
        shared_read_methods = {"determine_available_memory", "get_kv_cache_spec"}
        relay_to_cloud = method_name not in edge_only_methods and (
            edge_id == 0 or method_name == "execute_model" or method_name in shared_read_methods
        )
        client = self._get_multi_edge_control_client() if relay_to_cloud else None
        if client is not None:
            client.send(
                CloudRPCRequest(
                    edge_id=edge_id,
                    method=method,
                    timeout=timeout,
                    args=args,
                    kwargs=kwargs or {},
                )
            )

        # Every edge has one local worker. Request its response regardless of
        # global rank, then restore MultiprocExecutor's unique-reply contract.
        local_results = super().collective_rpc(
            method,
            timeout,
            args,
            kwargs,
            False,
            None,
            None,
        )
        cloud_results = client.receive(timeout) if client is not None else []

        if unique_reply_rank is not None or kv_output_aggregator is not None:
            return local_results[0]
        if relay_to_cloud:
            # Preserve global-rank indexing. Edge 0 is the startup authority;
            # other edges initialize independently, so duplicate its shape and
            # memory result into their otherwise absent slots.
            return [local_results[0]] * self.parallel_config.num_edges + [*cloud_results]
        # Edge workers use WorkerWrapperBase.global_rank to select cache config.
        return [local_results[0]] * (edge_id + 1)

    def execute_model(self, scheduler_output, non_block: bool = False):
        if (
            self.parallel_config.enable_edge_cloud
            and self.parallel_config.num_edges > 1
            and self.parallel_config.is_edge_node
        ):
            scheduler_output.edge_id = self.parallel_config.node_rank
        return super().execute_model(scheduler_output, non_block=non_block)

    def start_worker_monitor(self, inline=False) -> None:
        if not (
            inline
            and self.parallel_config.enable_edge_cloud
            and self.parallel_config.num_edges > 1
            and not self.parallel_config.is_edge_node
        ):
            return super().start_worker_monitor(inline=inline)

        # Preserve worker liveness monitoring while this headless process
        # serially serves all edge control streams.
        super().start_worker_monitor(inline=False)
        arbiter = MultiEdgeCloudArbiter(
            num_edges=self.parallel_config.num_edges,
            port=envs_ascend.VLLM_ASCEND_EDGE_CLOUD_CONTROL_PORT,
        )
        logger.info(
            "Multi-edge cloud arbiter listening on port %d for %d edges",
            arbiter.port,
            self.parallel_config.num_edges,
        )
        shared_read_methods = {"determine_available_memory", "get_kv_cache_spec"}
        shared_read_cache = {}

        def execute(request: CloudRPCRequest):
            method_name = request.method if isinstance(request.method, str) else None
            if request.edge_id > 0 and method_name in shared_read_methods:
                if method_name not in shared_read_cache:
                    raise DeferCloudRPC
                return shared_read_cache[method_name]

            logger.debug(
                "Multi-edge cloud RPC: edge_id=%d method=%s",
                request.edge_id,
                method_name or "callable",
            )

            result = MultiprocExecutor.collective_rpc(
                self,
                request.method,
                request.timeout,
                request.args,
                request.kwargs,
                False,
                None,
                None,
            )
            if request.edge_id == 0 and method_name in shared_read_methods:
                shared_read_cache[method_name] = result
            return result

        try:
            arbiter.run(execute, lambda: self.is_failed)
        finally:
            arbiter.close()

    def shutdown(self):
        client = getattr(self, "_multi_edge_control_client", None)
        if client is not None:
            client.close()
            self._multi_edge_control_client = None
        super().shutdown()


class AscendWorkerProc(WorkerProc):
    def _init_message_queues(self, input_shm_handle: Handle, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.enable_edge_cloud and parallel_config.num_edges > 1:
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(input_shm_handle, self.worker.local_rank)
            self.worker_response_mq = MessageQueue(1, 1)
            self.peer_response_handles = []
            return
        super()._init_message_queues(input_shm_handle, vllm_config)

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
        is_driver_worker: bool = False,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # Ready pipe to communicate readiness from child to parent
        ready_reader, ready_writer = context.Pipe(duplex=False)
        # Death pipe to let child detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)
        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            # Have the worker close parent end of this worker's pipes too
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=False,
        )

        proc.start()
        # Close child ends of pipes here in the parent
        ready_writer.close()
        death_reader.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)


vllm.v1.executor.multiproc_executor.MultiprocExecutor = AscendMultiprocExecutor
