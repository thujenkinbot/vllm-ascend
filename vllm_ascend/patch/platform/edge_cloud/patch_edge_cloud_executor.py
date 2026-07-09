#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Edge-cloud patch: MultiprocExecutor → AscendMultiprocExecutor.

Overrides vLLM's MultiprocExecutor._init_executor to support edge-cloud
role-sensitive rank assignment and worker topology.
"""

import logging
import weakref
from collections import deque

import vllm.envs as envs
import vllm.v1.executor.multiproc_executor
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_loopback_ip,
    get_open_port,
)
from vllm.utils.ompmultiprocessing import OMPProcessManager
from vllm.utils.system_utils import get_mp_context
from vllm.v1.executor.multiproc_executor import (
    FutureWrapper,
    MultiprocExecutor,
    UnreadyWorkerProcHandle,
    WorkerProc,
    set_multiprocessing_worker_envs,
)

from vllm_ascend.distributed.parallel_state import (
    get_cloud_npu_count,
    get_edge_npu_count,
    is_edge_cloud_pp_mode,
    is_edge_device,
)

logger = logging.getLogger(__name__)


class AscendMultiprocExecutor(MultiprocExecutor):
    def _init_executor(self) -> None:
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback = None

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        if not is_edge_cloud_pp_mode():
            assert self.world_size == tp_size * pp_size * pcp_size, (
                f"world_size ({self.world_size}) must be equal to the "
                f"tensor_parallel_size ({tp_size}) x pipeline"
                f"_parallel_size ({pp_size}) x prefill_context"
                f"_parallel_size ({pcp_size}). "
            )

        set_multiprocessing_worker_envs()

        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port()
        )
        self.rpc_broadcast_mq = None
        scheduler_output_handle = None
        if self.parallel_config.node_rank_within_dp == 0:
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            mq_connect_ip = get_ip()
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=mq_connect_ip,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers = []
        success = False
        try:
            if is_edge_cloud_pp_mode():
                global_start_rank = (
                    0
                    if is_edge_device()
                    else get_edge_npu_count()
                )
            else:
                global_start_rank = (
                    self.local_world_size * self.parallel_config.node_rank_within_dp
                )
            inherited_fds = (
                [] if context.get_start_method() == "fork" else None
            )

            cpu_omp_manager = OMPProcessManager(self.vllm_config)
            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                is_driver_worker = self._is_driver_worker(global_rank)
                with cpu_omp_manager.configure_omp_envs(
                    rank=global_rank, local_rank=local_rank
                ):
                    unready_worker_handle = WorkerProc.make_worker_process(
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

            self.workers = WorkerProc.wait_for_ready(unready_workers)

            if self.monitor_workers:
                self.start_worker_monitor()

            self.response_mqs = []
            if self.parallel_config.node_rank_within_dp == 0 and (
                not is_edge_cloud_pp_mode()
                or is_edge_device()
            ):
                for rank in range(self.world_size):
                    local_idx = rank - global_start_rank
                    if 0 <= local_idx < self.local_world_size:
                        local_message_queue = self.workers[
                            local_idx
                        ].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[
                            rank
                        ]
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)

            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()

            self.futures_queue = deque[FutureWrapper]()

            self._post_init_executor()

            success = True
        finally:
            if not success:
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        self.world_size = self.parallel_config.world_size
        if not is_edge_cloud_pp_mode():
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

    def _is_driver_worker(self, rank: int) -> bool:
        if is_edge_cloud_pp_mode():
            return rank == (
                0
                if is_edge_device()
                else get_edge_npu_count()
            )
        return rank % self.parallel_config.tensor_parallel_size == 0

    def _get_output_rank(self) -> int:
        if is_edge_cloud_pp_mode():
            return 0
        return super()._get_output_rank()


vllm.v1.executor.multiproc_executor.MultiprocExecutor = AscendMultiprocExecutor

# Patch serve.py's module-level binding so headless mode uses the patched class.
try:
    import importlib
    _serve_mod = importlib.import_module("vllm.entrypoints.cli.serve")
    _serve_mod.MultiprocExecutor = AscendMultiprocExecutor  # type: ignore[attr-defined]
except Exception:
    pass

logger.debug("patch_edge_cloud_executor applied successfully")
