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
#
"""ZMQ scheduler channels and Passive EngineCore process for vllm-ascend.

This module is the vllm-ascend home of four classes that previously lived in
``vllm/v1/engine/core.py`` of the vllm-pdmix downstream fork:

* :class:`PPSchedulerZmqPublisher` — pp rank0 → pp rank1 SchedulerOutput
  publisher (PUSH socket + background pickling thread).
* :class:`PPSchedulerZmqSubscriber` — pp rank1 receiver counterpart.
* :class:`PPSchedulerZmqChannel` — bidirectional channel that owns one
  publisher + one subscriber for the edge-cloud PD-separation flow.
* :class:`PassiveEngineCoreProc` — non-leader PP rank engine process driver
  that consumes scheduler decisions over ZMQ instead of producing them.

All hooks back into the upstream ``EngineCore`` / ``EngineCoreProc`` lifecycle
are installed by :mod:`vllm_ascend.patch.platform.patch_engine_core` so that
the upstream ``vllm/v1/engine/core.py`` stays untouched.
"""
from __future__ import annotations

import copy
import os
import pickle
import queue
import signal
import threading
import time
from typing import TYPE_CHECKING, Optional

import zmq
from vllm import envs
from vllm.logger import logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.tracing import maybe_init_worker_tracer
from vllm.v1.core.sched.output import BatchType, SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _import_passive_scheduler_module():
    """Lazily resolve the PassiveScheduler implementation.

    The module path ``vllm.v1.core.sched.passive_scheduler`` is aliased to the
    vllm-ascend implementation by
    :mod:`vllm_ascend.patch.platform.patch_pd_scheduler_shim`. We try that
    canonical alias first (so any callsite that legacy-imports from the vLLM
    path keeps working) and fall back to the direct ascend path otherwise.
    """
    try:
        import vllm.v1.core.sched.passive_scheduler as passive_scheduler
    except ImportError:
        try:
            import vllm_ascend.core.passive_scheduler as passive_scheduler
        except ImportError as err:
            raise RuntimeError(
                "PassiveScheduler is provided by the vllm-ascend plugin. "
                "Make sure vllm_ascend.patch.platform.patch_pd_scheduler_shim "
                "is imported before starting PassiveEngineCore."
            ) from err
    return passive_scheduler


class PPSchedulerZmqPublisher:
    """Publishes SchedulerOutput from pp rank0 EngineCore to pp rank1
    PassiveEngineCore via ZMQ PUSH/PULL pattern.

    Architecture: caller thread (scheduler loop) only enqueues the raw
    `SchedulerOutput` object into `_queue`. A dedicated background thread
    pulls from the queue, pickles, and sends over ZMQ. This keeps the
    scheduler step path free of pickling cost and mirrors the symmetric
    queue.Queue bridge used on the subscriber/PassiveScheduler side.
    """

    SHUTDOWN_TIMEOUT: float = 2.0

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._queue: queue.Queue[Optional[tuple[int, SchedulerOutput]]] = (
            queue.Queue(maxsize=1000)
        )
        self._running = True
        self._seq = 0

        # Set up ZMQ PUSH socket
        self._ctx = zmq.Context.instance()
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.set_hwm(1000)
        # Bind if wildcard (pp rank0), otherwise connect
        if "*" in endpoint or "::" in endpoint:
            self._push.bind(endpoint)
        else:
            self._push.connect(endpoint)

        logger.info("PP Scheduler ZMQ publisher started on %s", endpoint)

        # Start background publisher thread
        self._thread = threading.Thread(
            target=self._publisher_thread,
            daemon=True,
            name="pp-scheduler-zmq-pub",
        )
        self._thread.start()

    def publish(self, scheduler_output: SchedulerOutput) -> None:
        """Queue a SchedulerOutput for publishing. Non-blocking: drops the
        message if the bridge queue is full (back-pressure protection).
        """
        if not self._running or scheduler_output.batch_type is BatchType.EMPTY:
            return
        try:
            seq = self._seq
            self._seq += 1
            self._queue.put_nowait((seq, scheduler_output))
        except queue.Full:
            logger.warning(
                "PP Scheduler ZMQ publish queue full, dropping message"
            )

    def _publisher_thread(self) -> None:
        while self._running or self._queue.qsize() > 0:
            try:
                item = self._queue.get(timeout=0.1)
                if item is None:
                    break
                seq, scheduler_output = item
                try:
                    data = pickle.dumps(
                        scheduler_output, protocol=pickle.HIGHEST_PROTOCOL
                    )
                except Exception:
                    logger.exception(
                        "Failed to serialize SchedulerOutput for ZMQ"
                    )
                    continue
                seq_bytes = seq.to_bytes(8, "big")
                self._push.send_multipart((seq_bytes, data))
            except queue.Empty:
                continue
            except Exception:
                logger.exception("Error in PP scheduler ZMQ publisher thread")
                time.sleep(0.1)

    def shutdown(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        try:
            if self._push is not None:
                self._push.close(linger=0)
        except Exception:
            pass


class PPSchedulerZmqSubscriber:
    """Receives SchedulerOutput from pp rank0 EngineCore on pp rank1
    via ZMQ PUSH/PULL pattern.

    Runs a background thread that receives SchedulerOutput messages,
    saves them locally, and logs a summary.
    """

    SHUTDOWN_TIMEOUT: float = 2.0

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._running = True
        self._received_outputs: list[tuple[int, SchedulerOutput]] = []
        self._lock = threading.Lock()

        # Set up ZMQ PULL socket
        self._ctx = zmq.Context.instance()
        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.set_hwm(1000)
        self._pull.connect(endpoint)

        logger.info("PP Scheduler ZMQ subscriber connecting to %s", endpoint)

        # Start background subscriber thread
        self._thread = threading.Thread(
            target=self._subscriber_thread,
            daemon=True,
            name="pp-scheduler-zmq-sub",
        )
        self._thread.start()

    def _subscriber_thread(self) -> None:
        while self._running:
            try:
                if not self._pull.poll(timeout=100):
                    continue
                seq_bytes, data = self._pull.recv_multipart()
                seq = int.from_bytes(seq_bytes, "big")
                scheduler_output = pickle.loads(data)
                if scheduler_output.batch_type is BatchType.EMPTY:
                    continue
                with self._lock:
                    self._received_outputs.append((seq, scheduler_output))
                # logger.info(
                #     "PP rank1 received SchedulerOutput seq=%d, "
                #     "total_scheduled_tokens=%d, "
                #     "new_reqs=%d, cached_reqs=%d, "
                #     "finished_req_ids=%s",
                #     seq,
                #     scheduler_output.total_num_scheduled_tokens,
                #     len(scheduler_output.scheduled_new_reqs),
                #     scheduler_output.scheduled_cached_reqs.num_reqs,
                #     scheduler_output.finished_req_ids,
                # )
            except zmq.ZMQError:
                if self._running:
                    logger.exception("ZMQ error in PP scheduler subscriber")
            except Exception:
                if self._running:
                    logger.exception(
                        "Error in PP scheduler ZMQ subscriber thread"
                    )

    def get_latest_output(self) -> Optional[SchedulerOutput]:
        """Return the most recently received SchedulerOutput, or None."""
        with self._lock:
            if self._received_outputs:
                return self._received_outputs[-1][1]
        return None

    def get_all_outputs(self) -> list[tuple[int, SchedulerOutput]]:
        """Return all received (seq, SchedulerOutput) pairs."""
        with self._lock:
            return list(self._received_outputs)

    def consume_new_outputs(self) -> list[tuple[int, SchedulerOutput]]:
        """Return and clear all new (seq, SchedulerOutput) pairs since last
        call.
        """
        with self._lock:
            outputs = self._received_outputs
            self._received_outputs = []
            return outputs

    def shutdown(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        try:
            if self._pull is not None:
                self._pull.close(linger=0)
        except Exception:
            pass


class PPSchedulerZmqChannel:
    """Bidirectional ZMQ channel for SchedulerOutput exchange between two
    PP engines.

    A `PPSchedulerZmqChannel` owns one send side (a `PPSchedulerZmqPublisher`)
    and one receive side (a `PPSchedulerZmqSubscriber`), each backed by its
    own dedicated ZMQ PUSH / PULL socket on independent endpoints. It is the
    symmetric primitive needed by the edge-cloud PD-separation flow:

    - Edge constructs one channel with::

          send_endpoint = "tcp://*:<PRE_OUT_PORT>"          # bind, edge → cloud
          recv_endpoint = "tcp://<cloud_addr>:<POST_OUT_PORT>"   # connect

      and uses ``publish()`` to forward PREFILL_FIRST / DECODE_FIRST
      batches, and ``consume_new_outputs()`` to drain PREFILL_LAST /
      DECODE_LAST batches returned from the cloud.

    - Cloud constructs the mirror channel with::

          send_endpoint = "tcp://*:<POST_OUT_PORT>"          # bind, cloud → edge
          recv_endpoint = "tcp://<master_addr>:<PRE_OUT_PORT>"   # connect

      The same publish/consume API drives the opposite traffic direction.

    Both endpoints use the same PUSH/PULL + background-thread + queue.Queue
    bridge as the legacy unidirectional classes, so no scheduler-thread time
    is spent on pickling or socket I/O.

    Channel naming (``name``) is purely diagnostic; it is included in the
    log lines emitted by the underlying publisher / subscriber so the two
    edge-cloud channels can be told apart in a single combined log.
    """

    def __init__(
        self,
        send_endpoint: str,
        recv_endpoint: str,
        name: str = "pp-channel",
    ) -> None:
        self._name = name
        self._send_endpoint = send_endpoint
        self._recv_endpoint = recv_endpoint
        # Publisher binds-if-wildcard / connects-otherwise (see existing
        # `PPSchedulerZmqPublisher.__init__`); subscriber always connects.
        # The endpoints chosen by the caller therefore fully determine the
        # bind/connect roles of each side.
        self._publisher = PPSchedulerZmqPublisher(send_endpoint)
        self._subscriber = PPSchedulerZmqSubscriber(recv_endpoint)
        logger.info(
            "PPSchedulerZmqChannel[%s] up: send=%s, recv=%s",
            name,
            send_endpoint,
            recv_endpoint,
        )

    def publish(self, scheduler_output: SchedulerOutput) -> None:
        """Queue a SchedulerOutput for the peer. Non-blocking."""
        # logger.info(
        #     f"Send scheduler_output to edge, batch_type: "
        #     f"{scheduler_output.batch_type}",
        # )
        self._publisher.publish(scheduler_output)

    def consume_new_outputs(self) -> list[tuple[int, SchedulerOutput]]:
        """Return and clear all (seq, SchedulerOutput) pairs received since
        the last call. Suitable for use as the ``pp_subscriber`` argument
        of `PassiveScheduler`, which only relies on this method.
        """
        return self._subscriber.consume_new_outputs()

    def shutdown(self) -> None:
        self._publisher.shutdown()
        self._subscriber.shutdown()




def _trim_scheduler_output_for_worker_enqueue(
    scheduler_output: SchedulerOutput,
    prev_dispatch_req_ids: set[str] | None,
) -> SchedulerOutput:
    """Trim large cached token lists before cloud EngineCore -> worker MQ.

    Cloud worker ``_update_states`` only needs ``all_token_ids`` for cached
    requests that are not already in its persistent batch and have output
    tokens.  The best local approximation is the previous cloud dispatch batch:
    continuously dispatched requests can drop ``all_token_ids`` while newly
    appearing / resumed requests keep it.

    Entries that are kept MUST remain the complete token history
    (prompt + output): the worker recovery path slices
    ``all_token_ids[-num_output_tokens:]`` and assumes the prompt is
    included. Truncating entries corrupts the recovered output history and
    causes garbled decode output (see [EDGE-CLOUD-RECOVER]).
    """
    cached = scheduler_output.scheduled_cached_reqs
    if cached is None:
        return scheduler_output

    all_token_ids = getattr(cached, "all_token_ids", None)
    if not all_token_ids:
        return scheduler_output

    prev_dispatch_req_ids = prev_dispatch_req_ids or set()
    resumed_req_ids = getattr(cached, "resumed_req_ids", set()) or set()
    num_output_tokens_by_req = {
        req_id: num_output_tokens
        for req_id, num_output_tokens in zip(
            getattr(cached, "req_ids", ()),
            getattr(cached, "num_output_tokens", ()),
        )
    }
    # Keep every entry the cloud worker may need for the resume/recovery
    # path in _update_states. The recovery trigger on the worker is the
    # wire ``num_output_tokens`` (which INCLUDES async/spec placeholders),
    # so the keep condition must use the same placeholder-inclusive count.
    # NOTE: filtering by prev_dispatch_req_ids proved unsafe — interleaved
    # prefill layer-slices and placeholder accounting can evict a request
    # from the worker's persistent batch even when it appeared in the
    # previous dispatch, and a missing entry crashes the worker with a
    # KeyError. Prefer bandwidth over that risk.
    keep_req_ids = {
        req_id
        for req_id in all_token_ids
        if req_id in resumed_req_ids
        or num_output_tokens_by_req.get(req_id, 0) > 0
    }
    # NOTE: entries that are kept must carry the FULL token list
    # (prompt + outputs). The worker-side recovery path
    # (gpu_model_runner._update_states) reconstructs output_token_ids by
    # slicing ``all_token_ids[-num_output_tokens:]`` and therefore assumes
    # the array is the complete token history. Truncating the arrays here
    # (e.g. to the last num_output_tokens) silently drops the prompt, makes
    # the recovered output history empty or polluted with prompt tokens on
    # the cloud side, and shows up as garbled decode / repeated tokens once
    # a second request triggers a persistent-batch rebuild.
    trimmed_all_token_ids = {
        req_id: token_ids
        for req_id, token_ids in all_token_ids.items()
        if req_id in keep_req_ids
    }
    if len(trimmed_all_token_ids) == len(all_token_ids) and all(
        len(trimmed_all_token_ids[req_id]) == len(token_ids)
        for req_id, token_ids in all_token_ids.items()
    ):
        return scheduler_output

    before_tokens = sum(len(token_ids) for token_ids in all_token_ids.values())
    after_tokens = sum(
        len(token_ids) for token_ids in trimmed_all_token_ids.values()
    )
    # logger.info(
    #     "[CLOUD-MQ-TRIM] batch_type=%s reqs=%d prev_dispatch_reqs=%d "
    #     "resumed=%d all_token_ids entries %d->%d tokens %d->%d",
    #     scheduler_output.batch_type.value,
    #     len(getattr(cached, "req_ids", ())),
    #     len(prev_dispatch_req_ids),
    #     len(resumed_req_ids),
    #     len(all_token_ids),
    #     len(trimmed_all_token_ids),
    #     before_tokens,
    #     after_tokens,
    # )

    so_copy = copy.copy(scheduler_output)
    cached_copy = copy.copy(cached)
    cached_copy.all_token_ids = trimmed_all_token_ids
    so_copy.scheduled_cached_reqs = cached_copy
    return so_copy


class PassiveEngineCoreProc:
    """Passive EngineCore process for non-leader PP ranks.

    Mirrors the `EngineCore` / `EngineCoreProc` shape on rank0:

    - `step()` is the single-tick action: poll the ZMQ inbox, ask the
      `PassiveScheduler` for one batch, fan its slice plan out to the
      worker `rpc_broadcast_mq`.
    - `run_busy_loop()` is the long-running driver that keeps calling
      `step()` until the executor reports failure.

    Unlike rank0, there is no local scheduling decision — every batch
    comes pre-decided over the cloud-side scheduler input. The static
    :py:meth:`run_passive_engine_core` is the process entry point that
    constructs the executor + input channel, builds an instance, and hands
    off to `run_busy_loop`.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        executor,  # MultiprocExecutor — duck-typed to avoid heavy import
        scheduler_input,
        dispatch_policy=None,
        pp_pd_channel: Optional["PPSchedulerZmqChannel"] = None,
    ) -> None:
        passive_scheduler_module = _import_passive_scheduler_module()
        if dispatch_policy is None:
            dispatch_policy = (
                passive_scheduler_module.DispatchPolicy.EXPECT_ALTERNATION
            )
        self.vllm_config = vllm_config
        self.executor = executor
        # scheduler_input is either a single channel (1:1 / legacy) or a
        # dict[edge_id, channel] (multi-edge). Normalise to a per-edge
        # dict so the rest of the class is edge-agnostic. Each channel is
        # both the PRE_OUT subscriber (consume_new_outputs) and the
        # POST_OUT publisher (publish).
        if isinstance(scheduler_input, dict):
            self._channels: dict[int, Any] = dict(scheduler_input)
        else:
            self._channels = {0: scheduler_input}
        self._session_order: list[int] = sorted(self._channels)
        # One PassiveScheduler per edge (independent EXPECT_ALTERNATION
        # state machine). ``self.passive_scheduler`` points at the
        # "current" edge's scheduler, swapped by step(edge_id) during
        # round-robin (time-division, no cross-edge batch merging).
        self._sessions: dict[int, Any] = {
            eid: passive_scheduler_module.PassiveScheduler(
                vllm_config, ch, dispatch_policy=dispatch_policy)
            for eid, ch in self._channels.items()
        }
        self._current_edge_id: int = self._session_order[0]
        self.passive_scheduler = self._sessions[self._current_edge_id]
        # POST_OUT (cloud → edge) channel of the "current" edge, for
        # legacy code paths that read self._pp_pd_channel directly.
        self._pp_pd_channel = self._channels.get(self._current_edge_id)
        if getattr(vllm_config.parallel_config, "enable_edge_cloud", False):
            # PassiveEngineCore runs in a freshly-spawned subprocess; the
            # ``_ASCEND_CONFIG`` singleton may be empty here. ``init_ascend_config``
            # is idempotent and returns the cached singleton if already set.
            from vllm_ascend.ascend_config import init_ascend_config
            _ascend_config = init_ascend_config(vllm_config)
            _edge_cloud = getattr(_ascend_config, "edge_cloud_config", None)
            _pd_enabled = bool(
                _edge_cloud is not None
                and getattr(_edge_cloud, "enabled", False)
                and getattr(_edge_cloud, "pd_separation", None) is not None
                and _edge_cloud.pd_separation.enabled
            )
            logger.info(
                "PassiveEngineCore: edge-cloud mode enabled "
                "(pd_separation=%s, pd_channel=%s)",
                _pd_enabled,
                "on" if pp_pd_channel is not None else "off",
            )
            # [CHER] Cloud-side hidden early-receive: a built-in part of
            # PD-separation masking -- always active on the cloud role when
            # PD-separation is enabled (no separate flag).  step() fires a
            # recv-hint to the cloud worker's sideband cloud_recv_hint_mq so
            # the guard thread posts irecv ahead of execute_model.  Read from
            # vllm_config (parallel_config + additional_config dict, both
            # serialized fields) so the gate does not depend on ascend_config
            # singleton init order or on dynamic scheduler_config attributes.
            _pc = vllm_config.parallel_config
            _ac = getattr(vllm_config, "additional_config", None) or {}
            _ec = _ac.get("edge_cloud_config", {}) if isinstance(_ac, dict) else {}
            _pd = _ec.get("pd_separation", {}) if isinstance(_ec, dict) else {}
            self._cher_enabled = bool(
                getattr(_pc, "enable_edge_cloud", False)
                and not getattr(_pc, "is_edge_node", True)
                and _pd.get("enabled", False)
            )
            # Track which head_tokens we have already sent a hint for, so
            # layer-slicing's multiple first-slice steps fire it only once.
            self._cher_hint_sent: set[str] = set()
        else:
            self._cher_enabled = False
            self._cher_hint_sent = set()
        self._idle_sleep_seconds = 0.001

        # Per-edge "previously dispatched req_ids" so the
        # _trim_scheduler_output optimiser does not conflate requests
        # from different edges. self._prev_dispatch_req_ids is the
        # "current" edge's set (swapped by step(edge_id)).
        self._prev_dispatch_req_ids_per_edge: dict[int, set[str]] = {
            eid: set() for eid in self._channels}
        self._prev_dispatch_req_ids: set[str] = (
            self._prev_dispatch_req_ids_per_edge[self._current_edge_id])
        self._pending_post_out_by_head_token: dict[str, SchedulerOutput] = {}
        self._published_post_out_tokens: set[str] = set()
        # Multi-edge KV isolation: the cloud offsets each edge's block
        # IDs into its global KV pool by edge_id * (cloud_num_blocks //
        # num_edges). cloud_num_blocks is read from env (set to the
        # cloud worker's profiled num_blocks); 0 disables offset.
        from vllm_ascend import envs as _envs_ascend
        self._num_edges: int = getattr(
            vllm_config.parallel_config, "num_edges", 1)
        self._cloud_num_blocks: int = (
            _envs_ascend.VLLM_ASCEND_EDGE_CLOUD_CLOUD_NUM_BLOCKS)

    def _offset_worker_so_block_ids(
        self, worker_so: SchedulerOutput, edge_id: int,
    ) -> None:
        """Multi-edge KV isolation: shift this edge's block IDs into the
        cloud's global KV pool so two edges don't collide.

        Operates on the *worker copy* (``worker_scheduler_output``)
        only; the echoed original (``batch.scheduler_output``) keeps
        local block IDs so the edge tail segment indexes its own small
        KV pool correctly. No restore is needed anywhere.
        """
        if self._num_edges <= 1 or self._cloud_num_blocks <= 0:
            return
        stride = self._cloud_num_blocks // self._num_edges
        offset = edge_id * stride
        if offset == 0:
            return
        assert offset + stride <= self._cloud_num_blocks, (
            f"edge_id={edge_id} KV offset {offset}+{stride} exceeds "
            f"cloud_num_blocks={self._cloud_num_blocks}; raise "
            "VLLM_ASCEND_EDGE_CLOUD_CLOUD_NUM_BLOCKS")
        _shift = lambda b: b + offset if b >= 0 else b

        # NewRequestData.block_ids: tuple[list[int], ...] (a req's full
        # block list, grouped by kv_cache_group). tuple is immutable.
        for nrd in getattr(worker_so, "scheduled_new_reqs", None) or []:
            nrd.block_ids = tuple(
                [_shift(b) for b in g] for g in nrd.block_ids)
        # CachedRequestData.new_block_ids: list[tuple[list,...] | None]
        # (newly allocated blocks this step, per req per group).
        cached = getattr(worker_so, "scheduled_cached_reqs", None)
        if cached is not None and getattr(cached, "new_block_ids", None):
            cached.new_block_ids = [
                (tuple([_shift(b) for b in g] for g in t)
                 if t is not None else None)
                for t in cached.new_block_ids]
        # SchedulerOutput.new_block_ids_to_zero: list[int] | None.
        nbz = getattr(worker_so, "new_block_ids_to_zero", None)
        if nbz:
            worker_so.new_block_ids_to_zero = [_shift(b) for b in nbz]

    def _drain_worker_completion_acks(self) -> None:
        """Publish POST_OUT only after cloud workers complete the middle segment."""
        for mq in getattr(self.executor, "response_mqs", []):
            while True:
                try:
                    _status, result = mq.dequeue(timeout=0)
                except TimeoutError:
                    break
                except Exception:
                    logger.exception("Failed to drain cloud worker completion ack")
                    break

                if not (
                    isinstance(result, dict)
                    and result.get("__pp_scheduler_ack__")
                ):
                    continue

                if result.get("batch_type") not in (
                    BatchType.PREFILL_FIRST,
                    BatchType.DRAFT_FIRST,
                ):
                    continue
                head_token = result.get("head_token")
                if not head_token or head_token in self._published_post_out_tokens:
                    continue
                scheduler_output = self._pending_post_out_by_head_token.pop(
                    head_token, None
                )
                if scheduler_output is None:
                    continue
                # NOTE: do NOT add head_token to _published_post_out_tokens
                # here — _maybe_publish_post_out records it after actually
                # publishing (single idempotency point for both PL and DL).
                # logger.info(
                #     "[CLOUD-POST-OUT] Publishing PREFILL_LAST after worker done, "
                #     "head_token=%s",
                #     head_token,
                # )
                self._maybe_publish_post_out(scheduler_output)

    def step(self, edge_id: int | None = None) -> bool:
        """Single tick for one edge: poll ZMQ → pick batch → enqueue.

        In multi-edge mode run_busy_loop calls step(edge_id) round-robin
        (time-division, no merging). ``edge_id`` selects which edge's
        PassiveScheduler / POST_OUT channel / prev-dispatch set is active.

        Returns:
            True if at least one payload was enqueued, False if the
            scheduler had nothing to dispatch.
        """
        if edge_id is None:
            edge_id = self._current_edge_id
        # Switch the "current" edge context so the step body (which reads
        # self.passive_scheduler / self._pp_pd_channel /
        # self._prev_dispatch_req_ids) is edge-agnostic.
        self._current_edge_id = edge_id
        self.passive_scheduler = self._sessions[edge_id]
        self._pp_pd_channel = self._channels.get(edge_id)
        self._prev_dispatch_req_ids = (
            self._prev_dispatch_req_ids_per_edge[edge_id])

        _t0 = time.monotonic()
        self.passive_scheduler.poll_and_classify()
        _dt_poll = (time.monotonic() - _t0) * 1000

        _t0 = time.monotonic()
        batch = self.passive_scheduler.schedule()
        _dt_sched = (time.monotonic() - _t0) * 1000

        if batch.is_empty():
            if _dt_poll > 1.0 or _dt_sched > 1.0:
                logger.info(
                    "[CLOUD-STEP-EMPTY] poll=%.3f ms, schedule=%.3f ms "
                    "(edge_id=%d)",
                    _dt_poll, _dt_sched, edge_id,
                )
            return False

        _slice_info_str = "["
        for s in batch.slices:
            if s is not None:
                _slice_info_str += (
                    f"slice_index={s.slice_index},"
                    f"start={s.start_layer},"
                    f"end={s.end_layer},"
                    f"is_last={s.is_last_slice};"
                )
            else:
                _slice_info_str += "None;"
        _slice_info_str += "]"
        # logger.info(
        #     f"\r\n[Cloud] Step dispatched batch_type: "
        #     f"{batch.scheduler_output.batch_type}, "
        #     f"slices_count={len(batch.slices)}, "
        #     f"slice_info={_slice_info_str}",
        # )

        # [CHER] Fire a recv-hint so the cloud worker's guard thread posts
        # the edge->cloud prefill hidden irecv ahead of this batch's
        # execute_model.  Only for the first slice of a PREFILL_FIRST batch:
        # the hidden transfer is initiated by the edge P-head and consumed
        # by the cloud P-middle's first slice; later slices reuse the same
        # intermediate tensors and must not re-post.  Sent BEFORE the
        # pp_scheduler_output enqueue: the sideband MQ is independent of the
        # (possibly back-pressured) rpc_broadcast_mq, so the hint is never
        # delayed by pp_scheduler_output's enqueue even when busy_loop is
        # blocked.  No gating: schedule() ran already and P-middle is being
        # dispatched regardless; the hint only decides *when* the irecv is
        # posted, not whether P-middle runs.
        so = batch.scheduler_output
        if (
            self._cher_enabled
            and so.batch_type == BatchType.PREFILL_FIRST
            and getattr(so, "head_token", None)
        ):
            _is_first_slice = (
                not batch.slices
                or batch.slices[0] is None
                or getattr(batch.slices[0], "is_first_slice", True)
            )
            _ht = so.head_token
            if _is_first_slice and _ht not in self._cher_hint_sent:
                _channel = getattr(so, "hidden_channel", None)
                _hint = {
                    "head_token": _ht,
                    "hidden_channel": (
                        _channel.value if _channel is not None else None
                    ),
                    "num_tokens": so.total_num_scheduled_tokens,
                    # has_mrope is stamped by the edge PDSeparatedScheduler
                    # (it owns the request registry; the passive cloud does
                    # not - scheduled_cached_reqs carries only req_ids, so
                    # cached-req multimodality cannot be derived from the SO
                    # alone here). The stamp mirrors NPUModelRunner.
                    # step_has_multimodal_req exactly, so the guard-thread
                    # irecv expects exactly the mrope_positions the edge sender
                    # puts on the wire (eliminates the mixed-batch mismatch).
                    # Defaults True when unset (non-PD / no stamp) so mrope is
                    # received conservatively.
                    "has_mrope": getattr(so, "has_mrope", True),
                }
                _hint_mq = getattr(self.executor, "cloud_recv_hint_mq", None)
                if _hint_mq is not None:
                    try:
                        # Non-blocking (timeout=0): the hint is fire-and-forget.
                        # If the guard thread hasn't drained the sideband MQ
                        # (slow / contended on _early_recv_lock), we DROP the
                        # hint rather than block PassiveEC.step() here -- a
                        # blocked step can't drain acks, which fills
                        # response_mq, which blocks the worker's ack enqueue,
                        # which stops it from dequeuing rpc_broadcast_mq, which
                        # blocks PassiveEC's dispatch -> circular deadlock.
                        # When a hint is dropped, busy_loop's get_or_post_early
                        # _recv posts the irecv itself (synchronous), so only
                        # the early-post overlap is lost, never correctness.
                        _hint_mq.enqueue(
                            (b"pp_recv_hint", (_hint,), {}, None),
                            timeout=0,
                        )
                        self._cher_hint_sent.add(_ht)
                        logger.debug(
                            "[CHER] send recv-hint head_token=%s channel=%s",
                            _ht, _hint["hidden_channel"],
                        )
                    except TimeoutError:
                        logger.warning(
                            "[CHER] recv-hint dropped (ring full) "
                            "head_token=%s; busy_loop will post irecv itself",
                            _ht,
                        )
                    except Exception as _e:
                        logger.warning(
                            "[CHER] recv-hint dropped (error=%r) "
                            "head_token=%s; busy_loop will post irecv itself",
                            _e, _ht,
                        )

        for slice_info in batch.slices:
            _t0 = time.monotonic()
            worker_scheduler_output = _trim_scheduler_output_for_worker_enqueue(
                batch.scheduler_output,
                self._prev_dispatch_req_ids,
            )
            # Multi-edge KV isolation: offset this edge's block IDs into
            # the cloud's global KV pool (worker copy only; the echoed
            # original keeps local IDs for the edge tail segment).
            self._offset_worker_so_block_ids(worker_scheduler_output, edge_id)
            _dt_trim = (time.monotonic() - _t0) * 1000

            payload = (
                (worker_scheduler_output, slice_info)
                if slice_info is not None
                else (worker_scheduler_output,)
            )
            bt = batch.scheduler_output.batch_type.value
            # logger.info("[CLOUD-MQ] About to enqueue batch_type=%s", bt)
            _t0 = time.monotonic()
            self.executor.rpc_broadcast_mq.enqueue(
                (b"pp_scheduler_output", payload, {}, None)
            )
            self._prev_dispatch_req_ids = set(
                batch.scheduler_output.num_scheduled_tokens.keys()
            )
            # _dt_enqueue = (time.monotonic() - _t0) * 1000
            # if _dt_trim > 0.5 or _dt_enqueue > 0.5:
            #     logger.info(
            #         "[CLOUD-STEP] trim=%.3f ms, enqueue=%.3f ms, batch_type=%s, "
            #         "drain=%.3f ms, poll=%.3f ms, schedule=%.3f ms",
            #         _dt_trim, _dt_enqueue, bt,
            #         _dt_drain, _dt_poll, _dt_sched,
            #     )
            # else:
            #     logger.info(
            #         "[CLOUD-ENQUEUE] %s enqueue took %.3f ms",
            #         bt,
            #         _dt_enqueue,
            #     )
            # For prefill and draft, POST_OUT must mean the cloud middle
            # segment has completed. Store the original SchedulerOutput here
            # and publish it from _drain_worker_completion_acks() after the
            # worker reports done. Decode-last is prepared on the edge.
            if (
                batch.scheduler_output.batch_type
                in (BatchType.PREFILL_FIRST, BatchType.DRAFT_FIRST)
                and (slice_info is None or slice_info.is_last_slice)
            ):
                head_token = getattr(batch.scheduler_output, "head_token", None)
                if head_token:
                    self._pending_post_out_by_head_token[head_token] = (
                        batch.scheduler_output
                    )
        return True

    def _maybe_publish_post_out(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        """Rewrite + publish a head-segment batch as a tail-segment one
        on the POST_OUT (cloud → edge) channel.

        Mapping (cloud-side):
            PREFILL_FIRST → PREFILL_LAST
            DECODE_FIRST  → dropped (edge prepares DECODE_LAST)
            DRAFT_FIRST   → DRAFT_LAST
            anything else → dropped (legacy PP batches don't trigger return)

        Uses a shallow copy via :py:func:`dataclasses.replace` so the original
        SchedulerOutput (still about to be enqueued for the local executor)
        keeps its head-segment ``batch_type``.
        """
        # Multi-edge: route POST_OUT to the channel of the SO's edge_id
        # (each edge has its own POST_OUT ZMQ channel).
        _edge_id = getattr(
            scheduler_output, "edge_id", self._current_edge_id)
        _ch = self._channels.get(_edge_id)
        if _ch is None:
            return
        from dataclasses import replace
        bt = scheduler_output.batch_type
        if bt == BatchType.PREFILL_FIRST:
            tail = replace(
                scheduler_output, batch_type=BatchType.PREFILL_LAST
            )
        elif bt == BatchType.DECODE_FIRST:
            # The edge pre-generates DECODE_LAST, so the cloud does not
            # publish another control-plane response for DECODE_FIRST.
            logger.debug(
                "[Cloud] Skipping POST_OUT for DECODE_FIRST "
                "head_token=%s (edge pre-generates DECODE_LAST)",
                scheduler_output.head_token,
            )
            return
        elif bt == BatchType.DRAFT_FIRST:
            tail = replace(
                scheduler_output, batch_type=BatchType.DRAFT_LAST
            )
            if not tail.head_token:
                raise RuntimeError("DRAFT_LAST POST_OUT missing head_token")
            if not tail.draft_task_id:
                raise RuntimeError(
                    "DRAFT_LAST POST_OUT missing draft_task_id"
                )
            if tail.draft_step_idx is None:
                raise RuntimeError(
                    "DRAFT_LAST POST_OUT missing draft_step_idx"
                )
        else:
            return
        # Idempotency guard: publishing the same head_token twice would make
        # the edge process the tail segment twice. A duplicated DECODE_LAST
        # output subtracts num_output_placeholders a second time and drives
        # it negative (fatal assert in AsyncScheduler._update_request_with_output).
        # PREFILL_LAST is additionally gated by the worker-ack path; the
        # guard is harmless there and essential for DL.
        head_token = getattr(tail, "head_token", None)
        if head_token:
            if head_token in self._published_post_out_tokens:
                logger.warning(
                    "[CLOUD-POST-OUT] Suppressing duplicate %s publish for "
                    "head_token=%s",
                    tail.batch_type,
                    head_token,
                )
                return
            self._published_post_out_tokens.add(head_token)
        # Echo the head_token back so the edge can correlate the tail
        # segment with its suspended head state.
        _ch.publish(tail)

    def run_busy_loop(self) -> None:
        """Drive step() round-robin across edges until failure/shutdown."""
        try:
            while not self.executor.is_failed:
                # Drain worker completions once per round (not per edge):
                # acks are keyed by head_token (globally unique), so a
                # single pass publishes all edges' ready POST_OUTs.
                self._drain_worker_completion_acks()
                progressed = False
                for edge_id in self._session_order:
                    if self.step(edge_id):
                        progressed = True
                if not progressed:
                    time.sleep(self._idle_sleep_seconds)
        finally:
            for sched in self._sessions.values():
                sched.shutdown()

    @staticmethod
    def run_passive_engine_core(
        vllm_config: "VllmConfig",
        ready_pipe,  # multiprocessing.Connection for signaling readiness
    ):
        """Entry point for the passive EngineCore process.

        Creates a MultiprocExecutor to spawn workers, wires up the
        cloud-side PD-separation channel as the scheduler input when PD
        separation is enabled, then hands off to
        `PassiveEngineCoreProc.run_busy_loop`.
        """
        # Imported lazily so the patched-by-vllm-ascend
        # ``MultiprocExecutor`` (= ``AscendMultiprocExecutor``) is the one
        # we instantiate when this code runs in a child process.
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor

        maybe_register_config_serialize_by_value()

        # Mark this process as a non-leader PP rank running with passive
        # EngineCore, so that AscendMultiprocExecutor and AscendWorkerProc
        # set up dual message queues (local + cross-node).
        os.environ["VLLM_PP_NON_LEADER_ENGINE_CORE"] = "1"
        envs.disable_envs_cache()

        set_process_title("PassiveEngineCore")
        maybe_init_worker_tracer(
            "vllm.engine_core", "engine_core", "PassiveEngineCore"
        )
        decorate_logs()

        # Cloud-side PD-separation channel is constructed inside the try
        # block below (depends on `vllm_config`); declared here so the
        # `finally` clean-up can reference it unconditionally.
        pp_pd_channel: Optional[PPSchedulerZmqChannel] = None

        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        executor = None
        try:
            executor = MultiprocExecutor(vllm_config, monitor_workers=False)

            ready_pipe.send({"status": "READY"})
            ready_pipe.close()
            ready_pipe = None

            passive_scheduler_module = _import_passive_scheduler_module()
            dispatch_policy_cls = passive_scheduler_module.DispatchPolicy
            # Load PD-separation configuration from environment variables.
            from vllm_ascend.pd_separation_config import PDSeparationConfig
            pd_config = PDSeparationConfig.from_env()
            try:
                policy = dispatch_policy_cls(pd_config.dispatch_policy)
            except ValueError:
                logger.warning(
                    "Unknown VLLM_PP_PASSIVE_DISPATCH_POLICY=%r; "
                    "falling back to expect_alternation.",
                    pd_config.dispatch_policy,
                )
                policy = dispatch_policy_cls.EXPECT_ALTERNATION

            scheduler_input = None

            # Set up edge-cloud PD-separation channel (cloud side). The
            # cloud binds POST_OUT and connects PRE_OUT via master_addr
            # (the edge's IP) so PRE_OUT connects back.
            #
            # PassiveEngineCore runs in a freshly-spawned subprocess where
            # the ``_ASCEND_CONFIG`` singleton is empty; re-init from the
            # ``vllm_config`` we were handed. ``init_ascend_config`` is
            # idempotent on the singleton.
            from vllm_ascend.ascend_config import init_ascend_config
            _ascend_config = init_ascend_config(vllm_config)
            _edge_cloud = getattr(_ascend_config, "edge_cloud_config", None)
            _pd_enabled = bool(
                _edge_cloud is not None
                and getattr(_edge_cloud, "enabled", False)
                and getattr(_edge_cloud, "pd_separation", None) is not None
                and _edge_cloud.pd_separation.enabled
            )
            if _pd_enabled:
                master_port = vllm_config.parallel_config.master_port
                import torch.distributed as dist
                from datetime import timedelta
                from vllm.utils.network_utils import get_ip
                _cloud_ip = get_ip()
                # Multi-edge: the cloud fans in N edges. Each edge has
                # its own master_addr; num_edges==1 falls back to the
                # single master_addr. One PPSchedulerZmqChannel per edge
                # (ZMQ ports offset by edge_idx*2; TCPStore on
                # master_port+1+edge_idx, edge=master/cloud=client).
                _num_edges = getattr(
                    vllm_config.parallel_config, "num_edges", 1)
                if _num_edges > 1:
                    from vllm_ascend import envs as _envs_ascend
                    _addrs = [
                        a.strip() for a in
                        _envs_ascend.VLLM_ASCEND_EDGE_CLOUD_MASTER_ADDRS.split(
                            ",") if a.strip()]
                    if len(_addrs) != _num_edges:
                        raise RuntimeError(
                            f"num_edges={_num_edges} but "
                            f"VLLM_ASCEND_EDGE_CLOUD_MASTER_ADDRS has "
                            f"{len(_addrs)} addresses")
                else:
                    _addrs = [vllm_config.parallel_config.master_addr]
                scheduler_input: dict[int, Any] = {}
                for _edge_idx, _master_addr in enumerate(_addrs):
                    _addr_store = dist.TCPStore(
                        host_name=_master_addr,
                        port=master_port + 1 + _edge_idx,
                        world_size=2,
                        is_master=False,
                        timeout=timedelta(seconds=300),
                    )
                    _addr_store.set("cloud_ip", _cloud_ip)
                    del _addr_store
                    _pre_out_port = pd_config.pre_out_port + _edge_idx * 2
                    _post_out_port = pd_config.post_out_port + _edge_idx * 2
                    scheduler_input[_edge_idx] = PPSchedulerZmqChannel(
                        send_endpoint=f"tcp://*:{_post_out_port}",
                        recv_endpoint=f"tcp://{_master_addr}:{_pre_out_port}",
                        name=f"pd-cloud-edge{_edge_idx}",
                    )
                logger.info(
                    "PD-separation cloud channels: %d edge(s) "
                    "(pre_out_base=%s, post_out_base=%s)",
                    len(scheduler_input),
                    pd_config.pre_out_port, pd_config.post_out_port,
                )

            if scheduler_input is not None:
                executor.start_worker_monitor(inline=False)
                proc = PassiveEngineCoreProc(
                    vllm_config, executor, scheduler_input,
                    dispatch_policy=policy,
                )
                proc.run_busy_loop()
            else:
                # No scheduler input, just monitor workers inline.
                executor.start_worker_monitor(inline=True)

        except SystemExit:
            logger.debug("PassiveEngineCore exiting.")
        except Exception:
            logger.exception("PassiveEngineCore encountered a fatal error.")
            raise
        finally:
            if ready_pipe is not None:
                try:
                    ready_pipe.send({"status": "FAILED"})
                except Exception:
                    pass
                ready_pipe.close()
            if pp_pd_channel is not None:
                pp_pd_channel.shutdown()
            if executor is not None:
                executor.shutdown()
