# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exercise PD ownership and mixed-batch contracts without loading the NPU plugin."""

import ast
import importlib.util
import math
import sys
import time
from collections import deque, namedtuple
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from typing import cast
from unittest.mock import Mock, patch

import msgspec
import numpy as np
import pytest
import torch

from .test_lwd_v4_mixed_migration import RUNNER, load_methods, routing

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend"


def module_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tail = module_file("pd_tail_buffer", "worker/lwd_cloud_model/lwd_tail_buffer.py")
decode = module_file("pd_decode_state", "worker/lwd_cloud_model/lwd_decode_state.py")


def source_method(path, class_name, method_name, **scope):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    exec(compile(module, str(path), "exec"), scope)
    return scope[method_name]


def vllm_source():
    source = ROOT.parents[1] / "vllm/vllm"
    if not source.is_dir():
        pytest.skip("cross-repository regression requires the sibling vLLM source checkout")
    return source


def test_native_batch_queue_executes_zero_token_finished_flush_and_releases_edge_media():
    source = vllm_source()
    empty = lambda: NS(lwd_batch=None, total_num_scheduled_tokens=0, finished_req_ids=set())
    output_type = NS(make_empty=empty)
    scheduler_source = source / "v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py"
    pending = source_method(scheduler_source, "LwdEdgeScheduler", "has_requests")
    unembed = source_method(scheduler_source, "LwdEdgeScheduler", "_lwd_schedule_unembed", SchedulerOutput=output_type)
    scheduler = NS(
        unembed_notify_queue=deque(),
        finished_req_ids={"pd"},
        _lwd_pending_refills=[],
        lwd_has_pending=None,
        _lwd_aborted_req_ids={},
        _lwd_accounts={},
        _lwd_embed_dispatchable=lambda: False,
    )
    scheduler.has_requests = lambda: pending(scheduler)
    scheduler.schedule = lambda _: unembed(scheduler)
    scheduler.update_from_output = Mock(return_value={})
    worker_type = load_methods("worker/lwd_edge_model/lwd_edge_worker.py", {"execute_model"}, lwd_logger=Mock())
    worker = worker_type()
    worker.comm_service = NS(poll_completions=Mock())
    worker.model_runner = NS(
        mm_cache=NS(release_finished=Mock()),
        _lwd_watermarked_output=lambda: NS(req_ids=[], sampled_token_ids=[], lwd_up_send_watermarks={}),
    )
    execute = source_method(
        source / "v1/executor/multiproc_executor.py",
        "MultiprocExecutor",
        "execute_model",
        envs=NS(VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10),
    )

    def rpc(method, *, args, **kwargs):
        assert method == "execute_model"
        result = Future()
        result.set_result(worker.execute_model(args[0]))
        return result

    executor = NS(collective_rpc=rpc, output_rank=0, kv_output_aggregator=None, ec_output_aggregator=None)
    executor.execute_model = lambda output, **kw: execute(executor, output, **kw)
    executor.sample_tokens = Mock(side_effect=AssertionError("edge flush must not sample"))
    step = source_method(
        source / "v1/engine/core.py",
        "EngineCore",
        "step_with_batch_queue",
        cast=cast,
        Future=Future,
        ModelRunnerOutput=object,
    )
    engine = NS(
        scheduler=scheduler,
        model_executor=executor,
        batch_queue=deque(),
        batch_queue_size=2,
        is_ec_consumer=False,
        is_pooling_model=True,
        _should_throttle_prefills=lambda: False,
        log_error_detail=lambda _: nullcontext(),
        capture_iteration_details=lambda _: nullcontext(),
        _process_aborts_queue=Mock(),
        _attach_iteration_details=Mock(),
    )
    assert step(engine) == ({}, False)
    worker.model_runner.mm_cache.release_finished.assert_called_once_with({"pd"})
    assert not scheduler.has_requests() and not engine.batch_queue


def scheduler_output(**changes):
    output = NS(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        num_scheduled_tokens={},
        scheduled_cached_reqs=NS(resumed_req_ids=set(), req_ids=[], num_computed_tokens=[]),
    )
    output.__dict__.update(changes)
    return output


def test_d_tail_copies_actual_rows_and_rejects_foreign_or_retired_binding():
    worker_type = load_methods(
        "worker/lwd_cloud_model/lwd_cloud_worker.py",
        {"_lwd_buffer_received_tails"},
        LwdTailInput=tail.LwdTailInput,
    )
    worker = worker_type()
    worker._lwd_tail_buffer = tail.LwdTailBuffer()
    worker._lwd_up_hidden_size = 4
    worker._lwd_up_num_hash_layers = 1
    worker._lwd_up_num_experts_per_tok = 1
    key = ("connection", "up", 3)
    worker._lwd_tail_recv_fields = {
        key: NS(
            connection="connection",
            req_ids=["a", "b"],
            rows=[1, 1],
            prompt_offsets=[1023, 8191],
            num_tokens=2,
            binding_ids=["ba", "bb"],
            attempt_ids=["aa", "ab"],
        )
    }
    source = torch.arange(16, dtype=torch.bfloat16)
    npu = NS(Event=Mock, current_stream=Mock())
    with patch.object(torch, "npu", npu, create=True), patch.object(torch.Tensor, "record_stream"):
        assert worker._lwd_buffer_received_tails(key, source) == [("a", 1, 1023), ("b", 1, 8191)]
    source.zero_()
    with pytest.raises(RuntimeError, match="identity"):
        worker._lwd_tail_buffer.take("a", "foreign", "aa", 1023, 1)
    value = worker._lwd_tail_buffer.take("a", "ba", "aa", 1023, 1)
    assert value.payload.numel() == 8
    assert value.payload.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert value.payload.tolist() == [0, 1, 2, 3, 8, 9, 10, 11]
    worker._lwd_tail_buffer.retire("b", "bb", "ab")
    old = tail.LwdTailInput("connection", "bb", "ab", 8191, 1, value.payload, value.ready_event)
    assert not worker._lwd_tail_buffer.put("b", old)
    newer = tail.LwdTailInput("connection", "new-binding", "new-attempt", 8191, 1, value.payload, value.ready_event)
    assert worker._lwd_tail_buffer.put("b", newer)
    with pytest.raises(RuntimeError, match="not ready"):
        worker._lwd_tail_buffer.take("b", "bb", "ab", 8191, 1)


def test_d_submit_skips_full_prompt_staging_and_accepts_only_the_planned_tail():
    channel = NS(UP="up")
    data_plane = NS(submit_recv=Mock(return_value="future"))
    worker_type = load_methods(
        "worker/lwd_cloud_model/lwd_cloud_worker.py",
        {"_lwd_submit_early_recv", "_lwd_build_up_recv_request"},
        hash_payload_numel=routing.hash_payload_numel,
        LwdTransferRequest=lambda **kw: NS(**kw),
        LwdChannelType=channel,
        lwd_get_data_plane=lambda: data_plane,
    )
    worker = worker_type()
    worker._lwd_pd_decode = True
    worker._lwd_stager = NS(allocate=Mock(side_effect=AssertionError("D must not allocate P staging")))
    worker._lwd_up_hidden_size, worker._lwd_up_num_hash_layers, worker._lwd_up_num_experts_per_tok = 4, 0, 0
    worker._lwd_tail_recv_fields = {}
    worker.device = "cpu"
    connection = namedtuple("Connection", "edge_id cloud_id dp_idx")(0, 1, 3)
    fields = NS(
        connection=connection,
        seqno=0,
        num_tokens=1,
        aux_rows=0,
        req_ids=["r"],
        rows=[1],
        prompt_lens=[8192],
        prompt_offsets=[8191],
        binding_ids=["b"],
        attempt_ids=["a"],
    )
    assert worker._lwd_submit_early_recv(fields) == "future"
    request = data_plane.submit_recv.call_args.args[0]
    assert request.recv_buffer.numel() == request.num_elements == 4
    assert request.stage_segments == []
    fields.prompt_offsets = [0]
    with pytest.raises(ValueError, match="final prompt row"):
        worker._lwd_submit_early_recv(fields)


@pytest.mark.parametrize("valid", [[0, 0], [1, 0]])
def test_edge_drains_full_down_frame_but_only_recovers_valid_rows(valid):
    source = torch.arange(10).reshape(5, 2).float()
    future = NS(wait_on_current_stream=Mock(), result=lambda: NS(tensor=source))
    compute = Mock(return_value=torch.ones(sum(valid), 4))
    runner_type = load_methods(
        "worker/lwd_edge_model/lwd_edge_model_runner.py",
        {"execute_lwd_unembed"},
        time=time,
        lwd_logger=Mock(),
        connection_from=lambda _: NS(edge_id=0, cloud_id=1, dp_idx=3),
        LwdTransferRequest=lambda **kw: NS(**kw),
        LwdChannelType=NS(DOWN="down"),
        select_token_batch=lambda logits, ranks: [99] * len(ranks),
    )
    runner = runner_type()
    runner.get_model = lambda: NS(compute_logits=compute)
    runner.model_config = NS(get_hidden_size=lambda: 2)
    runner.worker = NS(comm_service=NS(submit_recv=Mock(return_value=future)))
    runner._lwd_watermarked_output = lambda: NS()
    batch = NS(
        req_ids=["a", "b"],
        recv_num_elements=10,
        physical_rows=[3, 2],
        valid_rows=valid,
        top_id_ths=[[0] * count for count in valid],
        num_accept_tokens=valid,
    )
    with (
        patch.object(torch, "npu", NS(current_stream=Mock()), create=True),
        patch.object(torch.Tensor, "record_stream"),
    ):
        output = runner.execute_lwd_unembed(7, batch)
    assert runner.worker.comm_service.submit_recv.call_args.args[0].num_elements == 10
    assert output.sampled_token_ids == [[99] * count for count in valid]
    if any(valid):
        torch.testing.assert_close(compute.call_args.args[0], source[:1])
    else:
        compute.assert_not_called()


@pytest.mark.parametrize("committed", [0, 1, 2])
def test_cloud_physical_carrier_to_edge_recovery_respects_native_commit_prefix(committed):
    source = vllm_source()
    spec = importlib.util.spec_from_file_location("pd_runtime_wire", source / "v1/lwd_control/lwd_comm/lwd_messages.py")
    wire = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wire
    spec.loader.exec_module(wire)
    scheduler_source = source / "v1/lwd_control/control_cloud_scheduler/lwd_cloud_scheduler.py"
    unpack = source_method(scheduler_source, "LwdCloudScheduler", "_lwd_carrier_to_meta", LwdC2eMeta=NS)
    diagnostics = NS(cloud_step=Mock(), trace_cloud_c2e_rows=Mock(), token_passthrough_enabled=lambda: False)
    append = source_method(
        scheduler_source,
        "LwdCloudScheduler",
        "_lwd_append_carrier_c2e",
        LwdControlDiag=diagnostics,
        LwdC2eNotify=wire.LwdC2eNotify,
    )

    def annotate(notify, physical, metadata):
        return msgspec.structs.replace(
            notify,
            row_counts=metadata["row_counts"],
            attempt_ids=metadata["attempt_ids"],
            binding_ids=metadata["binding_ids"],
        )

    scheduler = NS(
        _lwd_carrier_to_meta=unpack,
        _lwd_c2e_finish_reasons=lambda *args: [-1],
        _lwd_c2e_prefix_nonces=lambda *args: [],
        lwd_pd_runtime=NS(annotate=annotate),
        pending_c2e=[],
    )
    carrier = (torch.tensor([7, 8, 9, 2, 3]), ["d"], 6, 0)
    append(
        scheduler,
        carrier,
        {0: NS(outputs=[NS(request_id="d", new_token_ids=[7, 8][:committed])])},
        {},
        pd_metadata={"attempt_ids": ["a"], "binding_ids": ["b"], "row_counts": [3]},
    )
    notify = scheduler.pending_c2e[0]
    assert notify.row_counts == [3]
    assert notify.num_accepted_tokens == [committed]
    assert notify.top_id_ths == [[7, 8][:committed]]
    tree = ast.parse((source / "v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py").read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "lwd_build_unembed_batch"
    )
    scope = dict(
        SchedulerOutput=NS(make_empty=lambda: NS()),
        LwdBatch=NS,
        LwdBatchType=NS(LWD_UNEMBED="unembed"),
        LwdUnembedBatch=NS,
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(function)
    exec(compile(module, "edge_unembed", "exec"), scope)
    batch = scope[function.name]("connection", notify).lwd_batch.batch_meta
    worker_type = load_methods(
        "worker/lwd_edge_model/lwd_edge_model_runner.py",
        {"execute_lwd_unembed"},
        time=time,
        lwd_logger=Mock(),
        connection_from=lambda _: NS(edge_id=0, cloud_id=1, dp_idx=3),
        LwdTransferRequest=NS,
        LwdChannelType=NS(DOWN="down"),
        select_token_batch=lambda logits, ranks: list(ranks),
    )
    runner = worker_type()
    runner.model_config = NS(get_hidden_size=lambda: 2)
    runner.get_model = lambda: NS(compute_logits=lambda hidden: torch.zeros(hidden.shape[0], 16))
    runner.worker = NS(
        comm_service=NS(
            submit_recv=lambda _: NS(wait_on_current_stream=lambda: None, result=lambda: NS(tensor=torch.zeros(3, 2)))
        )
    )
    runner._lwd_watermarked_output = lambda: NS()
    with (
        patch.object(torch, "npu", NS(current_stream=Mock()), create=True),
        patch.object(torch.Tensor, "record_stream"),
    ):
        recovered = runner.execute_lwd_unembed(0, batch)
    assert recovered.sampled_token_ids == [[7, 8][:committed]]


def test_d_meta_placeholder_preserves_prompt_shape_without_prompt_storage():
    class Base:
        def _update_states(self, output):
            return output.scheduled_new_reqs[0].prompt_embeds

    runner_type = load_methods(RUNNER, {"_update_states"}, Base, extra_classes=("LwdPdRequest",))
    runner = runner_type()
    runner._lwd_rewind_preempted_requests = lambda _: None
    request = NS(
        req_id="d",
        prompt_embeds=torch.empty(8192, 16, device="meta"),
        lwd_pd_stage_role="decode",
        lwd_pd_binding_id="b",
        lwd_pd_attempt_id="a",
        lwd_pd_tail_start=8191,
        lwd_pd_tail_end=8192,
        lwd_pd_suppress_user_output=False,
    )
    value = runner._update_states(scheduler_output(scheduled_new_reqs=[request]))
    assert value.shape == (8192, 16)
    assert value.untyped_storage().nbytes() == 16 * value.element_size()


def test_p_full_prompt_mrope_is_sliced_at_actual_computation_end():
    runner_type = load_methods(RUNNER, {"_lwd_prime_mrope_positions"}, lwd_logger=Mock())
    runner = runner_type()
    runner.requests = {"p": NS(mrope_positions=torch.zeros(3, 3, dtype=torch.int64))}
    runner._lwd_pd_requests = {"p": NS(stage_role="prefill")}
    positions = torch.tensor([[0, 1, 4, 5], [0, 2, 3, 4], [0, 1, 2, 3]], dtype=torch.int64)
    runner._lwd_prime_mrope_positions("p", positions.numpy().tobytes(), 3)
    torch.testing.assert_close(runner.requests["p"].mrope_positions, positions[:, :3])
    assert runner.requests["p"].mrope_position_delta == 2


@pytest.mark.parametrize("sp", [False, True])
def test_hash_experts_follow_unequal_dp_hidden_padding_and_sp_shard(sp):
    path = ROOT / "worker/lwd_hash/lwd_hash_allgather.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    scope = dict(
        torch=torch,
        F=torch.nn.functional,
        enable_sp=lambda: sp,
        enable_sp_by_pass=lambda: False,
        _EXTRA_CTX=NS(pad_size=1, max_tokens_across_dp=3),
        get_tensor_model_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_rank=lambda: 1,
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    local = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    remote = torch.tensor([[20, 21]], dtype=torch.int32)
    expected = torch.cat((local, remote)) if sp else torch.cat((local, torch.nn.functional.pad(remote, (0, 0, 0, 2))))
    gather = Mock(return_value=expected)
    config = NS(dp_size=2, pcp_size=1, dp_group=NS(all_gather=gather))
    if sp:
        with patch.object(torch.ops.vllm, "maybe_all_gather_and_maybe_unpad", gather, create=True):
            value = scope[function.name](local, config)
        torch.testing.assert_close(gather.call_args.args[0], torch.tensor([[14, 15], [0, 0]], dtype=torch.int32))
    else:
        value = scope[function.name](local, config)
        torch.testing.assert_close(gather.call_args.args[0], local)
    torch.testing.assert_close(value, expected)


def test_request_owned_spec_state_survives_departure_but_not_replacement_or_rollback():
    owner = NS(num_computed_tokens=100, num_prompt_tokens=80)
    other = NS(num_computed_tokens=200, num_prompt_tokens=80)
    requests = {"a": owner, "b": other}
    cache = decode.LwdDecodeState()
    accepted = np.array([3, 2])
    cache.sync_accepted(requests, ["a", "b"], np.array([0, 1]), accepted)
    drafts = torch.tensor([[11, 12], [21, 22]])
    cache.preserve(scheduler_output(num_scheduled_tokens={"b": 3}), requests, {"a": 0, "b": 1}, drafts)
    drafts.fill_(-1)
    assert cache.take_draft("a", owner, 2).tolist() == [11, 12]
    accepted = np.array([1, 7])
    cache.sync_accepted(requests, ["a", "b"], np.array([-1, 0]), accepted)
    assert accepted.tolist() == [3, 7]
    requests["a"] = NS(num_computed_tokens=80, num_prompt_tokens=80)
    cache.preserve(scheduler_output(), requests, {}, None)
    accepted = np.array([9])
    cache.sync_accepted(requests, ["a"], np.array([-1]), accepted)
    assert accepted.tolist() == [1]
    cache.preserve(
        scheduler_output(
            scheduled_cached_reqs=NS(
                resumed_req_ids=set(),
                req_ids=["b"],
                num_computed_tokens=[79],
            )
        ),
        requests,
        {},
        None,
    )
    assert "b" not in cache.accepted


def test_d_tail_mixes_with_decode_without_full_prompt_hash_storage_or_cpu_roundtrip():
    runner_type = load_methods(
        RUNNER, {"_lwd_inject_remote_embeds", "_lwd_decode_expert_ids", "_lwd_prepare_hash_batch"}
    )
    runner = runner_type()
    runner.worker = NS(_lwd_pd_decode=True)
    runner.model_config = NS(get_hidden_size=lambda: 2)
    runner._lwd_hash_dims = lambda: (1, 2)
    runner.lwd_hash_state = NS(add_chunk=Mock(side_effect=AssertionError("D must not build a full prompt table")))
    runner._lwd_hash_experts = torch.zeros(8, 1, 2, dtype=torch.int32)
    runner._lwd_hash_prompt_mask = torch.zeros(8, dtype=torch.bool)
    runner.input_batch = NS(
        req_ids=["decode", "tail"],
        num_reqs=2,
        num_prompt_tokens=np.array([8192, 8192]),
        num_computed_tokens_cpu=np.array([9000, 8191]),
        is_token_ids=np.ones((2, 8192), dtype=bool),
    )
    runner.inputs_embeds = NS(gpu=torch.full((4, 2), 42.0))
    experts = torch.tensor([[[257, 511]]])
    payload = routing.pack_hash_payload(torch.tensor([[7, 8]], dtype=torch.bfloat16), experts)
    plan = [NS(req_id="tail", req_index=1, start=8191, rows=1)]
    with (
        patch.object(torch, "npu", NS(current_stream=Mock()), create=True),
        patch.object(torch.Tensor, "record_stream"),
    ):
        runner._lwd_inject_remote_embeds(payload, None, plan, np.array([3, 1]))
    runner._lwd_prepare_hash_batch(np.array([3, 1]))
    assert runner._lwd_hash_step_tokens == 4
    assert runner._lwd_hash_prompt_mask.tolist() == [False, False, False, True, False, False, False, False]
    assert runner._lwd_hash_experts[3].tolist() == [[257, 511]]
    assert runner.inputs_embeds.gpu.tolist() == [[42, 42], [42, 42], [42, 42], [7, 8]]


def test_p_suppression_is_immutable_for_the_output_generation():
    runner_type = load_methods(
        RUNNER, {"_lwd_collect_down_payload"}, time=time, lwd_logger=Mock(), extra_classes=("LwdDownPayload",)
    )
    runner = runner_type()
    runner._lwd_request_connection = lambda _: "connection"
    runner._lwd_pack_plain_down_rows = Mock(return_value=(torch.ones(1, 2), [torch.ones(1)], None, None))
    output = NS(lwd_pd_suppressed_req_ids=frozenset({"p"}), lwd_pd_request_identities={"d": ("a", "b")})
    # The live map may already have retired P while its old output is collected.
    runner._lwd_pd_requests = {}
    with patch.object(torch, "npu", NS(Event=Mock), create=True):
        payloads = runner._lwd_collect_down_payload(
            torch.ones(2, 2),
            torch.ones(2, 4),
            None,
            output,
            ["p", "d"],
            np.zeros(2, dtype=bool),
            NS(sampled_token_ids=torch.ones(2, 1)),
        )
    assert runner._lwd_pack_plain_down_rows.call_args.args[3].tolist() == [False, True]
    assert payloads[0].req_ids == ["d"]
    assert payloads[0].pd_identities == (("a", "b"),)


def mooncake_scheduler():
    path = ROOT / "distributed/kv_transfer/kv_p2p/mooncake_connector.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MooncakeConnectorScheduler")
    cls.bases = []
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "get_num_new_matched_tokens",
            "request_finished",
            "_state_prefill_token_count",
            "_truncate_request_for_prefill",
            "on_new_request",
        }
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    scope = dict(logger=Mock(), math=math, time=time, RequestStatus=NS(FINISHED_LENGTH_CAPPED="length"))
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[cls.name]()


def test_mooncake_embeddings_import_uses_i_once_and_export_uses_truncated_p_end():
    connector = mooncake_scheduler()
    connector.need_truncate = True
    request = NS(
        prompt_token_ids=None,
        num_prompt_tokens=1024,
        kv_transfer_params={
            "do_remote_prefill": True,
            "lwd_handoff": {"i_end": 1023},
        },
    )
    assert connector.get_num_new_matched_tokens(request, 128) == (895, True)
    assert connector.get_num_new_matched_tokens(request, 1023) == (0, False)
    request.kv_transfer_params = {"do_remote_decode": True}
    request.prompt_embeds = torch.empty(1024, 2, device="meta")
    request._all_token_ids = list(range(1024))
    request.max_tokens = 9
    connector.on_new_request(request)
    connector.on_new_request(request)
    assert request.num_prompt_tokens == request.prompt_embeds.shape[0] == 1023
    request.request_id, request.status, request.output_token_ids = "r", "length", [0]
    connector.block_size = 128
    connector._get_transfer_block_ids = Mock(return_value=([1, 2],))
    connector._get_swa_transfer_block_ids = lambda blocks: blocks
    connector._reqs_need_send = {}
    connector.engine_id, connector.side_channel_host, connector.side_channel_port = "p", "host", 1
    connector.pcp_size = connector.dcp_size = connector.tp_size = 1
    connector.multi_nodes_meta_mapping = {}
    delayed, params = connector.request_finished(request, ([1, 2, 3],))
    assert delayed and params["num_prompt_blocks"] == 8
    assert connector._get_transfer_block_ids.call_args.args[1] == 1023
