import pytest
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)

from vllm_ascend.worker.multi_edge import (
    namespace_cloud_request_ids,
    offset_cloud_kv_block_ids,
)


def make_scheduler_output(edge_id: int) -> SchedulerOutput:
    new_request = NewRequestData(
        req_id="new",
        prompt_token_ids=[1],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0, 3, -1],),
        num_computed_tokens=0,
        lora_request=None,
    )
    cached_request = CachedRequestData(
        req_ids=["cached"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[([4, -1],)],
        num_computed_tokens=[1],
        num_output_tokens=[1],
    )
    return SchedulerOutput(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=cached_request,
        num_scheduled_tokens={"new": 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        new_block_ids_to_zero=[5, -1],
        edge_id=edge_id,
    )


def test_offset_cloud_kv_block_ids_uses_per_edge_namespace() -> None:
    scheduler_output = make_scheduler_output(edge_id=1)

    offset_cloud_kv_block_ids(
        scheduler_output,
        num_edges=2,
        cloud_num_blocks=20,
    )

    assert scheduler_output.scheduled_new_reqs[0].block_ids == ([10, 13, -1],)
    assert scheduler_output.scheduled_cached_reqs.new_block_ids == [([14, -1],)]
    assert scheduler_output.new_block_ids_to_zero == [15, -1]


def test_offset_cloud_kv_block_ids_rejects_missing_cloud_capacity() -> None:
    with pytest.raises(ValueError, match="CLOUD_NUM_BLOCKS"):
        offset_cloud_kv_block_ids(
            make_scheduler_output(edge_id=0),
            num_edges=2,
            cloud_num_blocks=0,
        )


def test_offset_cloud_kv_block_ids_rejects_edge_over_quota() -> None:
    scheduler_output = make_scheduler_output(edge_id=1)
    scheduler_output.scheduled_new_reqs[0].block_ids = ([10],)

    with pytest.raises(ValueError, match="exceeds its cloud KV quota"):
        offset_cloud_kv_block_ids(
            scheduler_output,
            num_edges=2,
            cloud_num_blocks=20,
        )


def test_namespace_cloud_request_ids_isolates_worker_state() -> None:
    scheduler_output = make_scheduler_output(edge_id=1)
    scheduler_output.scheduled_cached_reqs.resumed_req_ids = {"cached"}
    scheduler_output.scheduled_cached_reqs.all_token_ids = {"cached": [1, 2]}
    scheduler_output.scheduled_spec_decode_tokens = {"new": [3]}
    scheduler_output.scheduled_encoder_inputs = {"new": [0]}
    scheduler_output.finished_req_ids = {"finished"}
    scheduler_output.preempted_req_ids = {"preempted"}

    namespace_cloud_request_ids(scheduler_output, num_edges=2)

    assert scheduler_output.scheduled_new_reqs[0].req_id == "edge-1:new"
    assert scheduler_output.scheduled_cached_reqs.req_ids == ["edge-1:cached"]
    assert scheduler_output.scheduled_cached_reqs.resumed_req_ids == {"edge-1:cached"}
    assert scheduler_output.scheduled_cached_reqs.all_token_ids == {"edge-1:cached": [1, 2]}
    assert scheduler_output.num_scheduled_tokens == {"edge-1:new": 1}
    assert scheduler_output.scheduled_spec_decode_tokens == {"edge-1:new": [3]}
    assert scheduler_output.scheduled_encoder_inputs == {"edge-1:new": [0]}
    assert scheduler_output.finished_req_ids == {"edge-1:finished"}
    assert scheduler_output.preempted_req_ids == {"edge-1:preempted"}
