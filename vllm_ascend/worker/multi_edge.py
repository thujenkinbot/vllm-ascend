"""Small, version-local helpers for the multi-edge-cloud MVP."""

from vllm.v1.core.sched.output import SchedulerOutput


def namespace_cloud_request_ids(scheduler_output: SchedulerOutput, *, num_edges: int) -> None:
    """Namespace request IDs in the cloud worker's serialized copy.

    Independent edge schedulers may emit the same request ID. Cloud model
    runners keep request-local state keyed by this value, so KV block offsets
    alone are insufficient isolation.
    """
    if num_edges <= 1:
        return
    edge_id = scheduler_output.edge_id
    if not 0 <= edge_id < num_edges:
        raise ValueError(f"edge_id={edge_id} is outside [0, {num_edges})")

    def scoped(request_id: str) -> str:
        return f"edge-{edge_id}:{request_id}"

    for request in scheduler_output.scheduled_new_reqs:
        request.req_id = scoped(request.req_id)

    cached = scheduler_output.scheduled_cached_reqs
    cached.req_ids = [scoped(request_id) for request_id in cached.req_ids]
    cached.resumed_req_ids = {scoped(request_id) for request_id in cached.resumed_req_ids}
    cached.all_token_ids = {scoped(request_id): token_ids for request_id, token_ids in cached.all_token_ids.items()}

    scheduler_output.num_scheduled_tokens = {
        scoped(request_id): num_tokens for request_id, num_tokens in scheduler_output.num_scheduled_tokens.items()
    }
    scheduler_output.scheduled_spec_decode_tokens = {
        scoped(request_id): token_ids for request_id, token_ids in scheduler_output.scheduled_spec_decode_tokens.items()
    }
    scheduler_output.scheduled_encoder_inputs = {
        scoped(request_id): input_ids for request_id, input_ids in scheduler_output.scheduled_encoder_inputs.items()
    }
    scheduler_output.finished_req_ids = {scoped(request_id) for request_id in scheduler_output.finished_req_ids}
    if scheduler_output.preempted_req_ids is not None:
        scheduler_output.preempted_req_ids = {scoped(request_id) for request_id in scheduler_output.preempted_req_ids}


def offset_cloud_kv_block_ids(
    scheduler_output: SchedulerOutput,
    *,
    num_edges: int,
    cloud_num_blocks: int,
) -> None:
    """Move edge-local block IDs into a static cloud-side namespace.

    The scheduler output received by a cloud worker is a serialized copy, so
    mutating it does not change the edge scheduler's local block IDs.
    """
    if num_edges <= 1:
        return
    if cloud_num_blocks <= 0:
        raise ValueError(
            "Cloud KV block count must be positive in multi-edge-cloud mode; "
            "set VLLM_ASCEND_EDGE_CLOUD_CLOUD_NUM_BLOCKS if automatic "
            "detection is unavailable"
        )
    edge_id = scheduler_output.edge_id
    if not 0 <= edge_id < num_edges:
        raise ValueError(f"edge_id={edge_id} is outside [0, {num_edges})")

    stride = cloud_num_blocks // num_edges
    if stride <= 0:
        raise ValueError(f"cloud_num_blocks={cloud_num_blocks} cannot be split across num_edges={num_edges}")
    offset = edge_id * stride

    def shift(block_id: int) -> int:
        if block_id < 0:
            return block_id
        if block_id >= stride:
            raise ValueError(f"edge-local block_id={block_id} exceeds its cloud KV quota of {stride} blocks")
        return block_id + offset

    for request in scheduler_output.scheduled_new_reqs:
        request.block_ids = tuple([shift(block_id) for block_id in group] for group in request.block_ids)

    cached = scheduler_output.scheduled_cached_reqs
    cached.new_block_ids = [
        (tuple([shift(block_id) for block_id in group] for group in block_groups) if block_groups is not None else None)
        for block_groups in cached.new_block_ids
    ]

    if scheduler_output.new_block_ids_to_zero:
        scheduler_output.new_block_ids_to_zero = [
            shift(block_id) for block_id in scheduler_output.new_block_ids_to_zero
        ]
