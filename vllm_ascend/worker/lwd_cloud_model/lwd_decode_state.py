# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Request-owned speculative state for temporary departures from a PD batch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


class LwdDecodeState:
    """Keep owned draft snapshots and acceptance separate from reusable rows."""

    def __init__(self) -> None:
        self.drafts: dict[str, tuple[object, Any]] = {}
        self.accepted: dict[str, tuple[object, int]] = {}

    def discard(self, req_ids: set[str] | list[str]) -> None:
        for req_id in req_ids:
            self.drafts.pop(req_id, None)
            self.accepted.pop(req_id, None)

    def preserve(
        self,
        output: SchedulerOutput,
        requests: dict[str, CachedRequestState],
        prev_map: dict[str, int],
        drafts: torch.Tensor | None,
    ) -> None:
        cached = output.scheduled_cached_reqs
        invalid = set(output.finished_req_ids)
        invalid.update(output.preempted_req_ids or ())
        invalid.update(cached.resumed_req_ids)
        invalid.update(req.req_id for req in output.scheduled_new_reqs)
        for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens):
            state = requests.get(req_id)
            if state is not None and computed < state.num_computed_tokens:
                invalid.add(req_id)
        self.discard(invalid)
        for table in (self.drafts, self.accepted):
            for req_id, (owner, _) in list(table.items()):
                if requests.get(req_id) is not owner:
                    table.pop(req_id)
        if drafts is None or not prev_map:
            return
        departing = [
            (req_id, row)
            for req_id, row in prev_map.items()
            if req_id not in output.num_scheduled_tokens and req_id not in invalid and req_id in requests
        ]
        if departing:
            # Graph/staging storage is reused next step. These views reference
            # our owned clone, never the live draft tensor.
            snapshot = drafts.detach().clone()
            for req_id, row in departing:
                self.drafts[req_id] = (requests[req_id], snapshot[row])

    def take_draft(self, req_id: str, owner: object, length: int) -> Any | None:
        entry = self.drafts.pop(req_id, None)
        if entry is None or entry[0] is not owner or length > len(entry[1]):
            return None
        return entry[1][:length]

    def sync_accepted(
        self,
        requests: dict[str, CachedRequestState],
        req_ids: list[str],
        prev_positions: np.ndarray,
        accepted: np.ndarray,
    ) -> None:
        """Restore only a returning decode owner, and publish the new snapshot."""
        for row, req_id in enumerate(req_ids):
            request = requests.get(req_id)
            if prev_positions[row] < 0:
                saved = self.accepted.get(req_id)
                accepted[row] = (
                    saved[1]
                    if saved is not None
                    and saved[0] is request
                    and request.num_computed_tokens >= request.num_prompt_tokens
                    else 1
                )
            if request is not None and accepted[row] > 0:
                self.accepted[req_id] = (request, int(accepted[row]))
