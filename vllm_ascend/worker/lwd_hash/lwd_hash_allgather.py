# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Collect per-token LWD Hash routing tables in the MoE input layout."""

import torch
import torch.nn.functional as F
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.utils import enable_sp, enable_sp_by_pass


def lwd_all_gather_hash_expert_ids(
    expert_ids: torch.Tensor,
    moe_config: FusedMoEConfig,
    *,
    multistream_overlap_gate: bool = False,
) -> torch.Tensor:
    """Align the local per-token Hash table with prepared ALLGATHER rows.

    Unlike vocabulary IDs, local row indices have no meaning on other DP
    ranks. Gather their expert IDs with the same padding and ordering as
    hidden states, then let the caller index the resulting global table.
    """
    if multistream_overlap_gate:
        # Gate overlap routes before prepare(), whereas this function routes
        # after prepare(). Do not silently gather an already local gate.
        raise ValueError("LWD Hash ALLGATHER requires multistream_overlap_gate=False")

    if enable_sp() or enable_sp_by_pass():
        # LWD keeps full-token routing metadata even when hidden states
        # are TP-sharded. Recreate that shard before the normal EP gather
        # so its per-DP unpadding also applies to the expert table.
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            expert_ids = F.pad(expert_ids, (0, 0, 0, pad_size))
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if expert_ids.shape[0] % tp_size:
            raise ValueError("LWD Hash routing rows must be divisible by TP after SP padding")
        expert_ids = expert_ids.chunk(tp_size, dim=0)[tp_rank].contiguous()
        expert_ids = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(expert_ids, True, True)
    elif moe_config.dp_size > 1:
        pad_size = _EXTRA_CTX.max_tokens_across_dp - expert_ids.shape[0]
        if pad_size < 0:
            raise ValueError("LWD Hash routing rows exceed the DP token maximum")
        if pad_size > 0:
            expert_ids = F.pad(expert_ids, (0, 0, 0, pad_size))
        expert_ids = moe_config.dp_group.all_gather(expert_ids, 0)

    if moe_config.pcp_size > 1:
        pad_size = _EXTRA_CTX.max_tokens_across_pcp - expert_ids.shape[0]
        if pad_size > 0:
            expert_ids = F.pad(expert_ids, (0, 0, 0, pad_size))
        expert_ids = get_pcp_group().all_gather(expert_ids, dim=0)
    return expert_ids
