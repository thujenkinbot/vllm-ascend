# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.utils import enable_sp

if TYPE_CHECKING:
    from typing import Any

    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.sample.metadata import SamplingMetadata


class LwdMTPProposer(AscendEagleProposer):
    """Adapt externally supplied LWD embeddings to the MTP drafter."""

    def _lwd_build_first_pass_embeds(
        self,
        num_tokens: int,
        num_input_tokens: int,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        cad: CommonAttentionMetadata,
    ) -> torch.Tensor | None:
        """Reject token-id fallback for continued prefill with prefix caching."""
        enabled = self.vllm_config.cache_config.enable_prefix_caching
        needs_prompt = False
        runner = self.runner
        if enabled and runner is not None:
            batch = runner.input_batch
            computed = batch.num_computed_tokens_cpu
            if len(computed) < len(batch.req_ids):
                raise RuntimeError("LWD prefix MTP is missing computed-token offsets")
            needs_prompt = any(
                int(computed[i]) + 1 < int(batch.num_prompt_tokens[i]) for i in range(len(batch.req_ids))
            )
            if needs_prompt and getattr(runner, "pcp_manager", None) is not None:
                raise RuntimeError("LWD prefix MTP cannot use PCP embedding layout")
        result = super()._lwd_build_first_pass_embeds(
            num_tokens,
            num_input_tokens,
            next_token_ids,
            token_indices_to_sample,
            cad,
        )
        if enabled and needs_prompt and result is None:
            raise RuntimeError("LWD prefix MTP requires authoritative prompt embeddings")
        return result

    def _run_merged_draft(
        self,
        num_input_tokens: int,
        batch_size: int,
        token_indices_to_sample: torch.Tensor,
        target_positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        multi_steps_attn_metadata: Any,
        num_tokens: int,
        is_prefill: bool | None = None,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        """Match replicated external embeddings to the draft's token shard."""
        context = get_forward_context()
        if (
            inputs_embeds is not None
            and enable_sp(self.vllm_config)
            and context.num_tokens is not None
            and not self.runner.is_multimodal_model
        ):
            tp = get_tp_group()
            if tp.world_size > 1:
                if inputs_embeds.shape[0] != num_input_tokens:
                    raise ValueError("LWD MTP SP expects full-token embeddings before slicing")
                pad = context.padded_num_tokens - context.num_tokens
                if pad < 0 or (inputs_embeds.shape[0] + pad) % tp.world_size:
                    raise ValueError("LWD MTP SP padding does not match the TP layout")
                if pad:
                    inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, pad))
                # A reduce-scatter would sum already replicated embeddings.
                inputs_embeds = inputs_embeds.chunk(tp.world_size, dim=0)[tp.rank_in_group]
        return super()._run_merged_draft(
            num_input_tokens,
            batch_size,
            token_indices_to_sample,
            target_positions,
            inputs_embeds,
            multi_steps_attn_metadata,
            num_tokens,
            is_prefill,
            sampling_metadata,
        )
