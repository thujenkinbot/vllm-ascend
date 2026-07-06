#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

from itertools import islice
from typing import Any

import torch
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5_MoeMixtureOfExperts,
)
from vllm.model_executor.models.qwen3_next import QwenNextMixtureOfExperts
from vllm.sequence import IntermediateTensors
from vllm_ascend.edge_cloud_materialized import (
    apply_final_norm,
    make_boundary_tensors,
    restore_boundary_state,
)


def _forward_edge_cloud_segment_qwen3_5(
    self: Qwen3_5Model,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    is_first_segment: bool | None = None,
    is_last_segment: bool | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    num_layers = len(self.layers)
    assert 0 <= start_layer <= end_layer <= num_layers, (
        f"Invalid segment range [{start_layer}, {end_layer}) for {num_layers} layers"
    )

    if is_first_segment is None:
        is_first_segment = start_layer == 0 and get_pp_group().is_first_rank
    if is_last_segment is None:
        is_last_segment = end_layer == num_layers and get_pp_group().is_last_rank

    if is_first_segment:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None, (
            "intermediate_tensors is None in edge-cloud segment; "
            "check that all TP ranks receive tensors correctly."
        )
        hidden_states, residual = restore_boundary_state(self, intermediate_tensors)

    for layer in islice(self.layers, start_layer, end_layer):
        hidden_states, residual = layer(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
            **extra_layer_kwargs,
        )

    if not is_last_segment:
        return make_boundary_tensors(self, hidden_states, residual)

    return apply_final_norm(self.norm, hidden_states, residual)


def _qwen3_5_lm_forward_edge_cloud_segment(
    self: Qwen3_5ForCausalLMBase,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    return self.model.forward_edge_cloud_segment(
        start_layer,
        end_layer,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        **extra_layer_kwargs,
    )


def _qwen3_5_cond_forward_edge_cloud_segment(
    self: Qwen3_5ForConditionalGeneration,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **extra_layer_kwargs: Any,
) -> torch.Tensor | IntermediateTensors:
    return self.language_model.forward_edge_cloud_segment(
        start_layer,
        end_layer,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        **extra_layer_kwargs,
    )


Qwen3_5Model.forward_edge_cloud_segment = _forward_edge_cloud_segment_qwen3_5
Qwen3_5ForCausalLMBase.forward_edge_cloud_segment = (
    _qwen3_5_lm_forward_edge_cloud_segment
)
Qwen3_5ForCausalLM.forward_edge_cloud_segment = _qwen3_5_lm_forward_edge_cloud_segment
Qwen3_5MoeForCausalLM.forward_edge_cloud_segment = (
    _qwen3_5_lm_forward_edge_cloud_segment
)
Qwen3_5ForConditionalGeneration.forward_edge_cloud_segment = (
    _qwen3_5_cond_forward_edge_cloud_segment
)
Qwen3_5MoeForConditionalGeneration.forward_edge_cloud_segment = (
    _qwen3_5_cond_forward_edge_cloud_segment
)


# ---------------------------------------------------------------------------
# Monkey-patch MoE methods so they tolerate PPMissingLayer in edge-cloud mode
# ---------------------------------------------------------------------------

def _qwen_next_update_physical_experts_metadata(
    self, num_physical_experts: int, num_local_physical_experts: int
) -> None:
    from vllm.model_executor.models.qwen3_next import (
        Qwen3NextDecoderLayer,
        Qwen3NextSparseMoeBlock,
    )

    assert self.num_local_physical_experts == num_local_physical_experts
    self.num_physical_experts = num_physical_experts
    self.num_local_physical_experts = num_local_physical_experts
    self.num_redundant_experts = num_physical_experts - self.num_logical_experts
    for layer in self.model.layers:
        if not isinstance(layer, Qwen3NextDecoderLayer):
            continue
        if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
            moe = layer.mlp
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


def _qwen_next_set_moe_parameters(self) -> None:
    from vllm.model_executor.models.qwen3_next import (
        Qwen3NextDecoderLayer,
        Qwen3NextSparseMoeBlock,
    )

    self.expert_weights = []
    self.moe_layers = []
    example_moe = None
    for layer in self.model.layers:
        if isinstance(layer, Qwen3NextDecoderLayer) and isinstance(
            layer.mlp, Qwen3NextSparseMoeBlock
        ):
            example_moe = layer.mlp
            self.moe_layers.append(layer.mlp.experts)

    if example_moe is None:
        self.num_moe_layers = 0
        self.num_expert_groups = 0
        self.num_shared_experts = 0
        self.num_logical_experts = 0
        self.num_physical_experts = 0
        self.num_local_physical_experts = 0
        self.num_routed_experts = 0
        self.num_redundant_experts = 0
        logger.warning("No Qwen3Next MoE layer found in the model.layers.")
        return

    self.num_moe_layers = len(self.moe_layers)
    self.num_expert_groups = 1
    self.num_shared_experts = 0
    self.num_logical_experts = example_moe.n_logical_experts
    self.num_physical_experts = example_moe.n_physical_experts
    self.num_local_physical_experts = example_moe.n_local_physical_experts
    self.num_routed_experts = example_moe.n_routed_experts
    self.num_redundant_experts = example_moe.n_redundant_experts


def _qwen3_5_update_physical_experts_metadata(
    self, num_physical_experts: int, num_local_physical_experts: int
) -> None:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
    from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

    assert self.num_local_physical_experts == num_local_physical_experts
    self.num_physical_experts = num_physical_experts
    self.num_local_physical_experts = num_local_physical_experts
    self.num_redundant_experts = num_physical_experts - self.num_logical_experts
    for layer in self.language_model.model.layers:
        if not isinstance(layer, Qwen3_5DecoderLayer):
            continue
        if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
            moe = layer.mlp
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


def _qwen3_5_set_moe_parameters(self) -> None:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
    from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

    self.expert_weights = []
    self.moe_layers = []
    example_moe = None
    for layer in self.language_model.model.layers:
        if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
            layer.mlp, Qwen3NextSparseMoeBlock
        ):
            example_moe = layer.mlp
            self.moe_layers.append(layer.mlp.experts)

    if example_moe is None:
        self.num_moe_layers = 0
        self.num_expert_groups = 0
        self.num_shared_experts = 0
        self.num_logical_experts = 0
        self.num_physical_experts = 0
        self.num_local_physical_experts = 0
        self.num_routed_experts = 0
        self.num_redundant_experts = 0
        logger.warning(
            "No Qwen3_5 MoE layer found in the language_model.model.layers."
        )
        return

    self.num_moe_layers = len(self.moe_layers)
    self.num_expert_groups = 1
    self.num_shared_experts = 0
    self.num_logical_experts = example_moe.n_logical_experts
    self.num_physical_experts = example_moe.n_physical_experts
    self.num_local_physical_experts = example_moe.n_local_physical_experts
    self.num_routed_experts = example_moe.n_routed_experts
    self.num_redundant_experts = example_moe.n_redundant_experts


QwenNextMixtureOfExperts.update_physical_experts_metadata = (
    _qwen_next_update_physical_experts_metadata
)
QwenNextMixtureOfExperts.set_moe_parameters = _qwen_next_set_moe_parameters
Qwen3_5_MoeMixtureOfExperts.update_physical_experts_metadata = (
    _qwen3_5_update_physical_experts_metadata
)
Qwen3_5_MoeMixtureOfExperts.set_moe_parameters = _qwen3_5_set_moe_parameters
