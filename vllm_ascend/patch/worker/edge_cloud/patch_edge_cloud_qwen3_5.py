#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

from vllm.logger import logger
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5_MoeMixtureOfExperts,
)
from vllm.model_executor.models.qwen3_next import QwenNextMixtureOfExperts


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
