from typing import Any

import torch

from vllm.sequence import IntermediateTensors


MATERIALIZED_BOUNDARY_MODEL_TYPES = {
    "qwen3_5",
    "qwen3_5_text",
    "minimax_m2",
    "glm_moe_dsa",
    "kimi_k2",
    "kimi_k25",
}

HIDDEN_ONLY_BOUNDARY_MODEL_TYPES = {
    "deepseek_v4",
}


def _model_type(config: Any | None) -> str:
    return getattr(config, "model_type", "") if config is not None else ""


def materialized_boundary_model_types_from_config(model_config: Any) -> set[str]:
    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return {
        _model_type(hf_text_config),
        _model_type(hf_config),
    }


def supports_materialized_boundary_for_config(model_config: Any) -> bool:
    model_types = materialized_boundary_model_types_from_config(model_config)
    return bool(model_types & MATERIALIZED_BOUNDARY_MODEL_TYPES)


def supports_hidden_only_boundary_for_config(model_config: Any) -> bool:
    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    model_types = {
        _model_type(hf_text_config),
        _model_type(hf_config),
    }
    return bool(model_types & HIDDEN_ONLY_BOUNDARY_MODEL_TYPES) or any(
        hasattr(config, "hc_mult") for config in (hf_text_config, hf_config)
    )


def supports_single_hidden_boundary_for_config(model_config: Any) -> bool:
    return (
        supports_materialized_boundary_for_config(model_config)
        or supports_hidden_only_boundary_for_config(model_config)
    )


def uses_materialized_boundary(model: Any) -> bool:
    if getattr(model, "_vllm_ascend_materialized_pp_boundary", False):
        return True
    return _model_type(getattr(model, "config", None)) in MATERIALIZED_BOUNDARY_MODEL_TYPES


def uses_hidden_only_boundary(model: Any) -> bool:
    if getattr(model, "_vllm_ascend_hidden_only_pp_boundary", False):
        return True
    config = getattr(model, "config", None)
    return (
        _model_type(config) in HIDDEN_ONLY_BOUNDARY_MODEL_TYPES
        or hasattr(config, "hc_mult")
    )


def restore_boundary_state(
    model: Any,
    intermediate_tensors: IntermediateTensors,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    hidden_states = intermediate_tensors["hidden_states"]
    if uses_materialized_boundary(model) or uses_hidden_only_boundary(model):
        return hidden_states, None
    return hidden_states, intermediate_tensors["residual"]


def make_boundary_tensors(
    model: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> IntermediateTensors:
    if uses_materialized_boundary(model):
        if residual is not None:
            hidden_states = hidden_states + residual
        return IntermediateTensors({"hidden_states": hidden_states})
    if uses_hidden_only_boundary(model):
        return IntermediateTensors({"hidden_states": hidden_states})
    if residual is None:
        residual = torch.zeros_like(hidden_states)
    return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})


def apply_final_norm(
    norm: Any,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    if residual is None:
        return norm(hidden_states)
    hidden_states, _ = norm(hidden_states, residual)
    return hidden_states
