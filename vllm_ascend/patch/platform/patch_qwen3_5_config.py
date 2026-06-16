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
# ----------------------------------------------------------------------------
# Patch target:
#   - vllm.transformers_utils.configs.qwen3_5.Qwen3_5Config
#   - vllm.transformers_utils.configs.qwen3_5_moe.Qwen3_5MoeConfig
#
# Why:
#   Qwen3.5 / Qwen3.5-MoE are multimodal models whose top-level config
#   stores text-model fields (num_hidden_layers, hidden_size, vocab_size,
#   layer_types, num_experts ...) only on the nested ``text_config``.
#   Several places in vllm-ascend (and downstream PD-mix code) reach for
#   these fields directly on the top-level ``hf_config`` object — e.g.
#   ``vllm_config.model_config.hf_config.num_hidden_layers`` — and would
#   otherwise raise ``AttributeError``.
#
# How:
#   Bind read-only ``@property`` descriptors on the two config classes
#   that delegate to ``self.text_config``.  Because data descriptors live
#   on the class object and take precedence over instance ``__dict__``,
#   this works for already-instantiated configs as well.
#
#   Loaded as a platform-stage patch so it is in place before any model
#   config is constructed.  Idempotent: re-import is a no-op.
# ----------------------------------------------------------------------------

from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeConfig

# Sentinel attribute used for idempotency.  Avoids overwriting properties
# if this module is imported more than once (e.g. by both the platform
# patch __init__ and a manual ``import`` elsewhere).
_PATCH_FLAG = "_vllm_ascend_text_config_proxy_patched"


def _make_text_config_getter(name: str):
    """Return a getter that proxies ``name`` to ``self.text_config``."""

    def _getter(self):
        return getattr(self.text_config, name)

    _getter.__name__ = f"_get_{name}_from_text_config"
    return _getter


# Fields shared by the dense and MoE multimodal configs.
_COMMON_TEXT_FIELDS = (
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_size",
    "vocab_size",
    "layer_types",
)

# MoE-only extra field.
_MOE_EXTRA_FIELDS = ("num_experts",)


def _install_proxies(cls, fields):
    if getattr(cls, _PATCH_FLAG, False):
        return
    for name in fields:
        # Skip if the class (or one of its bases) already defines the
        # attribute as a real descriptor — never silently shadow that.
        existing = getattr(cls, name, None)
        if isinstance(existing, property):
            continue
        setattr(cls, name, property(_make_text_config_getter(name)))
    setattr(cls, _PATCH_FLAG, True)


_install_proxies(Qwen3_5Config, _COMMON_TEXT_FIELDS)
_install_proxies(
    Qwen3_5MoeConfig, _COMMON_TEXT_FIELDS + _MOE_EXTRA_FIELDS
)
