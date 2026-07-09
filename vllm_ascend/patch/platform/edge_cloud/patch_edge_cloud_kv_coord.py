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

"""Edge-cloud patch: KV cache coordinator.

Handles edge-cloud empty kv_cache_groups (embedding_only edge) and single-group
configs by routing to the appropriate coordinator implementation.
"""

import logging
import sys

import vllm.v1.core.kv_cache_coordinator as _kv_coord_mod
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    KVCacheCoordinatorNoPrefixCache,
    UnitaryKVCacheCoordinator,
)

logger = logging.getLogger(__name__)

# ---- 7.1 get_kv_cache_coordinator ----

_orig_get_kv_cache_coordinator = _kv_coord_mod.get_kv_cache_coordinator


def _ascend_get_kv_cache_coordinator(
    kv_cache_config,
    max_model_len: int,
    max_num_batched_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    eagle_attn_layer_names: list[str] | None = None,
    metrics_collector=None,
) -> KVCacheCoordinator:
    if not enable_caching or len(kv_cache_config.kv_cache_groups) == 0:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    return _orig_get_kv_cache_coordinator(
        kv_cache_config,
        max_model_len,
        max_num_batched_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size,
        pcp_world_size,
        hash_block_size,
        eagle_attn_layer_names=eagle_attn_layer_names,
        metrics_collector=metrics_collector,
    )


_kv_coord_mod.get_kv_cache_coordinator = _ascend_get_kv_cache_coordinator

# Also update kv_cache_manager's cached binding if already loaded.
_kv_cache_manager_mod = sys.modules.get("vllm.v1.core.kv_cache_manager")
if _kv_cache_manager_mod is not None:
    _kv_cache_manager_mod.get_kv_cache_coordinator = _ascend_get_kv_cache_coordinator  # type: ignore[attr-defined]

# ---- 7.2 AscendHybridKVCacheCoordinator.verify_and_split_kv_cache_groups ----

if hasattr(_kv_coord_mod, "AscendHybridKVCacheCoordinator"):
    _orig_verify = _kv_coord_mod.AscendHybridKVCacheCoordinator.verify_and_split_kv_cache_groups

    def _ascend_verify_and_split_kv_cache_groups(self) -> None:
        from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        attention_groups = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, (
                        "Expected same manager class for identical KV cache specs."
                    )
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))

        if not attention_groups:
            self.attention_groups = []
            self.lcm_block_size = self.hash_block_size
            self.eagle_attn_group_indices: set[int] = set()
            return

        # Delegate to original implementation for the rest.
        _orig_verify(self)

    _kv_coord_mod.AscendHybridKVCacheCoordinator.verify_and_split_kv_cache_groups = _ascend_verify_and_split_kv_cache_groups  # type: ignore[method-assign]

logger.debug("patch_edge_cloud_kv_coord applied successfully")
