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

"""Edge-cloud patch: KV cache utils.

Patches vllm.v1.core.kv_cache_utils to handle:
- Edge-cloud non-contiguous / heterogeneous layer grouping
- Empty-spec workers (embedding_only edge)
- Unified page size with padded alignment
"""

import logging
from collections import defaultdict
from functools import partial

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.worker_base import logger as _kv_logger

logger = logging.getLogger(__name__)

# ---- 6.1 _get_kv_cache_groups_uniform_page_size ----

_orig_get_kv_cache_groups_uniform_page_size = (
    vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size
)


def _ascend_get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    from vllm.v1.core.kv_cache_utils import create_kv_cache_group_specs

    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    per_type_counts: dict[type, int] = defaultdict(int)
    for spec, names in same_type_layers.items():
        per_type_counts[type(spec)] += len(names)
    min_num_layers = min(per_type_counts.values())
    group_size = min_num_layers
    max_num_layers = max(per_type_counts.values())
    if max_num_layers < min_num_layers * 1.5:
        group_size = max_num_layers

    grouped_layers = []
    for layers in same_type_layers.values():
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            _kv_logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size = (
    _ascend_get_kv_cache_groups_uniform_page_size
)

# ---- 6.2 _project_kv_cache_groups_to_worker ----

_orig_project_kv_cache_groups_to_worker = (
    vllm.v1.core.kv_cache_utils._project_kv_cache_groups_to_worker
)


def _ascend_project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    projected_groups = []
    for group in global_kv_cache_groups:
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        if not worker_layer_names:
            continue
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        projected_groups.append(
            KVCacheGroupSpec(
                worker_layer_names,
                group_spec,
                is_eagle_group=group.is_eagle_group and bool(worker_layer_names),
            )
        )
    return projected_groups


vllm.v1.core.kv_cache_utils._project_kv_cache_groups_to_worker = (
    _ascend_project_kv_cache_groups_to_worker
)

# ---- 6.3 get_kv_cache_configs ----

_orig_get_kv_cache_configs = vllm.v1.core.kv_cache_utils.get_kv_cache_configs


def _ascend_get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    from vllm.v1.core.kv_cache_utils import (
        _auto_fit_max_model_len,
        _check_enough_kv_cache_memory,
        _estimate_max_model_len_from_groups,
        _max_memory_usage_bytes_from_groups,
        _pool_bytes_per_block,
        _report_kv_cache_config,
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )

    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    projected_groups_per_worker = [
        _ascend_project_kv_cache_groups_to_worker(
            global_kv_cache_groups,
            worker_spec if worker_spec else merged_kv_cache_specs,
        )
        for worker_spec in kv_cache_specs
    ]

    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        adjusted_memory = []
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            if not groups:
                adjusted_memory.append(avail_mem)
                continue
            bytes_per_block = _pool_bytes_per_block(groups)
            _kv_logger.info(
                "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
                avail_mem // bytes_per_block,
                override,
            )
            adjusted_memory.append(override * bytes_per_block)
        available_memory = adjusted_memory

    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )

    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

    kv_cache_configs = []
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        if kv_cache_spec_one_worker:
            assert sum(len(group.layer_names) for group in projected_groups) == len(
                kv_cache_spec_one_worker
            ), "Some layers are not assigned to any group."
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )

    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks
        if len(kv_cache_config.kv_cache_groups) > 0:
            _report_kv_cache_config(vllm_config, kv_cache_config)

    return kv_cache_configs


vllm.v1.core.kv_cache_utils.get_kv_cache_configs = _ascend_get_kv_cache_configs

# ---- 6.4 generate_scheduler_kv_cache_config ----


def _ascend_generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    import copy

    if len(kv_cache_configs) > 1:
        max_groups = max(len(cfg.kv_cache_groups) for cfg in kv_cache_configs)
        if max_groups > 0:
            max_group_configs = [
                cfg
                for cfg in kv_cache_configs
                if len(cfg.kv_cache_groups) == max_groups
            ]
            if len(max_group_configs) < len(kv_cache_configs):
                kv_cache_configs = max_group_configs

    assert all(
        cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs
    ), "All configs must have the same num_blocks."

    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    return cfg


vllm.v1.core.kv_cache_utils.generate_scheduler_kv_cache_config = (
    _ascend_generate_scheduler_kv_cache_config
)

# core.py imports get_kv_cache_configs and generate_scheduler_kv_cache_config
# directly at module level; update those cached bindings as well.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.get_kv_cache_configs = _ascend_get_kv_cache_configs
vllm.v1.engine.core.generate_scheduler_kv_cache_config = (
    _ascend_generate_scheduler_kv_cache_config
)

# ---- 6.5 unify_kv_cache_spec_page_size ----

_orig_unify_kv_cache_spec_page_size = (
    vllm.v1.core.kv_cache_utils.unify_kv_cache_spec_page_size
)


def _ascend_unify_kv_cache_spec_page_size(kv_cache_spec):
    from dataclasses import replace

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            if getattr(layer_spec, "page_size_padded", None) is not None:
                new_page_size_padded = layer_spec.page_size_padded * ratio
                new_spec = replace(
                    layer_spec,
                    block_size=new_block_size,
                    page_size_padded=new_page_size_padded,
                )
            else:
                new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


vllm.v1.core.kv_cache_utils.unify_kv_cache_spec_page_size = (
    _ascend_unify_kv_cache_spec_page_size
)

logger.debug("patch_edge_cloud_kv_utils applied successfully")
