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

"""Edge-cloud patch: make_layers → edge-cloud-aware non-contiguous layer assignment.

In edge-cloud mode the model is split into head + cloud + tail segments,
so the local worker may own non-contiguous layer ranges.  This patch
replaces vLLM's contiguous make_layers with a version that respects
get_edge_cloud_local_indices().
"""

import logging
from typing import TYPE_CHECKING

import torch
import vllm.model_executor.models.utils
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_ascend.distributed.parallel_state import (
    get_edge_cloud_local_indices,
    is_edge_cloud_pp_mode,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import LayerFn

logger = logging.getLogger(__name__)

_orig_make_layers = vllm.model_executor.models.utils.make_layers


def make_layers(
    num_hidden_layers: int,
    layer_fn: "LayerFn",
    prefix: str,
) -> tuple[int, int, torch.nn.ModuleList]:
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.distributed.utils import get_pp_indices
    from vllm.model_executor.offloader import get_offloader

    if is_edge_cloud_pp_mode():
        local_indices = get_edge_cloud_local_indices(num_hidden_layers)
        if local_indices is not None:
            sorted_idx = sorted(local_indices)
            offloader = get_offloader()
            if sorted_idx:
                real_layers = offloader.wrap_modules(
                    layer_fn(prefix=f"{prefix}.{idx}") for idx in sorted_idx
                )
                real_iter = iter(zip(sorted_idx, real_layers))
            else:
                real_iter = iter([])
            next_idx, next_layer = next(real_iter, (None, None))
            modules_list = []
            for idx in range(num_hidden_layers):
                if idx == next_idx:
                    modules_list.append(next_layer)
                    next_idx, next_layer = next(real_iter, (None, None))
                else:
                    modules_list.append(PPMissingLayer())
            if sorted_idx:
                start_layer = sorted_idx[0]
                end_layer = sorted_idx[-1] + 1
            else:
                start_layer = 0
                end_layer = 0
            return start_layer, end_layer, torch.nn.ModuleList(modules_list)

    return _orig_make_layers(num_hidden_layers, layer_fn, prefix)


vllm.model_executor.models.utils.make_layers = make_layers

logger.debug("patch_edge_cloud_model applied successfully")
