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

"""Edge-cloud collaborative inference monkey-patches (platform layer).

This package consolidates all edge-cloud-related monkey-patches applied
before the worker starts.  Import order matters: patches are applied in
the order listed below.

Modules:
    patch_edge_cloud_executor   -- MultiprocExecutor → AscendMultiprocExecutor
    patch_edge_cloud_model      -- make_layers non-contiguous layer assignment
    patch_edge_cloud_config     -- ParallelConfig.local_world_size override
    patch_edge_cloud_parallel   -- GroupCoordinator PP rank semantics
    patch_edge_cloud_mq         -- MessageQueue same-side detection
    patch_edge_cloud_kv_utils   -- KV cache utils (grouping, projection, config)
    patch_edge_cloud_kv_coord   -- KV cache coordinator (empty-group fallback)
"""

import logging

logger = logging.getLogger(__name__)

import vllm_ascend.patch.platform.edge_cloud.patch_edge_cloud_executor  # noqa: E402,F401
import vllm_ascend.patch.platform.edge_cloud.patch_edge_cloud_model      # noqa: E402,F401
import vllm_ascend.patch.platform.edge_cloud.patch_edge_cloud_config     # noqa: E402,F401
import vllm_ascend.patch.platform.edge_cloud.patch_edge_cloud_parallel   # noqa: E402,F401
import vllm_ascend.patch.platform.edge_cloud.patch_edge_cloud_mq         # noqa: E402,F401
import vllm_ascend.patch.platform.edge_cloud.patch_edge_cloud_kv_utils   # noqa: E402,F401
import vllm_ascend.patch.platform.edge_cloud.patch_edge_cloud_kv_coord   # noqa: E402,F401

logger.debug("edge_cloud patches applied successfully")
