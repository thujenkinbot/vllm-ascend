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

"""Edge-cloud patch: ParallelConfig.local_world_size → role-sensitive value.

In edge-cloud mode each side (edge / cloud) has its own local NPU count,
which differs from the standard world_size-based calculation.
"""

import logging

from vllm.config import ParallelConfig

from vllm_ascend.distributed.parallel_state import (
    get_cloud_npu_count,
    get_edge_npu_count,
    is_edge_cloud_pp_mode,
    is_edge_device,
)

logger = logging.getLogger(__name__)

_original_local_world_size = ParallelConfig.local_world_size.fget


@property  # type: ignore[misc]
def _ascend_local_world_size(self):
    if is_edge_cloud_pp_mode():
        return (
            get_edge_npu_count()
            if is_edge_device()
            else get_cloud_npu_count()
        )
    return _original_local_world_size(self)


ParallelConfig.local_world_size = _ascend_local_world_size

logger.debug("patch_edge_cloud_config applied successfully")
