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

"""Edge-cloud patch: parallel_state PP semantics.

Patches GroupCoordinator.is_first_rank / is_last_rank to support edge-cloud
override semantics (via _override_is_first_rank / _override_is_last_rank)
and role-based rank detection.
"""

import logging

import vllm.distributed.parallel_state as _ps

from vllm_ascend.distributed.parallel_state import (
    get_edge_device_flag,
    is_cloud_device,
    is_edge_device,
    reset_edge_device_flag,
)

logger = logging.getLogger(__name__)

# ---- is_first_rank / is_last_rank ----

@property  # type: ignore[misc]
def _patched_is_first_rank(self):
    if hasattr(self, "_override_is_first_rank"):
        return self._override_is_first_rank
    if get_edge_device_flag() is not None and getattr(self, "unique_name", "").startswith("pp"):
        return not is_cloud_device()
    return self.rank == self.first_rank


@property  # type: ignore[misc]
def _patched_is_last_rank(self):
    if hasattr(self, "_override_is_last_rank"):
        return self._override_is_last_rank
    if get_edge_device_flag() is not None and getattr(self, "unique_name", "").startswith("pp"):
        return not is_cloud_device()
    return self.rank == self.last_rank


_ps.GroupCoordinator.is_first_rank = _patched_is_first_rank
_ps.GroupCoordinator.is_last_rank = _patched_is_last_rank

# ---- destroy_model_parallel ----

_orig_destroy_model_parallel = _ps.destroy_model_parallel


def _ascend_destroy_model_parallel() -> None:
    _orig_destroy_model_parallel()
    reset_edge_device_flag()


_ps.destroy_model_parallel = _ascend_destroy_model_parallel

logger.debug("patch_edge_cloud_parallel applied successfully")
