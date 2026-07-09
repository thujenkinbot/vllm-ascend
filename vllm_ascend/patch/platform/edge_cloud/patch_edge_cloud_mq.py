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

"""Edge-cloud patch: MessageQueue same-side detection.

When edge and cloud workers are on different nodes, the MessageQueue must
detect whether the reader and writer are co-located (same_node) to decide
whether to use local or remote buffer I/O.
"""

import logging
from typing import cast

import torch.distributed as dist
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.platforms import current_platform

from vllm_ascend.distributed.parallel_state import (
    in_the_same_node_as_edge_cloud,
    is_edge_cloud_pp_mode,
)

logger = logging.getLogger(__name__)

_original_create_single_reader = MessageQueue.create_from_process_group_single_reader


@staticmethod  # type: ignore[misc]
def _ascend_create_from_process_group_single_reader(
    pg,
    max_chunk_bytes,
    max_chunks,
    reader_rank: int = 0,
    blocking: bool = False,
) -> tuple["MessageQueue", list[Handle]]:
    rank = dist.get_rank()
    ranks = dist.get_process_group_ranks(pg)
    reader_rank_in_group = ranks.index(reader_rank)
    rank_in_group = ranks.index(rank)

    if is_edge_cloud_pp_mode():
        same_node_status = in_the_same_node_as_edge_cloud(
            pg, source_rank=reader_rank_in_group
        )
        same_node = same_node_status[rank_in_group]
    else:
        local_size = current_platform.device_count()
        same_node = rank // local_size == reader_rank // local_size

    buffer_io = MessageQueue(
        n_reader=1,
        n_local_reader=1 if same_node else 0,
        max_chunk_bytes=max_chunk_bytes,
        max_chunks=max_chunks,
    )
    handle = buffer_io.export_handle()
    handles = [None] * dist.get_world_size(pg) if rank == reader_rank else None
    dist.gather_object(handle, handles, dst=reader_rank, group=pg)
    if blocking:
        buffer_io.wait_until_ready()
    return buffer_io, cast(list[Handle], handles or [])


MessageQueue.create_from_process_group_single_reader = _ascend_create_from_process_group_single_reader

logger.debug("patch_edge_cloud_mq applied successfully")
