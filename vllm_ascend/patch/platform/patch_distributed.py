#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
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
# Adapted from vllm/model_executor/models/qwen2_vl.py
# This file is a part of the vllm-ascend project.

import torch

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


class NullHandle:
    def __init__(self):
        pass

    def wait(self):
        pass


def communication_adaptation_310p():
    def broadcast310p_wrapper(fn):
        def broadcast310p(tensor, src=0, group=None, async_op=False, group_src=None):
            root = group_src if group_src is not None else src

            if tensor.device == torch.device("cpu"):
                return fn(tensor, src=root, group=group, async_op=async_op)
            rank = torch.distributed.get_rank(group)
            world_size = torch.distributed.get_world_size(group)
            tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
            tensor_list[rank] = tensor
            torch.distributed.all_gather(tensor_list, tensor, group=group)
            tensor[...] = tensor_list[src]
            if async_op:
                return NullHandle()
            else:
                return None

        return broadcast310p

    torch.distributed.broadcast = broadcast310p_wrapper(torch.distributed.broadcast)
    torch.distributed.distributed_c10d.broadcast = broadcast310p_wrapper(torch.distributed.distributed_c10d.broadcast)

    def all_reduce_wrapper_310p(fn):
        def all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.SUM,
            group=None,
            async_op=False,
        ):
            if tensor.dtype != torch.int64:
                return fn(tensor, op, group, async_op)
            rank = torch.distributed.get_rank(group)
            world_size = torch.distributed.get_world_size(group)
            tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
            tensor_list[rank] = tensor
            torch.distributed.all_gather(tensor_list, tensor, group=group)
            if op == torch.distributed.ReduceOp.SUM:
                return torch.stack(tensor_list).sum(0)
            elif op == torch.distributed.ReduceOp.MAX:
                return torch.tensor(
                    torch.stack(tensor_list).cpu().numpy().max(0),
                    device=tensor.device,
                )
            else:
                raise RuntimeError(f"not implement op {op}")

        return all_reduce

    torch.distributed.all_reduce = all_reduce_wrapper_310p(torch.distributed.all_reduce)
    torch.distributed.distributed_c10d.all_reduce = all_reduce_wrapper_310p(
        torch.distributed.distributed_c10d.all_reduce
    )


if get_ascend_device_type() == AscendDeviceType._310P:
    communication_adaptation_310p()


# ---------------------------------------------------------------------------
# Monkey-patch EP group creation for edge-cloud mode.
# In edge-cloud mode, EP groups must be split by side (edge vs cloud)
# so that MoE all-to-all communication stays within each side.
# This supports both head_tail and embedding_only modes.
# ---------------------------------------------------------------------------

def _apply_ep_edge_cloud_patch():
    import vllm.distributed.parallel_state as _ps
    from vllm.config import get_current_vllm_config_or_none

    _orig_init_model_parallel_group = _ps.init_model_parallel_group

    def _init_model_parallel_group_wrapper(
        group_ranks,
        local_rank,
        backend,
        use_message_queue_broadcaster=False,
        group_name=None,
        use_device_communicator=True,
    ):
        if group_name == "ep" and _ps.is_edge_cloud_pp_mode():
            import torch

            vllm_config = get_current_vllm_config_or_none()
            if vllm_config is not None:
                edge_npu_count = getattr(
                    vllm_config.parallel_config, "edge_npu_count", None
                )
                if edge_npu_count is not None:
                    world_size = torch.distributed.get_world_size()
                    edge_ranks = list(range(edge_npu_count))
                    cloud_ranks = list(range(edge_npu_count, world_size))
                    new_group_ranks = []
                    if edge_ranks:
                        new_group_ranks.append(edge_ranks)
                    if cloud_ranks:
                        new_group_ranks.append(cloud_ranks)
                    group_ranks = new_group_ranks

        return _orig_init_model_parallel_group(
            group_ranks,
            local_rank,
            backend,
            use_message_queue_broadcaster=use_message_queue_broadcaster,
            group_name=group_name,
            use_device_communicator=use_device_communicator,
        )

    _ps.init_model_parallel_group = _init_model_parallel_group_wrapper


_apply_ep_edge_cloud_patch()
