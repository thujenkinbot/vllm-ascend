from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.executor import Executor

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_ORIGINAL_EXECUTOR_GET_CLASS = Executor.get_class


def get_executor_class(vllm_config: VllmConfig) -> type[Executor]:
    parallel_config = vllm_config.parallel_config
    if parallel_config.enable_edge_cloud and parallel_config.num_edges > 1:
        from vllm_ascend.patch.platform.patch_multiproc_executor import (
            AscendMultiprocExecutor,
        )

        return AscendMultiprocExecutor
    return _ORIGINAL_EXECUTOR_GET_CLASS(vllm_config)


Executor.get_class = staticmethod(get_executor_class)
