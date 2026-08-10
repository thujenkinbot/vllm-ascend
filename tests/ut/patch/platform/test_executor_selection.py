from types import SimpleNamespace

from vllm.v1.executor import Executor

from vllm_ascend.patch.platform.patch_executor_selection import (
    get_executor_class,
)


def test_multi_edge_selects_ascend_executor() -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_edge_cloud=True,
            num_edges=2,
            distributed_executor_backend="mp",
        )
    )

    executor_class = get_executor_class(vllm_config)

    assert executor_class.__name__ == "AscendMultiprocExecutor"
    assert Executor.get_class(vllm_config) is executor_class
