from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.patch.platform.patch_multiproc_executor import (
    AscendMultiprocExecutor,
)


def make_executor(*, multi_edge: bool = True) -> AscendMultiprocExecutor:
    executor = object.__new__(AscendMultiprocExecutor)
    executor.parallel_config = SimpleNamespace(
        enable_edge_cloud=multi_edge,
        is_edge_node=True,
        num_edges=2 if multi_edge else 1,
        pipeline_parallel_size=2,
    )
    executor.scheduler_config = SimpleNamespace(async_scheduling=False)
    return executor


def test_multi_edge_executor_limits_concurrent_batches_to_one() -> None:
    executor = make_executor()

    assert executor.max_concurrent_batches == 1


def test_multi_edge_non_block_rpc_returns_completed_future() -> None:
    executor = make_executor()

    with patch.object(
        executor,
        "_multi_edge_collective_rpc_sync",
        return_value="edge-result",
    ) as sync_rpc:
        result = executor.collective_rpc("execute_model", non_block=True)

    assert isinstance(result, Future)
    assert result.done()
    assert result.result() == "edge-result"
    sync_rpc.assert_called_once_with(
        "execute_model",
        None,
        (),
        None,
        None,
        None,
    )


def test_multi_edge_non_block_rpc_reports_sync_error_through_future() -> None:
    executor = make_executor()

    with patch.object(
        executor,
        "_multi_edge_collective_rpc_sync",
        side_effect=RuntimeError("cloud failed"),
    ):
        result = executor.collective_rpc("execute_model", non_block=True)

    assert isinstance(result, Future)
    assert result.done()
    assert isinstance(result.exception(), RuntimeError)
    assert str(result.exception()) == "cloud failed"
