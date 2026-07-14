# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for NPUPlatform._inject_edge_cloud_config re-entry.

Background: upstream EngineCore re-invokes ``VllmConfig.__post_init__()``
after the handshake, and worker subprocesses inherit an already-injected
``ParallelConfig`` via serialization. The first injection sets PP=2, so a
second pass without the re-entry guard would hit the "PP/TP must be 1"
validation and crash startup.
"""

from types import SimpleNamespace

import pytest


def _make_vllm_config():
    parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
        world_size=3,
        nnodes=1,
        node_rank=0,
        distributed_executor_backend="uni",
    )
    return SimpleNamespace(parallel_config=parallel_config, additional_config={})


def test_inject_edge_cloud_config_idempotent(monkeypatch):
    """Second __post_init__ pass must not raise after the first injection
    sets PP=2."""
    monkeypatch.setenv("VLLM_ASCEND_EDGE_CLOUD_ENABLED", "true")
    monkeypatch.setenv("VLLM_ASCEND_EDGE_CLOUD_EDGE_NPU_COUNT", "1")
    monkeypatch.setenv("VLLM_ASCEND_EDGE_CLOUD_CLOUD_NPU_COUNT", "2")
    monkeypatch.setenv("VLLM_ASCEND_EDGE_CLOUD_ROLE", "edge")

    from vllm_ascend.platform import NPUPlatform

    vllm_config = _make_vllm_config()

    # First pass: injects edge-cloud topology (PP 1 -> 2) and sets the flag.
    NPUPlatform._inject_edge_cloud_config(vllm_config)
    assert vllm_config.parallel_config.pipeline_parallel_size == 2
    assert vllm_config.additional_config["_edge_cloud_config_injected"] is True

    # Second pass (mimics EngineCore re-invoking __post_init__): must NOT
    # raise despite PP now being 2.
    NPUPlatform._inject_edge_cloud_config(vllm_config)
    assert vllm_config.parallel_config.pipeline_parallel_size == 2


def test_inject_edge_cloud_config_child_inherits_flag(monkeypatch):
    """A serialized-then-rebuilt config (child process) carries the flag in
    additional_config, so the child skips re-injection too."""
    monkeypatch.setenv("VLLM_ASCEND_EDGE_CLOUD_ENABLED", "true")
    monkeypatch.setenv("VLLM_ASCEND_EDGE_CLOUD_EDGE_NPU_COUNT", "1")
    monkeypatch.setenv("VLLM_ASCEND_EDGE_CLOUD_CLOUD_NPU_COUNT", "2")
    monkeypatch.setenv("VLLM_ASCEND_EDGE_CLOUD_ROLE", "cloud")

    from vllm_ascend.platform import NPUPlatform

    parent = _make_vllm_config()
    NPUPlatform._inject_edge_cloud_config(parent)
    assert parent.parallel_config.pipeline_parallel_size == 2

    # Child rebuilds ParallelConfig from serialization (PP already 2) but
    # keeps additional_config verbatim.
    child = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
            data_parallel_size=1,
            world_size=3,
            nnodes=2,
            node_rank=1,
            distributed_executor_backend="mp",
        ),
        additional_config=dict(parent.additional_config),
    )
    # Must not raise on the inherited PP=2.
    NPUPlatform._inject_edge_cloud_config(child)
    assert child.parallel_config.pipeline_parallel_size == 2


def test_inject_edge_cloud_config_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_EDGE_CLOUD_ENABLED", "false")
    from vllm_ascend.platform import NPUPlatform

    vllm_config = _make_vllm_config()
    NPUPlatform._inject_edge_cloud_config(vllm_config)
    assert vllm_config.parallel_config.pipeline_parallel_size == 1
    assert "_edge_cloud_config_injected" not in vllm_config.additional_config
