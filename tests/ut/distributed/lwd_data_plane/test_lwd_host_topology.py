# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exercise host bootstrap and routing with fake transport/device endpoints."""

from __future__ import annotations

import ast
import enum
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

HOST = Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/lwd_data_plane/host"


class Channel(enum.Enum):
    UP = "lwd_up"
    DOWN = "lwd_down"


def load_functions(path: Path, names: set[str], scope: dict) -> None:
    source = ast.parse(path.read_text())
    module = ast.parse("from __future__ import annotations")
    module.body.extend(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names)
    exec(compile(module, str(path), "exec"), scope)


def connection(edge: int, cloud: int, dp: int) -> SimpleNamespace:
    return SimpleNamespace(
        connection_id=SimpleNamespace(edge_id=edge, cloud_id=cloud, dp_idx=dp),
        cloud=SimpleNamespace(addr=f"10.1.0.{cloud + 1}", data_port=7000 + dp),
    )


def bootstrap(role: str, connections: list, dp_rank: int = 0) -> dict:
    listeners = []

    class Listener:
        def __init__(self, addr, port):
            assert (addr, port) not in [(item.addr, item.port) for item in listeners]
            self.addr, self.port = addr, port
            self.started = 0
            listeners.append(self)

        def start(self):
            self.started += 1

    config = SimpleNamespace(role=role, instance_id=0, topology=object(), connections=connections)
    scope = {
        "_UCX_CHANNELS_INITIALIZED": False,
        "_UCX_IOS": {},
        "_UCX_LISTENERS": {},
        "_HOST_MEM_BUDGET_MB": 10000,
        "get_current_vllm_config": lambda: SimpleNamespace(lwd_config=config),
        "_my_dp_identity": lambda: (dp_rank, 0),
        "_resolve_up_block_mb": lambda: 64,
        "_host_plane": lambda: SimpleNamespace(down_buffer_size=32, deployment="in_process"),
        "_host_memory_need_mb": lambda *args, **kwargs: 1,
        "_qos_data_plane": lambda: SimpleNamespace(traffic_class=None, sl=None),
        "LwdUcxListener": Listener,
        "LwdNpuUcxChannelIo": lambda channel, stream, cfg: cfg,
        "LwdNpuUcxChannelIoConfig": lambda **kwargs: SimpleNamespace(**kwargs),
        "LwdChannelType": Channel,
        "torch": SimpleNamespace(npu=SimpleNamespace(Stream=lambda **kw: object())),
        "logger": Mock(),
        "hashlib": hashlib,
        "lwd_warm_up_host_channels": Mock(),
    }
    load_functions(
        HOST / "lwd_host_channels.py",
        {"_link_token", "lwd_init_host_channels", "_io", "lwd_get_host_io"},
        scope,
    )
    scope["lwd_init_host_channels"]()
    return scope


def test_edge_dp_zero_connects_to_every_p_and_d_engine() -> None:
    connections = [connection(0, cloud, dp) for cloud, size in [(0, 2), (1, 4)] for dp in range(size)]
    scope = bootstrap("edge", connections)
    ios = scope["_UCX_IOS"]
    assert len(ios) == 12
    assert (0, 0, 1, Channel.UP, "send") in ios
    assert (0, 1, 3, Channel.DOWN, "recv") in ios
    assert len({io.token for io in ios.values()}) == 12
    scope["lwd_warm_up_host_channels"].assert_called_once()
    # Explicit identity must select the D link even with P links present.
    assert scope["lwd_get_host_io"](Channel.DOWN, 3, "recv", 0, 1) is ios[0, 1, 3, Channel.DOWN, "recv"]
    with pytest.raises(RuntimeError, match="ambiguous"):
        scope["lwd_get_host_io"](Channel.DOWN, 3, "recv")


def test_cloud_dp_reuses_one_listener_across_edges() -> None:
    connections = [connection(edge, 1, dp) for edge in (0, 1) for dp in range(4)]
    scope = bootstrap("cloud", connections, dp_rank=3)
    listeners = list(scope["_UCX_LISTENERS"].values())
    assert len(listeners) == 1
    assert listeners[0].started == 1
    ios = scope["_UCX_IOS"]
    assert len(ios) == 4
    assert all(io.listener is listeners[0] for io in ios.values())
    assert all(key[2] == 3 for key in ios)


def test_host_queue_looks_up_the_full_stage_connection() -> None:
    source = ast.parse((HOST / "lwd_host_transfer_queue.py").read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_io")
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    lookup = Mock(return_value=object())
    scope = {"lwd_get_host_io": lookup}
    exec(compile(module, str(HOST / "lwd_host_transfer_queue.py"), "exec"), scope)
    queue = SimpleNamespace(
        channel_type=Channel.UP,
        dp_idx=3,
        op="send",
        connection=SimpleNamespace(edge_id=2, cloud_id=1, dp_idx=3),
    )
    first = scope["_io"](queue)
    assert scope["_io"](queue) is first
    lookup.assert_called_once_with(Channel.UP, 3, "send", edge_id=2, cloud_id=1)
