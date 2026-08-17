from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_script() -> ModuleType:
    script_path = Path(__file__).parents[3] / "examples" / "edge_cloud" / "simulate_edge_usage.py"
    spec = importlib.util.spec_from_file_location(
        "simulate_edge_usage",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def simulator() -> ModuleType:
    return load_script()


def test_usage_payload_ranges_and_total(simulator: ModuleType) -> None:
    rng = random.Random(7)

    for _ in range(100):
        payload = simulator.build_usage_payload(
            edge_id="edge-0",
            model="test-model",
            rng=rng,
        )
        usage = payload["usage"]

        assert 1_000 <= usage["prompt_tokens"] <= 1_000_000
        assert 10 <= usage["completion_tokens"] <= 10_000
        assert usage["total_tokens"] == (usage["prompt_tokens"] + usage["completion_tokens"])
        assert payload["edge_id"] == "edge-0"
        assert payload["model"] == "test-model"
        assert payload["status_code"] == 200


def test_bootstrap_payload_is_zero(simulator: ModuleType) -> None:
    payload = simulator.build_usage_payload(
        edge_id="edge-1",
        model="test-model",
        rng=random.Random(9),
        bootstrap=True,
    )

    assert payload["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert payload["duration_ms"] == 0


def test_post_usage_sends_key_and_json(
    simulator: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(simulator, "urlopen", fake_urlopen)
    payload = simulator.build_usage_payload(
        edge_id="edge-0",
        model="test-model",
        rng=random.Random(11),
    )

    status = simulator.post_usage(
        url="http://higress/internal/edge-usage",
        api_key="secret-key",
        payload=payload,
        timeout_seconds=3.0,
    )

    request = captured["request"]
    assert status == 200
    assert captured["timeout"] == 3.0
    assert request.get_header("X-api-key") == "secret-key"
    assert json.loads(request.data) == payload
