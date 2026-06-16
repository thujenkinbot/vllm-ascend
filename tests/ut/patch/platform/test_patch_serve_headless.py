# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


_PATCH_PATH = (
    Path(__file__).parents[4]
    / "vllm_ascend"
    / "patch"
    / "platform"
    / "patch_serve_headless.py"
)


class FakeVllmEnvs(ModuleType):
    def __init__(self):
        super().__init__("vllm.envs")
        self.cache_disabled = False

    def disable_envs_cache(self):
        self.cache_disabled = True


class FakeReadyReader:
    def __init__(self, response):
        self.response = response
        self.closed = False

    def recv(self):
        return self.response

    def close(self):
        self.closed = True


class FakeReadyWriter:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, target, kwargs, name):
        self.target = target
        self.kwargs = kwargs
        self.name = name
        self.started = False
        self.join_calls = []
        self.terminated = False
        self.exitcode = 0

    def start(self):
        self.started = True

    def join(self, timeout=None):
        self.join_calls.append(timeout)

    def is_alive(self):
        return False

    def terminate(self):
        self.terminated = True


class FakeContext:
    def __init__(self, response):
        self.reader = FakeReadyReader(response)
        self.writer = FakeReadyWriter()
        self.processes = []

    def Pipe(self, duplex=False):
        assert duplex is False
        return self.reader, self.writer

    def Process(self, target, kwargs, name):
        proc = FakeProcess(target=target, kwargs=kwargs, name=name)
        self.processes.append(proc)
        return proc


def _install_fake_modules(monkeypatch, context):
    fake_envs = FakeVllmEnvs()
    fake_vllm = ModuleType("vllm")
    fake_vllm.envs = fake_envs

    class FakeAsyncEngineArgs:
        @staticmethod
        def from_cli_args(args):
            return SimpleNamespace(
                data_parallel_hybrid_lb=False,
                create_engine_config=lambda usage_context, headless: SimpleNamespace(
                    parallel_config=SimpleNamespace(
                        data_parallel_size_local=1,
                        node_rank_within_dp=1,
                        master_addr="10.0.0.1",
                        master_port=29501,
                    ),
                    shutdown_timeout=5,
                ),
            )

    fake_vllm.AsyncEngineArgs = FakeAsyncEngineArgs

    fake_serve = ModuleType("vllm.entrypoints.cli.serve")
    fake_serve.run_headless_called = False

    def original_run_headless(args):
        fake_serve.run_headless_called = True

    fake_serve.run_headless = original_run_headless
    fake_serve.logger = SimpleNamespace(info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)

    fake_usage = ModuleType("vllm.usage.usage_lib")
    fake_usage.UsageContext = SimpleNamespace(OPENAI_API_SERVER="openai")

    fake_system_utils = ModuleType("vllm.utils.system_utils")
    fake_system_utils.get_mp_context = lambda: context

    class FakePassiveEngineCoreProc:
        @staticmethod
        def run_passive_engine_core(**kwargs):
            return None

    fake_engine_core = ModuleType("vllm.v1.engine.core")
    fake_engine_core.PassiveEngineCoreProc = FakePassiveEngineCoreProc

    fake_version = ModuleType("vllm.version")
    fake_version.__version__ = "test-version"

    fake_ascend_core = ModuleType("vllm_ascend.core")
    fake_passive_scheduler = ModuleType("vllm_ascend.core.passive_scheduler")
    fake_ascend_core.passive_scheduler = fake_passive_scheduler

    module_names = {
        "vllm": fake_vllm,
        "vllm.envs": fake_envs,
        "vllm.entrypoints": ModuleType("vllm.entrypoints"),
        "vllm.entrypoints.cli": ModuleType("vllm.entrypoints.cli"),
        "vllm.entrypoints.cli.serve": fake_serve,
        "vllm.usage": ModuleType("vllm.usage"),
        "vllm.usage.usage_lib": fake_usage,
        "vllm.utils": ModuleType("vllm.utils"),
        "vllm.utils.system_utils": fake_system_utils,
        "vllm.v1": ModuleType("vllm.v1"),
        "vllm.v1.engine": ModuleType("vllm.v1.engine"),
        "vllm.v1.engine.core": fake_engine_core,
        "vllm.version": fake_version,
        "vllm_ascend.core": fake_ascend_core,
        "vllm_ascend.core.passive_scheduler": fake_passive_scheduler,
    }
    for name, module in module_names.items():
        monkeypatch.setitem(sys.modules, name, module)
    return fake_serve, fake_envs, fake_engine_core


def _load_patch_module():
    spec = importlib.util.spec_from_file_location(
        "test_patch_serve_headless", _PATCH_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_serve_patch_launches_passive_engine_core_for_non_leader_rank(monkeypatch):
    context = FakeContext(response={"status": "READY"})
    fake_serve, fake_envs, fake_engine_core = _install_fake_modules(monkeypatch, context)
    monkeypatch.delenv("VLLM_PP_SCHEDULER_ZMQ_ADDR", raising=False)
    monkeypatch.setenv("VLLM_PP_SCHEDULER_ZMQ_PORT", "6000")

    _load_patch_module()
    fake_serve.run_headless(SimpleNamespace())

    assert fake_serve.run_headless_called is False
    assert os.environ["VLLM_PP_SCHEDULER_ZMQ_ADDR"] == "tcp://10.0.0.1:6000"
    assert fake_envs.cache_disabled is True
    assert len(context.processes) == 1
    proc = context.processes[0]
    assert proc.name == "PassiveEngineCore"
    assert proc.target.__name__ == "_run_passive_engine_core_with_ascend_shims"
    assert proc.kwargs["ready_pipe"] is context.writer
    assert proc.started is True
    assert context.writer.closed is True
    assert context.reader.closed is True
    assert proc.join_calls == [None]


def test_serve_patch_reports_passive_engine_startup_failure(monkeypatch):
    context = FakeContext(response={"status": "FAILED", "message": "boom"})
    fake_serve, _, _ = _install_fake_modules(monkeypatch, context)

    _load_patch_module()

    try:
        fake_serve.run_headless(SimpleNamespace())
    except RuntimeError as exc:
        assert "PassiveEngineCore failed to start" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
