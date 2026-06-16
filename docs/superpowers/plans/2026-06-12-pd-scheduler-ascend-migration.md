# PD Scheduler Ascend 迁移实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 `PDSeparatedScheduler` 与 `PassiveScheduler` 从 vLLM fork 迁移到 vllm-ascend 插件内，同时保留 `BatchType` / `HiddenChannelType` / `SchedulerOutput` schema 在 vLLM core 中。

**架构：** vllm-ascend 新增 `vllm_ascend.core.pd_separated_scheduler` 与 `vllm_ascend.core.passive_scheduler`，通过 `scheduler_config.scheduler_cls` 选择 Ascend scheduler。已有 headless/passive 启动 patch 改为使用 Ascend 模块路径，避免继续依赖 `vllm.v1.core.sched.pd_separated_scheduler` / `passive_scheduler`。

**技术栈：** Python、pytest、vLLM v1 Scheduler、vllm-ascend platform plugin/patch。

---

## 文件结构

- 创建：`vllm_ascend/core/pd_separated_scheduler.py`
  - 放置 edge 侧 PD 分离 scheduler，继承 vLLM `Scheduler` / `AsyncScheduler`，消费 vLLM core 中的 `BatchType`、`HiddenChannelType`、`SchedulerOutput` schema。
- 创建：`vllm_ascend/core/passive_scheduler.py`
  - 放置 cloud / non-leader PP rank 被动 scheduler，接收 `PPSchedulerZmqSubscriber` 的 `SchedulerOutput`，按 `BatchType` 分类并生成 layer slice dispatch plan。
- 修改：`vllm_ascend/platform.py`
  - 当 `scheduler_config.enable_pd_separation` 为真时设置 `scheduler_config.scheduler_cls` 到 vllm-ascend scheduler 类路径。
- 修改：`vllm_ascend/patch/platform/patch_serve_headless.py`
  - 在启动 `PassiveEngineCoreProc.run_passive_engine_core` 前安装一个小型 shim，使 upstream `PassiveEngineCoreProc` 内部对 `vllm.v1.core.sched.passive_scheduler` 的导入解析到 `vllm_ascend.core.passive_scheduler`。
- 修改：`vllm_ascend/scheduler_conflicts.py`
  - 增加 PD schema 存在性校验，缺少 vLLM core schema 时 fail fast。
- 测试：`tests/ut/test_scheduler_pd_separation_conflicts.py`
  - 覆盖 schema 校验和 no-op 路径。
- 新建测试：`tests/ut/test_pd_scheduler_migration.py`
  - 验证 Ascend scheduler 模块可导入、类路径正确、headless shim 可安装。

---

### 任务 1：补最小失败测试

**文件：**
- 修改：`tests/ut/test_scheduler_pd_separation_conflicts.py`
- 创建：`tests/ut/test_pd_scheduler_migration.py`

- [ ] **步骤 1：为 schema 校验写失败测试**

在 `tests/ut/test_scheduler_pd_separation_conflicts.py` 增加：

```python
def test_pd_separation_requires_vllm_pd_scheduler_schema(monkeypatch):
    import vllm_ascend.scheduler_conflicts as conflicts

    monkeypatch.setattr(conflicts, "_vllm_pd_scheduler_schema_available", lambda: False)

    with pytest.raises(ValueError, match="BatchType.*HiddenChannelType.*SchedulerOutput"):
        conflicts.validate_pd_separation_scheduler_conflicts(_vllm_config(), _ascend_config())
```

- [ ] **步骤 2：为 Ascend scheduler 模块路径写失败测试**

创建 `tests/ut/test_pd_scheduler_migration.py`：

```python
import importlib
from types import SimpleNamespace


def test_pd_separated_scheduler_module_is_owned_by_vllm_ascend():
    module = importlib.import_module("vllm_ascend.core.pd_separated_scheduler")

    assert module.PDSeparatedScheduler.__module__ == "vllm_ascend.core.pd_separated_scheduler"
    assert module.AsyncPDSeparatedScheduler.__module__ == "vllm_ascend.core.pd_separated_scheduler"


def test_passive_scheduler_module_is_owned_by_vllm_ascend():
    module = importlib.import_module("vllm_ascend.core.passive_scheduler")

    assert module.PassiveScheduler.__module__ == "vllm_ascend.core.passive_scheduler"
    assert module.DispatchPolicy.EXPECT_ALTERNATION.value == "expect_alternation"


def test_pd_scheduler_cls_is_set_to_ascend_path():
    from vllm_ascend.platform import NPUPlatform

    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_pd_separation=True,
            async_scheduling=False,
            scheduler_cls=None,
        )
    )

    NPUPlatform._configure_pd_separation_scheduler(vllm_config)

    assert vllm_config.scheduler_config.scheduler_cls == (
        "vllm_ascend.core.pd_separated_scheduler.PDSeparatedScheduler"
    )


def test_async_pd_scheduler_cls_is_set_to_ascend_path():
    from vllm_ascend.platform import NPUPlatform

    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_pd_separation=True,
            async_scheduling=True,
            scheduler_cls=None,
        )
    )

    NPUPlatform._configure_pd_separation_scheduler(vllm_config)

    assert vllm_config.scheduler_config.scheduler_cls == (
        "vllm_ascend.core.pd_separated_scheduler.AsyncPDSeparatedScheduler"
    )
```

- [ ] **步骤 3：运行测试确认失败**

运行：

```bash
cd vllm-ascend-pdmix && pytest -q tests/ut/test_scheduler_pd_separation_conflicts.py tests/ut/test_pd_scheduler_migration.py
```

预期：失败，原因包括 `vllm_ascend.core.pd_separated_scheduler` 不存在、`_configure_pd_separation_scheduler` 不存在、schema helper 不存在。

---

### 任务 2：迁移 scheduler 模块

**文件：**
- 创建：`vllm_ascend/core/pd_separated_scheduler.py`
- 创建：`vllm_ascend/core/passive_scheduler.py`

- [ ] **步骤 1：复制并调整 `PDSeparatedScheduler`**

从 `vllm-pdmix/vllm/v1/core/sched/pd_separated_scheduler.py` 复制到 `vllm-ascend-pdmix/vllm_ascend/core/pd_separated_scheduler.py`。

保持这些 import 来自 vLLM core：

```python
from vllm.v1.core.sched.output import BatchType, HiddenChannelType, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
```

- [ ] **步骤 2：复制并调整 `PassiveScheduler`**

从 `vllm-pdmix/vllm/v1/core/sched/passive_scheduler.py` 复制到 `vllm-ascend-pdmix/vllm_ascend/core/passive_scheduler.py`。

保持 schema import 来自 vLLM core：

```python
from vllm.v1.core.sched.output import BatchType, SchedulerOutput
```

- [ ] **步骤 3：运行模块导入测试**

运行：

```bash
cd vllm-ascend-pdmix && pytest -q tests/ut/test_pd_scheduler_migration.py::test_pd_separated_scheduler_module_is_owned_by_vllm_ascend tests/ut/test_pd_scheduler_migration.py::test_passive_scheduler_module_is_owned_by_vllm_ascend
```

预期：PASS。

---

### 任务 3：接入 scheduler_cls 和 schema fail-fast

**文件：**
- 修改：`vllm_ascend/platform.py`
- 修改：`vllm_ascend/scheduler_conflicts.py`

- [ ] **步骤 1：实现 schema helper**

在 `vllm_ascend/scheduler_conflicts.py` 增加：

```python
def _vllm_pd_scheduler_schema_available() -> bool:
    try:
        from vllm.v1.core.sched.output import BatchType, HiddenChannelType, SchedulerOutput
    except ImportError:
        return False

    required_batch_types = (
        "PD_MIX",
        "PURE_PREFILL",
        "PURE_DECODE",
        "EMPTY",
        "PREFILL_FIRST",
        "PREFILL_LAST",
        "DECODE_FIRST",
        "DECODE_LAST",
    )
    if any(not hasattr(BatchType, name) for name in required_batch_types):
        return False

    required_channels = ("PREFILL_1", "PREFILL_2", "DECODE")
    if any(not hasattr(HiddenChannelType, name) for name in required_channels):
        return False

    fields = getattr(SchedulerOutput, "__dataclass_fields__", {})
    return all(name in fields for name in ("batch_type", "head_token", "hidden_channel"))
```

然后在 `validate_pd_separation_scheduler_conflicts()` 的 PD enabled 分支加入：

```python
    if not _vllm_pd_scheduler_schema_available():
        raise ValueError(
            "scheduler_config.enable_pd_separation requires vLLM PD scheduler schema: "
            "BatchType, HiddenChannelType, and SchedulerOutput.batch_type/head_token/hidden_channel."
        )
```

- [ ] **步骤 2：实现 scheduler_cls 配置 helper**

在 `NPUPlatform` 类中增加：

```python
    @classmethod
    def _configure_pd_separation_scheduler(cls, vllm_config: VllmConfig) -> None:
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        if not getattr(scheduler_config, "enable_pd_separation", False):
            return
        if getattr(scheduler_config, "async_scheduling", False):
            scheduler_config.scheduler_cls = (
                "vllm_ascend.core.pd_separated_scheduler.AsyncPDSeparatedScheduler"
            )
        else:
            scheduler_config.scheduler_cls = (
                "vllm_ascend.core.pd_separated_scheduler.PDSeparatedScheduler"
            )
```

并在 `check_and_update_config()` 调用 `validate_pd_separation_scheduler_conflicts()` 后调用：

```python
        cls._configure_pd_separation_scheduler(vllm_config)
```

- [ ] **步骤 3：运行相关测试**

运行：

```bash
cd vllm-ascend-pdmix && pytest -q tests/ut/test_scheduler_pd_separation_conflicts.py tests/ut/test_pd_scheduler_migration.py
```

预期：PASS。

---

### 任务 4：接入 PassiveScheduler 路径 shim

**文件：**
- 修改：`vllm_ascend/patch/platform/patch_serve_headless.py`
- 测试：`tests/ut/test_pd_scheduler_migration.py`

- [ ] **步骤 1：添加 shim 函数测试**

在 `tests/ut/test_pd_scheduler_migration.py` 增加：

```python
def test_install_passive_scheduler_shim_aliases_upstream_import():
    import sys
    import vllm_ascend.patch.platform.patch_serve_headless as patch_serve_headless

    patch_serve_headless._install_ascend_passive_scheduler_shim()

    assert sys.modules["vllm.v1.core.sched.passive_scheduler"] is sys.modules[
        "vllm_ascend.core.passive_scheduler"
    ]
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
cd vllm-ascend-pdmix && pytest -q tests/ut/test_pd_scheduler_migration.py::test_install_passive_scheduler_shim_aliases_upstream_import
```

预期：FAIL，`_install_ascend_passive_scheduler_shim` 不存在。

- [ ] **步骤 3：实现 shim**

在 `vllm_ascend/patch/platform/patch_serve_headless.py` 增加：

```python
def _install_ascend_passive_scheduler_shim() -> None:
    import sys
    import vllm_ascend.core.passive_scheduler as passive_scheduler

    sys.modules["vllm.v1.core.sched.passive_scheduler"] = passive_scheduler
```

并在 `_launch_passive_engine_core()` 中 import `PassiveEngineCoreProc` 前调用：

```python
    _install_ascend_passive_scheduler_shim()
    from vllm.v1.engine.core import PassiveEngineCoreProc
```

- [ ] **步骤 4：运行 shim 测试**

运行：

```bash
cd vllm-ascend-pdmix && pytest -q tests/ut/test_pd_scheduler_migration.py::test_install_passive_scheduler_shim_aliases_upstream_import
```

预期：PASS。

---

### 任务 5：验证和收尾

**文件：**
- 所有修改文件

- [ ] **步骤 1：运行定向单测**

运行：

```bash
cd vllm-ascend-pdmix && pytest -q tests/ut/test_scheduler_pd_separation_conflicts.py tests/ut/test_pd_scheduler_migration.py tests/ut/patch/platform/test_patch_serve_headless.py
```

预期：PASS。

- [ ] **步骤 2：运行语法检查**

运行：

```bash
cd vllm-ascend-pdmix && python -m compileall vllm_ascend/core/pd_separated_scheduler.py vllm_ascend/core/passive_scheduler.py vllm_ascend/scheduler_conflicts.py vllm_ascend/platform.py vllm_ascend/patch/platform/patch_serve_headless.py
```

预期：PASS，输出不包含 SyntaxError。

- [ ] **步骤 3：检查没有新的 vLLM scheduler 反向依赖**

运行：

```bash
cd vllm-ascend-pdmix && python - <<'PY'
from pathlib import Path
bad = []
for path in Path('vllm_ascend').rglob('*.py'):
    text = path.read_text(encoding='utf-8')
    if 'vllm.v1.core.sched.pd_separated_scheduler' in text:
        bad.append(str(path))
print('\n'.join(bad))
raise SystemExit(1 if bad else 0)
PY
```

预期：无输出，退出码 0。
