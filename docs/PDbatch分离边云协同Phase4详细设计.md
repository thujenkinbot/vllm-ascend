# PD batch 分离边云协同推理 — Phase 4 详细设计文档

> 本文档基于《PDbatch分离分布式边云协同推理设计说明书》中 4.3 节 Phase 4 的功能点进行细化设计：
>
> 1. 边侧 D 首请求下发执行成功
> 2. 云测 D 中请求下发执行成功
> 3. 边侧 D 尾请求下发执行成功
> 4. 单请求执行过程打通（curl 能够正常输出 prefill + 至少一轮 decode token）
>
> Phase 4 的目标是在 Phase 2/3 已搭好的 "P 首 → P 中 → P 尾" 闭环之上，把 decode 路径以**完全对称**的方式接入同一套基础设施：调度阶段、ZMQ 通道、POST_OUT 改写、PP 通信组、worker segment 分发，全部复用 Phase 2/3 的实现，只在"决策何时进入 D 段、把 batch_type 改成什么"以及"worker 对 DECODE_FIRST/LAST 的 segment 绑定"两处增量。

---

## 0 术语与上下文

| 术语 | 含义 |
|------|------|
| 边侧 (Edge / rank0) | leader EngineCore，承担 Embedding + 首若干 Transformer 层 + 尾若干 Transformer 层 + LM Head + 采样 |
| 云测 (Cloud / rank1) | passive EngineCore，承担中间 Transformer 层 |
| D 首 / D 尾 | Decode 的边侧首段执行（embedding 完整 + segment_a） / 边侧尾段执行（segment_e + sampler） |
| D 中 | Decode 的云测中间段执行（segment_c） |
| `decodes_last_ready` | 边侧待执行 D 尾段的 `SchedulerOutput` 队列；Phase 2 已建结构，**Phase 4 真正使用** |
| `DECODE_FIRST` / `DECODE_LAST` | 已在 Phase 2 加入 `BatchType` 枚举的两类批型，本期投入实际流水 |
| `SchedulingPhase.DECODE_LAST` | **Phase 4 新增**的调度阶段枚举值，与现有 `DECODE` 共同覆盖 D 段两端 |

### 当前基线（Phase 1 / 2 / 3 完成后）

- 边侧调度阶段：`PREFILL_FIRST` / `PREFILL_LAST` / `DECODE`
- 边侧产出的 `batch_type`：`PREFILL_FIRST` / `PREFILL_LAST` / `EMPTY`；**`PURE_DECODE` 仍是 `_pick_decode_batch` 的硬编码输出**
- POST_OUT 改写：云测 `PassiveEngineCoreProc._maybe_publish_post_out` 已经覆盖 `PREFILL_FIRST → PREFILL_LAST` **和** `DECODE_FIRST → DECODE_LAST` 两条规则（Phase 2/3 实现时一并加上，作为前置基础设施）
- POST_OUT 消费：边侧 `EngineCore` 主循环已经会把回传的 `PREFILL_LAST` 入 `prefills_last_ready`、把 `DECODE_LAST` 入 `decodes_last_ready`（同上）
- `PassiveScheduler` 分类规则：`DECODE_FIRST → ready_decodes`，`PURE_DECODE → ready_decodes`（同上）
- `decodes_last_ready: deque[SchedulerOutput]` 字段已存在但为空

### 距离 Phase 4 验收的差距

1. `PDSeparatedScheduler._pick_decode_batch` 仍打标 `PURE_DECODE`，需要在 PD 分离模式下改打 `DECODE_FIRST`
2. `PDSeparatedScheduler` 缺少 `_pick_decode_last_batch`，无法把 `decodes_last_ready` 里的 `SchedulerOutput` 取出投放给 executor
3. `SchedulingPhase` 枚举没有 `DECODE_LAST`；`_select_scheduling_phase` 没有对应优先级
4. `schedule()` 主入口缺少 `DECODE_LAST` 分支
5. Worker 端 `execute_pp_scheduler_output` 在 Phase 2/3 仅覆盖 `PREFILL_FIRST` / `PREFILL_LAST` / `PURE_DECODE`（兼容老链路），需要新增 `DECODE_FIRST` / `DECODE_LAST` 两个分支
6. decode 路径的 hidden_state PP isend/irecv 需要在 ascend 端按"每 token 1 行 hidden_size"形状对齐（与 prefill 的 `[total_seq_len, hidden_size]` 不同）

---

## 1 总体执行链路（Phase 4 完整闭环）

```
┌──────────────────────────────  边侧 rank0  ──────────────────────────────┐
│  PDSeparatedScheduler                                                   │
│  ─ waiting / chunk_prefill_first / running                               │
│  ─ prefills_last_ready / decodes_last_ready  (Phase 4 真正使用后者)       │
│                                                                          │
│  schedule() 决策一个段                                                    │
│   ├─ P 首  → SchedulerOutput(batch_type=PREFILL_FIRST)   [Phase 2]      │
│   ├─ P 尾  → SchedulerOutput(batch_type=PREFILL_LAST)    [Phase 3]      │
│   ├─ D 首  → SchedulerOutput(batch_type=DECODE_FIRST)    [Phase 4 新增] │
│   └─ D 尾  → SchedulerOutput(batch_type=DECODE_LAST)     [Phase 4 新增] │
│                                                                          │
│  EngineCore.step():                                                      │
│   ① POST_OUT consume → 同时填 prefills_last_ready / decodes_last_ready   │
│   ② scheduler.schedule() 得到 SchedulerOutput                            │
│   ③ 仅 PREFILL_FIRST / DECODE_FIRST 通过 PRE_OUT publish 给云测           │
│   ④ enqueue executor                                                     │
│   ⑤ Worker 按 batch_type 走 segment：                                    │
│        PREFILL_FIRST / DECODE_FIRST → segment_a                          │
│        PREFILL_LAST  / DECODE_LAST  → segment_e + sampler                │
└──────────────────────────────────────────────────────────────────────────┘
                  │ PRE_OUT  (Edge→Cloud, SchedulerOutput)
                  │ PP hidden_state (Edge→Cloud, isend/irecv)
                  ▼
┌──────────────────────────────  云测 rank1  ──────────────────────────────┐
│  PassiveScheduler                                                        │
│  ─ ready_prefills (含 PREFILL_FIRST) / ready_decodes (含 DECODE_FIRST)   │
│                                                                          │
│  PassiveEngineCoreProc.step():                                            │
│   ① poll_and_classify PRE_OUT 通道                                        │
│   ② schedule() 取出一个 PREFILL_FIRST / DECODE_FIRST 的 SchedulerOutput   │
│   ③ POST_OUT publish 回传 (PREFILL_FIRST→LAST / DECODE_FIRST→LAST)        │
│        [改写规则 Phase 2 已加 PREFILL，Phase 4 直接复用同函数的 DECODE 分支]│
│   ④ enqueue executor                                                     │
│   ⑤ Worker：irecv → segment_c → isend                                    │
└──────────────────────────────────────────────────────────────────────────┘
                  │ POST_OUT (Cloud→Edge, SchedulerOutput)
                  │ PP hidden_state (Cloud→Edge, isend/irecv)
                  ▼
回到边侧 PDSeparatedScheduler.decodes_last_ready
   → 下一轮 schedule() 优先弹出 → segment_e + sampler → next-token
   → next-token 写回 request → 下一轮 schedule() 又选 DECODE_FIRST
   → 直到 EOS / max_tokens
```

**关键观察：** P 链路与 D 链路在调度、通道、worker 三层全是同构的；区别只在 `chunk_prefill_first` 与 `running` 这两个父类队列的角色（D 路径全程从 `self.running` 里取请求）。

---

## 2 BatchType / SchedulingPhase 扩展

### 2.1 BatchType（已在 Phase 2 完成）

`vllm/v1/core/sched/output.py` 的 `BatchType` 已包含：

```python
PREFILL_FIRST = "prefill_first"
PREFILL_LAST  = "prefill_last"
DECODE_FIRST  = "decode_first"   # Phase 2 占位，Phase 4 启用
DECODE_LAST   = "decode_last"    # Phase 2 占位，Phase 4 启用
```

**Phase 4 不再改枚举**，只把"占位"变成"在生产代码路径上真正会被发出"。

### 2.2 SchedulingPhase（Phase 4 新增 `DECODE_LAST`）

文件：[`vllm/v1/core/sched/pd_separated_scheduler.py`](vllm-pdmix/vllm/v1/core/sched/pd_separated_scheduler.py)

```python
class SchedulingPhase(enum.Enum):
    PREFILL_FIRST = "prefill_first"
    PREFILL_LAST  = "prefill_last"
    DECODE        = "decode"          # = DECODE_FIRST 的发起端
    DECODE_LAST   = "decode_last"     # ★ Phase 4 新增
```

> **为什么不把 `DECODE` 改名为 `DECODE_FIRST`？**
> `SchedulingPhase.DECODE` 在 PD 分离 **关闭** 时也是有效的（对应单机 `BatchType.PURE_DECODE`）。改名会污染非边云模式的调用方。
> 折中策略：在 PD 分离启用时，`SchedulingPhase.DECODE` 等价于 "D 首发起"，由 `_pick_decode_batch` 内部按 `enable_pd_separation` 决定打 `DECODE_FIRST` 还是 `PURE_DECODE`。

---

## 3 PDSeparatedScheduler 改造

### 3.1 `_pick_decode_batch` 打标改造（Phase 4）

文件：[`vllm/v1/core/sched/pd_separated_scheduler.py`](vllm-pdmix/vllm/v1/core/sched/pd_separated_scheduler.py)

**当前实现（Phase 2/3 残留）：**

```python
def _pick_decode_batch(self) -> SchedulerOutput:
    so = super().schedule()
    if so.total_num_scheduled_tokens == 0:
        so.batch_type = BatchType.EMPTY
    else:
        so.batch_type = BatchType.PURE_DECODE      # ← 不区分边云
    return so
```

**Phase 4 改造后：**

```python
def _pick_decode_batch(self) -> SchedulerOutput:
    so = super().schedule()
    if so.total_num_scheduled_tokens == 0:
        so.batch_type = BatchType.EMPTY
    elif self.vllm_config.parallel_config.enable_edge_cloud:
        so.batch_type = BatchType.DECODE_FIRST     # 边云模式：发 D 首段给云测
    else:
        so.batch_type = BatchType.PURE_DECODE      # 单机模式：保持原有
    return so
```

> 与 `_pick_prefill_first_batch` 对称：那边也是按 `enable_edge_cloud` 选 `PREFILL_FIRST` 还是 `PURE_PREFILL`（Phase 2 已落地）。

### 3.2 `_pick_decode_last_batch`（Phase 4 新增，与 `_pick_prefill_last_batch` 对称）

```python
def _pick_decode_last_batch(self) -> SchedulerOutput:
    """从 decodes_last_ready 队列取一个云测回传的 SchedulerOutput。

    云测回传时已把 batch_type 改写为 DECODE_LAST，并保留了原始
    KV block / sampling_params / num_scheduled_tokens (每请求 1 token)
    等元数据。边侧直接送 executor 跑 segment_e + sampler。
    """
    if not self.decodes_last_ready:
        return SchedulerOutput.make_empty()  # 防御性兜底
    so = self.decodes_last_ready.popleft()
    assert so.batch_type == BatchType.DECODE_LAST
    # decode 段的请求始终在 self.running 中，不需要再从 chunk_prefill_first 移除；
    # 也不要重复 super().schedule()，KV block 在 D 首阶段已确认就绪。
    return so
```

**与 P 尾的差别（重要）：**

| 维度 | `_pick_prefill_last_batch` | `_pick_decode_last_batch` |
|------|----------------------------|---------------------------|
| 请求来自 | `chunk_prefill_first`（首段还未完成） | `self.running`（已进入 decode） |
| 处理后请求位置 | 父类 `update_from_output` 后迁入 `self.running` | 仍在 `self.running`，仅 `num_computed_tokens` +1 |
| 是否需要从 `chunk_prefill_first` 清理 | **是**（避免父类 `update_from_output` 双计） | **否**（请求不在 `chunk_prefill_first`） |
| KV block 分配 | P 首阶段已分配 | D 首阶段已分配（每 step 增量分配 1 block / 整批） |

### 3.3 `_select_scheduling_phase()` 扩展

**Phase 4 之前：**

```python
def _select_scheduling_phase(self) -> SchedulingPhase:
    if self.prefills_last_ready:
        return SchedulingPhase.PREFILL_LAST
    if self.chunk_prefill_first or self.waiting:
        return SchedulingPhase.PREFILL_FIRST
    if self.running:
        return SchedulingPhase.DECODE
    return SchedulingPhase.PREFILL_FIRST
```

**Phase 4 改造后：**

```python
def _select_scheduling_phase(self) -> SchedulingPhase:
    # ── 1. 尾段（已经从云测拿回 SchedulerOutput）优先 ──
    #     无论 P 尾 / D 尾，都该尽快采样以释放 KV 反压
    if self.prefills_last_ready:
        return SchedulingPhase.PREFILL_LAST
    if self.decodes_last_ready:
        return SchedulingPhase.DECODE_LAST

    # ── 2. 首段：按 pd_scheduling_policy 在 P 首 / D 首之间二选一 ──
    can_prefill = bool(self.chunk_prefill_first or self.waiting)
    can_decode  = bool(self.running)

    policy = self.scheduler_config.pd_scheduling_policy
    if policy == "decode_first" and can_decode:
        return SchedulingPhase.DECODE
    if policy == "prefill_first" and can_prefill:
        return SchedulingPhase.PREFILL_FIRST

    # 兜底：能 prefill 就 prefill，否则 decode
    if can_prefill:
        return SchedulingPhase.PREFILL_FIRST
    if can_decode:
        return SchedulingPhase.DECODE
    return SchedulingPhase.PREFILL_FIRST   # 全空兜底
```

**优先级说明：**

| 优先级 | 候选 | 原因 |
|--------|------|------|
| 1 (最高) | `PREFILL_LAST` | P 尾会把请求从 chunk_prefill_first 迁出，并产生首 token，释放 KV / 让请求进入 decode |
| 2 | `DECODE_LAST` | D 尾产生 next token，推进请求生命周期；积压 D 尾会让 KV cache 持续占用 |
| 3 | `PREFILL_FIRST` 或 `DECODE` | 按 `pd_scheduling_policy` 切换；Phase 4 不引入复杂调度，**只要能跑通** |
| 4 (兜底) | `PREFILL_FIRST` | 全空时返回，避免 `schedule()` 漏返回 |

> 真正的 1P1D / 2P1D 在 Phase 5 / 7 重写。Phase 4 这里只确保"D 尾不饿死、P 尾不饿死、首段能切换"三个不变式。

### 3.4 `schedule()` 主入口

```python
def schedule(self) -> SchedulerOutput:
    phase = self._select_scheduling_phase()
    if phase == SchedulingPhase.PREFILL_LAST:
        return self._pick_prefill_last_batch()
    if phase == SchedulingPhase.DECODE_LAST:         # ★ Phase 4 新增
        return self._pick_decode_last_batch()
    if phase == SchedulingPhase.PREFILL_FIRST:
        return self._pick_prefill_first_batch()
    return self._pick_decode_batch()
```

### 3.5 `update_from_output` 改造

边侧 D 尾执行完（采样完）后，`super().update_from_output` 会：

- 把 sampled token 写回 request
- `num_computed_tokens += 1`
- 检查是否 EOS / 达 max_tokens → finished
- 若 finished，从 `self.running` 移除

**Phase 4 不需要新增逻辑**，因为：

- D 尾的请求一直在 `self.running` 中
- `chunk_prefill_first` 已在 P 尾路径中清理过
- finished_req_ids 走父类同步路径

唯一需要确认的是 P 尾对 `chunk_prefill_first` 的清理已经覆盖；该清理在 Phase 3 已落地。

---

## 4 ZMQ 通道与 POST_OUT 改写（Phase 2/3 基础设施直接复用）

### 4.1 通道

PRE_OUT / POST_OUT 两条通道与 Phase 2/3 完全一致：

- PRE_OUT (Edge → Cloud)：边侧 publish `PREFILL_FIRST` / `DECODE_FIRST`
- POST_OUT (Cloud → Edge)：云测 publish `PREFILL_LAST` / `DECODE_LAST`

**Phase 4 不新增任何通道，不改任何环境变量。**

### 4.2 云测 POST_OUT 改写规则

`PassiveEngineCoreProc._maybe_publish_post_out` 在 Phase 2/3 实现时已经覆盖：

```python
def _maybe_publish_post_out(self, scheduler_output) -> None:
    if self._pp_pd_channel is None:
        return
    bt = scheduler_output.batch_type
    if bt == BatchType.PREFILL_FIRST:
        tail = replace(scheduler_output, batch_type=BatchType.PREFILL_LAST)
    elif bt == BatchType.DECODE_FIRST:
        tail = replace(scheduler_output, batch_type=BatchType.DECODE_LAST)
    elif bt == BatchType.EMPTY:
        tail = scheduler_output
    else:
        return
    self._pp_pd_channel.publish(tail)
```

**Phase 4 不需要改动这段代码**，验收阶段仅做实际链路验证。

### 4.3 边侧 POST_OUT consume

`EngineCore.step` 入口处的 POST_OUT 消费在 Phase 2/3 已经覆盖 `DECODE_LAST` 分支：

```python
if self._pp_post_channel is not None:
    for _seq, so in self._pp_post_channel.consume_new_outputs():
        if so.batch_type == BatchType.PREFILL_LAST:
            self.scheduler.prefills_last_ready.append(so)
        elif so.batch_type == BatchType.DECODE_LAST:
            self.scheduler.decodes_last_ready.append(so)
        else:
            logger.warning(...)
```

**Phase 4 不需要改动这段代码**。`decodes_last_ready` 在 Phase 4 之前一直没人往里取，Phase 4 由 `_pick_decode_last_batch` 把它消化掉。

---

## 5 模型执行段切分与 hidden_state 传输（Phase 4 增量）

> 本节涉及 `vllm-ascend-pdmix`。Phase 2/3 已经把 `PREFILL_FIRST` / `PREFILL_LAST` 接入 `execute_pp_scheduler_output`；Phase 4 在同一处增加 `DECODE_FIRST` / `DECODE_LAST` 两个对称分支。

### 5.1 Worker 端的 batch_type → segment 分发

文件：`vllm-ascend-pdmix/vllm_ascend/worker/model_runner_v1.py`

```python
def execute_pp_scheduler_output(self, scheduler_output, slice_info=None):
    bt = scheduler_output.batch_type
    if is_edge_device():
        if bt == BatchType.PREFILL_FIRST:                   # Phase 2
            out = self.segment_a_wrapper(scheduler_output)
            self._isend_hidden_to_cloud(out)
            return None
        elif bt == BatchType.PREFILL_LAST:                  # Phase 3
            intermediate = self._irecv_hidden_from_cloud()
            out = self.segment_e_wrapper(scheduler_output, intermediate_tensors=intermediate)
            return self.sampler(out)

        # ── ★ Phase 4 新增 ──
        elif bt == BatchType.DECODE_FIRST:
            # 与 PREFILL_FIRST 同构：embedding + segment_a → isend
            out = self.segment_a_wrapper(scheduler_output)
            self._isend_hidden_to_cloud(out)
            return None
        elif bt == BatchType.DECODE_LAST:
            # 与 PREFILL_LAST 同构：irecv → segment_e → sampler
            intermediate = self._irecv_hidden_from_cloud()
            out = self.segment_e_wrapper(scheduler_output, intermediate_tensors=intermediate)
            return self.sampler(out)

        elif bt == BatchType.EMPTY:
            return self._sync_only_output(scheduler_output)
        else:
            raise NotImplementedError(f"edge: unhandled batch_type {bt}")

    else:  # cloud
        if bt in (BatchType.PREFILL_FIRST, BatchType.DECODE_FIRST):   # Phase 2 + Phase 4
            intermediate = self._irecv_hidden_from_edge()
            out = self.segment_c_wrapper(scheduler_output, intermediate_tensors=intermediate)
            self._isend_hidden_to_edge(out)
            return None
        elif bt == BatchType.EMPTY:
            return self._sync_only_output(scheduler_output)
        else:
            raise NotImplementedError(f"cloud: unhandled batch_type {bt}")
```

**关键点：**

- 云测分支天然合并 `PREFILL_FIRST` 和 `DECODE_FIRST`，因为云测只关心"做 segment_c"这一件事，prefill / decode 仅影响 hidden_state 的 sequence 维大小（前者 `[total_seq_len, hidden]`，后者 `[batch_size, hidden]`，按 `num_scheduled_tokens` 推算即可）。
- 边侧 `segment_a_wrapper` / `segment_e_wrapper` 已经能从 `scheduler_output` 内推 token 形状，对 prefill / decode 不必特殊化。
- `_isend_hidden_to_cloud` / `_irecv_hidden_from_cloud` 的张量形状由 `IntermediateTensors` 内部 metadata 承载，prefill / decode 不影响发送方代码路径。

### 5.2 hidden_state 跨节点通信

复用 Phase 2/3 已建好的 PP 通信组（成员 `[edge_rank0, cloud_rank0]`），方向规则与 P 路径一致：

| 段 | 通信操作 | 通信组 | 形状 |
|----|---------|--------|------|
| 边侧 `DECODE_FIRST` 完成 | `isend(IntermediateTensors)` → cloud | `_PP` | `[batch_size, hidden_size]` |
| 云测 `DECODE_FIRST` 入口 | `irecv(IntermediateTensors)` from edge | `_PP` | `[batch_size, hidden_size]` |
| 云测 `DECODE_FIRST` 完成 | `isend(IntermediateTensors)` → edge | `_PP` | `[batch_size, hidden_size]` |
| 边侧 `DECODE_LAST` 入口 | `irecv(IntermediateTensors)` from cloud | `_PP` | `[batch_size, hidden_size]` |

> **形状差异不需要新通信组**：`IntermediateTensors` 是 dict[str, Tensor]，HCCL isend 内部按 tensor metadata 处理 shape，prefill / decode 共用同一 collective、同一通信组。Phase 6 引入"双/三通信组"是为了分离 Prefill / Decode 流量、避免同一组内 prefill 大包阻塞 decode 小包，**不是 Phase 4 的硬需求**。

### 5.3 与 PP 通信组方向反转的复用

Phase 3 已确认昇腾 HCCL 在同一通信组内支持 `segment_c → segment_e` 的"反向 isend/irecv"。Phase 4 直接复用此能力，**不再做额外验证**。若 Phase 3 联调发现方向反转有问题，需要把 Phase 6 的"双通信组"前置到 Phase 4 / 5 之间，但当前假设 Phase 3 已通过这一关卡。

---

## 6 EngineCore 主循环（Phase 2/3 实现直接生效，无需改动）

### 6.1 边侧 `EngineCore.step`

Phase 2/3 已实现：

- 入口处 consume POST_OUT，按 `PREFILL_LAST` / `DECODE_LAST` 入对应 ready 队列
- `scheduler.schedule()` 返回的 `SchedulerOutput`，只要 `batch_type ∈ {PREFILL_FIRST, DECODE_FIRST}` 就 PRE_OUT publish
- enqueue executor 不区分 batch_type，由 worker 自己分发

**Phase 4 不改 `EngineCore.step`。** `decodes_last_ready` 是 Phase 4 新晋"活跃路径"，而不是新基础设施。

### 6.2 云测 `PassiveEngineCoreProc.step`

Phase 2/3 已实现 `_maybe_publish_post_out` 对 `DECODE_FIRST → DECODE_LAST` 的改写。**Phase 4 不改云测主循环**。

---

## 7 测试方案

### 7.1 单元测试

#### `tests/v1/core/test_pd_separation.py`（扩展，Phase 4）

| 用例 | 断言 |
|------|------|
| `test_pick_decode_batch_tags_decode_first_when_edge_cloud` | `enable_edge_cloud=True` 时 `_pick_decode_batch` 输出 `batch_type == DECODE_FIRST` |
| `test_pick_decode_batch_tags_pure_decode_when_not_edge_cloud` | `enable_edge_cloud=False` 时回到 `PURE_DECODE`，保护非边云回归 |
| `test_pick_decode_last_pops_from_ready_queue` | 手工往 `decodes_last_ready` 塞一个 SchedulerOutput，验证 `_pick_decode_last_batch` 弹出且 `batch_type == DECODE_LAST` |
| `test_pick_decode_last_empty_when_no_ready` | `decodes_last_ready` 为空时返回 `SchedulerOutput.make_empty()`，不抛异常 |
| `test_select_phase_prefers_decode_last_over_first_segments` | `decodes_last_ready` 非空且 `running` / `waiting` 也非空时优先返回 `DECODE_LAST` |
| `test_select_phase_prefers_prefill_last_over_decode_last` | 两个 last 队列都非空时 `PREFILL_LAST` 优先（与 §3.3 优先级表对齐） |
| `test_decode_first_does_not_touch_chunk_prefill_first` | D 首调度后 `chunk_prefill_first` 不增不减 |

#### `tests/v1/engine/test_passive_engine_core_proc.py`（已在 Phase 2/3 覆盖 DECODE_FIRST，Phase 4 仅做回归验证）

无需新增用例。`test_post_out_publishes_decode_first_as_decode_last` 在 Phase 2/3 已通过（22/22 / 12/12 测试全绿）。

#### `tests/v1/core/test_passive_scheduler.py`（已在 Phase 2/3 覆盖 DECODE_FIRST 路由，Phase 4 不改）

无需新增用例。`test_classify_decode_first_routes_to_decode_queue` 已存在。

### 7.2 集成测试（手工）

| 场景 | 步骤 | 期望日志 |
|------|------|---------|
| 单请求 D 首下发 + 云测接收 | curl 单请求（max_tokens≥2），观察 P 首/中/尾完成后的下一个 step | 边侧 `[PD] _pick_decode_batch tagged DECODE_FIRST`；云测 `PassiveScheduler classified batch_type=decode_first` |
| 云测 D 中执行 + 回传 SchedulerOutput | 同上 | 云测 `POST_OUT publish ... batch_type=decode_last`；边侧 `decodes_last_ready append seq=...` |
| 边侧 D 尾执行 + 采样 | 同上 | 边侧 `_pick_decode_last_batch popped`；sampler 输出 token；curl 持续 stream / 拿到完整字符串 |
| 多 token 解码循环 | curl `max_tokens=20`，观察 D 首 ↔ D 尾交替 | 边侧反复打 `DECODE_FIRST` → `DECODE_LAST`；最终 finish_reason="length"/"stop" |

### 7.3 验收标准

**Phase 4 必达：**

- 边侧能产出 `batch_type == DECODE_FIRST` 的 `SchedulerOutput` 并 PRE_OUT publish
- 云测能接收 `DECODE_FIRST`、执行 segment_c、POST_OUT 回传 `DECODE_LAST`
- 边侧能从 `decodes_last_ready` 取出 `SchedulerOutput`、执行 segment_e + sampler、产出 next token
- 单请求 `curl` 能拿到 ≥ 2 个 token 的完整响应（含 prefill 闭环 + 至少一轮 decode 闭环）
- 多 token 解码不卡死、不死锁、不报错；`finish_reason` 正常（"length" 或 "stop"）

**Phase 4 不要求：**

- 多并发请求的吞吐量（Phase 5 调度算法的事）
- benchmark 性能（Phase 5+）
- prefill / decode 流量隔离（Phase 6）

---

## 8 实施步骤（建议顺序）

| 顺序 | 任务 | 涉及文件 |
|------|------|---------|
| 1 | `SchedulingPhase` 新增 `DECODE_LAST` 枚举值 | `vllm/v1/core/sched/pd_separated_scheduler.py` |
| 2 | `_pick_decode_batch` 按 `enable_edge_cloud` 切换 `DECODE_FIRST` / `PURE_DECODE` | 同上 |
| 3 | 新增 `_pick_decode_last_batch` 方法 | 同上 |
| 4 | `_select_scheduling_phase` 增加 `DECODE_LAST` 分支（最高优先级紧次于 PREFILL_LAST） | 同上 |
| 5 | `schedule()` 主入口路由 `DECODE_LAST` → `_pick_decode_last_batch` | 同上 |
| 6 | 补单元测试（§7.1 列出的 7 个用例） | `tests/v1/core/test_pd_separation.py` |
| 7 | Worker 端 `execute_pp_scheduler_output` 增加 `DECODE_FIRST` / `DECODE_LAST` 两个分支（ascend 侧落地） | `vllm-ascend-pdmix/vllm_ascend/worker/model_runner_v1.py` |
| 8 | 单请求 P 全闭环 + D 单轮联调（验证 D 首 → D 中 → D 尾 走通） | — |
| 9 | 多 token 解码联调（验证 D 首 ↔ D 尾循环不卡死） | — |
| 10 | curl 端到端验收（max_tokens=20） | — |

---

## 9 不在 Phase 4 范围

| 能力 | 推迟到 |
|------|--------|
| 1P1D batch 调度算法（边侧每轮在 P/D 段间智能切换） | Phase 5 |
| 云测 P 中 / D 中 batch 穿插调度（提升云测吞吐） | Phase 5 |
| PP 双通道拓展为三通道（Prefill / Decode hidden_state 独立通信组） | Phase 6 |
| 2P1D batch 调度算法（更激进的吞吐优化） | Phase 7 |
| `decodes_last_ready` 的 backpressure / 队列上限 | Phase 5（与调度算法一并设计） |
| decode segment 的 CUDA Graph 捕获优化（小 batch 高频 forward） | 后续性能阶段 |
| 多请求并发下的端到端延迟测试 / benchmark | Phase 5 验收期 |

---

## 10 风险与注意事项

### 10.1 `decodes_last_ready` 与 `running` 的状态一致性

D 尾 SchedulerOutput 里的请求始终在 `self.running` 中。云测改写 `batch_type=DECODE_LAST` 后回传，边侧弹出消费时**不能**再次 `super().schedule()`，否则父类会重新申请 KV block / 重新分配 slot mapping，造成 KV 错位。

**对策：** `_pick_decode_last_batch` 直接 `popleft` 后返回原 `SchedulerOutput`，不走父类 schedule。这与 `_pick_prefill_last_batch` 同构。

### 10.2 D 首 / D 尾 双向飞行中的请求计数

理论上一个请求可能"D 首已 PRE_OUT publish 出去、D 尾尚未 POST_OUT 回来"——也就是处于 in-flight 状态。此时如果上层调用 `get_num_unfinished_requests` / `make_stats`，请求应该被算作 unfinished。

**当前实现**：父类按 `self.running` 计数 unfinished decode 请求，in-flight 的 D 首请求仍在 `self.running` 中（`_pick_decode_batch` 不 `remove`），因此计数正确。**Phase 4 不需要新增 in-flight 计数器**。

### 10.3 单请求场景下 `_select_scheduling_phase` 的死锁风险

单请求场景下时序可能是：

```
step 1: P 首发出 → 等 P 中回传
step 2: 边侧 schedule() 时所有队列都空（waiting/chunk_prefill_first/running/*_last_ready 都空）
        → 返回 PREFILL_FIRST → _pick_prefill_first_batch → make_empty (EMPTY)
        → 不 publish PRE_OUT，本轮 idle
step 3: POST_OUT 回传 P 尾 → 边侧 prefills_last_ready 入队 → schedule() 选 PREFILL_LAST
...
```

`_select_scheduling_phase` 的兜底返回 `PREFILL_FIRST`、配合 `_pick_prefill_first_batch` 的空批 → `BatchType.EMPTY`，保证主循环不死锁（EMPTY 不 PRE_OUT publish，executor sync 一下即可）。

**Phase 4 不需要新增 idle 处理逻辑**。

### 10.4 PP 通信组方向反转在 decode 路径的稳定性

prefill 的 hidden_state 是 `[total_seq_len, hidden_size]`（典型几百 tokens），decode 是 `[batch_size, hidden_size]`（典型 1~32 tokens）。decode 流量小、频率高，HCCL 在同一通信组内的反向 isend/irecv 表现需要联调验证。

**对策：** Phase 4 联调时密切观察以下指标——

- decode 路径的 P50/P99 hidden_state 通信延迟（应远低于 prefill）
- 是否出现 P / D hidden_state 包在通信组内乱序（理论上同 socket FIFO，但需复核）
- 多轮 decode 是否触发 HCCL 资源泄漏（每 token 一次 isend/irecv）

若出现性能瓶颈或乱序问题，**前置 Phase 6 的双通信组改造**到 Phase 4 / 5 之间。

### 10.5 D 路径 sampler 的 EOS / max_tokens 触发

D 尾 sampler 触发 EOS 后，请求进入 finished 状态：

- 父类 `update_from_output` 把请求从 `self.running` 移除、加入 `finished_req_ids`
- 下一轮 `_select_scheduling_phase` 因 `running` 为空且 `_last_ready` 也空，落到 `PREFILL_FIRST` 兜底
- `finished_req_ids` 通过 `EMPTY` 批型同步给云测，云测据此释放对应资源

**对策：** Phase 4 联调时验证以下场景——

- 单请求 `max_tokens=20` 且无早停：触发 length finish
- 单请求碰到 EOS：触发 stop finish
- 两种 finish 后云测能正确释放 KV cache（通过 `EMPTY` 批型的 `finished_req_ids` 字段同步）

### 10.6 与现有 16 + 12 + 18 测试的兼容

- `test_passive_scheduler.py`（22 个测试）：DECODE_FIRST 已在 Phase 2 路由测试中覆盖，Phase 4 不动
- `test_passive_engine_core_proc.py`（12 个测试）：DECODE_FIRST → DECODE_LAST 改写已在 Phase 2 测试覆盖，Phase 4 不动
- `test_pd_separation.py`（18 个测试 + Phase 4 新增 7 个 = 25 个）：Phase 4 新增用例全在新的测试类中，旧用例不动

预期总测试数：22 + 12 + 25 = **59 个**，Phase 4 完成后全部通过。

### 10.7 `recv_object` 的隐含同步握手（decode 路径放大效应）

`GroupCoordinator.irecv_tensor_dict`（[`vllm/distributed/parallel_state.py:1031`](vllm-pdmix/vllm/distributed/parallel_state.py#L1031)）名字里的 `i` 只覆盖 **tensor body** 的 `torch.distributed.irecv`；它在第 1074 行先调用了 `recv_object(...)` 来收 metadata（dict 的 keys / shapes / dtypes / device 描述）。`recv_object` 内部连续做两次 **blocking** `torch.distributed.recv`（size + pickle body），走的是 cpu_group（gloo）。对端 `isend_tensor_dict` 的第一步同样是 blocking `send_object`。

**这构成边云两侧每次跨节点传输的 rendezvous 点。**

| 路径 | 单次 hidden body | 单次 metadata | 同步开销占比 |
|------|-----------------|--------------|------------|
| Prefill（P 首 ↔ P 中 ↔ P 尾） | 几 MB | 几百字节 | 低（被 body 时间稀释） |
| **Decode（D 首 ↔ D 中 ↔ D 尾）** | 几十 KB | 几百字节 | **高** —— 每 token 一次 metadata blocking RTT |

**对 Phase 4 的具体后果：**

1. **每个 decode token 都要付一次 metadata RTT**：D 首发送、D 中接收、D 中发送、D 尾接收，单 token 解码循环里**至少 2 次 metadata 握手**。
2. **POST_OUT "ZMQ-first" 优化被部分抵消**：Phase 2/3 设计让云测先 ZMQ 推送 `DECODE_LAST` 信号、后发 hidden_state，本意是让边侧 `_pick_decode_last_batch` 尽早拿到调度信号。但边侧真正执行 segment_e 时 `irecv_tensor_dict` 第一步仍要 blocking `recv_object`，边侧 worker 会在 segment_e 入口空转直到云测走完 `send_object`。
3. **与 §10.4 PP 反向通信稳定性叠加**：方向反转 + 每 token 同步握手 = 性能问题最容易在 Phase 4 联调时第一个暴露。

**Phase 4 处置：**

- **不修改 `parallel_state.py`** —— GroupCoordinator 是公共基础设施，影响面广，不属于 Phase 4 "最小可工作" 范围。
- **联调采集指标** —— 在 §10.4 风险验证的基础上加测：单 token decode 路径的 `recv_object` 阻塞时长，占整段 `irecv_tensor_dict + wait` 的比例。建议阈值：> 30% 时判定为 Phase 6 必须前置优化的信号。
- **记入 Phase 6 优化清单** —— 候选方案：metadata 缓存（按 `(comm_group, key_tuple, shape, dtype)` 签名，warmup 后跳过 send_object/recv_object）、metadata 内联进 hidden_state header、预协商 schema。建议与 Phase 6 三通信组改造合并完成。

> **不变式：** Phase 4 验收只看功能正确性（单请求 curl 能持续输出 token），`recv_object` 同步带来的延迟劣化不影响 Phase 4 PASS / FAIL 判定，但会被记录在联调报告中作为 Phase 5/6 的优化输入。

---

## 11 Phase 4 完成后的下一步

完成 Phase 4 后，PD 分离边云协同推理的"功能正确性"链路全部打通：

```
P 首 ↔ P 中 ↔ P 尾 → 首 token → D 首 ↔ D 中 ↔ D 尾 → next token → ... → EOS
```

Phase 5 起将聚焦"吞吐 / 调度算法"：

- 边侧 1P1D 调度算法：每轮 step 在 P 首 / P 尾 / D 首 / D 尾 四者间智能切换，避免某段饥饿
- 云测 PD 穿插调度算法：在 ready_prefills / ready_decodes 间根据流量动态选择
- benchmark 多请求验证：确保功能正确性下吞吐量符合预期
