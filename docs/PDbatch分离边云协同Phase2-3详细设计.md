# PD batch 分离边云协同推理 — Phase 2/3 详细设计文档

> 本文档基于《PDbatch分离分布式边云协同推理设计说明书》中 4.3 节 Phase 2、Phase 3 的功能点进行细化设计，并采纳如下三点调整：
>
> 1. `PDSeparatedScheduler` 的 `chunk_prefill` 队列更名为 `chunk_prefill_first`，明确语义为"P 首段尚未做完的请求"。
> 2. 边侧与云测之间建立**两条**独立的 ZMQ 通道，用于双向传输 `SchedulerOutput`；两条通道**统一采用 PUSH/PULL + 队列桥接**的对称模式。
> 3. `PDSeparatedScheduler` 新增 `prefills_last_ready` 和 `decodes_last_ready` 两个队列，承接云测做完中间层后回传给边侧的 SchedulerOutput，作为 P 尾 / D 尾段在边侧的入队点。
>
> Phase 2 关注点：**P 首（边侧）→ P 中（云测）** 链路打通，含调度、ZMQ、hidden state 传输与日志。
> Phase 3 关注点：**P 中（云测）→ P 尾（边侧）** 回传链路打通，含 ZMQ 回传、`prefills_last_ready` 入队、P 尾段调度执行与采样。

---

## 0 术语与上下文

| 术语 | 含义 |
|------|------|
| 边侧 (Edge / rank0) | leader EngineCore，承担 Embedding + 首若干 Transformer 层 + 尾若干 Transformer 层 + LM Head + 采样 |
| 云测 (Cloud / rank1) | passive EngineCore，承担中间 Transformer 层 |
| P 首 / P 尾 | Prefill 的边侧首段执行 / 边侧尾段执行 |
| P 中 | Prefill 的云测中间段执行 |
| D 首 / D 尾 / D 中 | Decode 对应的三段（Phase 4 实现，本文档不展开） |
| **PRE_OUT 通道** | Edge → Cloud 方向的 `SchedulerOutput` ZMQ 通道（"pre-out" = pre-middle-stage output） |
| **POST_OUT 通道** | Cloud → Edge 方向的 `SchedulerOutput` ZMQ 通道（"post-out" = post-middle-stage output） |
| `chunk_prefill_first` | 边侧 P 首段尚未完成（仍需继续切 chunk 或正在跨节点流水的）请求列表，**原 `chunk_prefill` 改名** |
| `prefills_last_ready` | 边侧待执行 P 尾段的 `SchedulerOutput` 队列（云测做完 P 中后通过 POST_OUT 通道回传） |
| `decodes_last_ready` | 边侧待执行 D 尾段的 `SchedulerOutput` 队列（D 中回传后入队，Phase 4 真正使用，本期仅建结构） |

**当前基线（Phase 1 已完成）：**
- `--enable-edge-cloud` / `--enable-pd-separation` 配置链路打通
- 边侧加载首尾层、云测加载中间层（`LayerShardLoader`）
- 边侧 / 云测 TP 不均等，通信组按边云布局
- 单条 ZMQ 通道（`PPSchedulerZmqPublisher` PUSH on edge，`PPSchedulerZmqSubscriber` PULL on cloud），用于 Edge → Cloud 单向下发 `SchedulerOutput`
- `PassiveScheduler` 已能分类 `BatchType` 并把 `SchedulerOutput` 入队 `executor.rpc_broadcast_mq`
- `PDSeparatedScheduler.schedule` 已能在 Prefill / Decode 之间二选一，并对 `SchedulerOutput.batch_type` 打标（`PURE_PREFILL` / `PURE_DECODE` / `EMPTY`）

**当前距离 Phase 2/3 验收的差距：**
1. `BatchType` 只有 `PURE_PREFILL` / `PURE_DECODE`，没有"P 首 / P 尾 / D 首 / D 尾"的细粒度标签
2. 边 → 云只有一条单向通道；云 → 边无任何 `SchedulerOutput` 回传通路
3. 边侧 `PDSeparatedScheduler` 没有"P 尾就绪队列"，无法把云测回传的请求重新入队执行 LM Head
4. 边侧、云测的模型执行还是"整段 forward"，没有按 segment（首/中/尾）切分调用
5. hidden state 的跨节点 isend / irecv 未与"段切换"对齐

---

## 1 总体执行链路（Phase 2 + Phase 3 完整闭环）

```
┌──────────────────────────────  边侧 rank0  ──────────────────────────────┐
│  PDSeparatedScheduler                                                   │
│  ─ waiting / chunk_prefill_first / running                               │
│  ─ prefills_last_ready / decodes_last_ready  (新增)                       │
│                                                                          │
│  schedule() 决策一个段                                                    │
│   ├─ P 首  → SchedulerOutput(batch_type=PREFILL_FIRST)                   │
│   └─ P 尾  → SchedulerOutput(batch_type=PREFILL_LAST)                    │
│                                                                          │
│  EngineCore.step():                                                      │
│   ① 通过 PRE_OUT 通道 publish SchedulerOutput 给云测 (仅 PREFILL_FIRST)   │
│   ② 把 SchedulerOutput 入 executor.rpc_broadcast_mq                       │
│   ③ Worker 按 batch_type 走对应 segment：                                  │
│        PREFILL_FIRST → segment_a (head 层)                                │
│        PREFILL_LAST  → segment_e (tail 层) + sampler                      │
│   ④ segment_a 完成后 → isend hidden_states 给云测 (PP 通信组)              │
│   ⑤ segment_e 开始前 → irecv hidden_states 自云测                          │
└──────────────────────────────────────────────────────────────────────────┘
                  │ PRE_OUT (Edge→Cloud, PUSH/PULL)
                  │ hidden_state via PP isend/irecv
                  ▼
┌──────────────────────────────  云测 rank1  ──────────────────────────────┐
│  PassiveScheduler                                                        │
│  ─ ready_prefills / ready_pdmixes / ready_decodes / ready_empties        │
│  ─ schedule() → ScheduledBatch  (复用 Phase 1 实现，按 batch_type 透传)    │
│                                                                          │
│  PassiveEngineCoreProc.step():                                            │
│   ① poll_and_classify PRE_OUT 通道                                        │
│   ② schedule() 取出一个 PREFILL_FIRST 的 SchedulerOutput                   │
│   ③ 通过 POST_OUT 通道 publish SchedulerOutput 回传给边侧                  │
│        (此时 batch_type 已被云测改写为 PREFILL_LAST)                       │
│   ④ 把 SchedulerOutput 入 executor.rpc_broadcast_mq                       │
│   ⑤ Worker irecv hidden_states + segment_c forward + isend hidden_states │
└──────────────────────────────────────────────────────────────────────────┘
                  │ POST_OUT (Cloud→Edge, PUSH/PULL)
                  ▼
回到边侧 PDSeparatedScheduler.prefills_last_ready
```

---

## 2 BatchType 扩展（共享改造，Phase 2 完成）

> Phase 2 的所有调度、ZMQ、segment 调度逻辑都要以"四种细粒度 batch type"为分支依据，因此把它放在最前面。

### 2.1 BatchType 枚举扩展

文件：[`vllm/v1/core/sched/output.py`](vllm-pdmix/vllm/v1/core/sched/output.py)

```python
class BatchType(enum.Enum):
    PD_MIX        = "pd_mix"          # 兼容老逻辑
    PURE_PREFILL  = "pure_prefill"    # 兼容老逻辑 (非边云模式)
    PURE_DECODE   = "pure_decode"     # 兼容老逻辑 (非边云模式)
    EMPTY         = "empty"

    # ── 新增四种细粒度类型 (仅在 enable_pd_separation=True 时使用) ──
    PREFILL_FIRST = "prefill_first"   # 边侧执行首段 (head 层)
    PREFILL_LAST  = "prefill_last"    # 边侧执行尾段 (tail 层 + lm_head)
    DECODE_FIRST  = "decode_first"    # Phase 4 使用，本期仅占位
    DECODE_LAST   = "decode_last"     # Phase 4 使用，本期仅占位
```

**兼容性原则：** `PURE_PREFILL` / `PURE_DECODE` 仍保留，原生 PP / PDmix 模式继续使用。当 `enable_pd_separation=True` 时，`PDSeparatedScheduler` 输出的 `batch_type` 只会是 `PREFILL_FIRST` / `PREFILL_LAST` / `DECODE_FIRST` / `DECODE_LAST` / `EMPTY` 之一，不再产出 `PURE_PREFILL` / `PURE_DECODE`。

### 2.2 PassiveScheduler 分类规则更新

文件：[`vllm/v1/core/sched/passive_scheduler.py`](vllm-pdmix/vllm/v1/core/sched/passive_scheduler.py)

云测 PassiveScheduler 只关心**云测自己要执行的段**，即 P 中 / D 中。因此：

| 入站 batch_type | 路由到 PassiveScheduler 的队列 | 云测执行段 |
|----------------|--------------------------------|----------|
| `PREFILL_FIRST` | `ready_prefills` | segment_c (中间层 prefill) |
| `DECODE_FIRST` | `ready_decodes` | segment_c (中间层 decode) |
| `PURE_PREFILL` | `ready_prefills` | 兼容原生 PP / PDmix |
| `PURE_DECODE` | `ready_decodes` | 兼容原生 PP / PDmix |
| `PD_MIX` | `ready_pdmixes` | 兼容原生 PP / PDmix |
| `EMPTY` | `ready_empties` | sync only |

**注意：** 云测 PassiveScheduler **不会**接收 `PREFILL_LAST` / `DECODE_LAST`（这两类是边侧自留段，不通过 PRE_OUT 通道下发）。云测处理完 `PREFILL_FIRST` 后，会把同一个 `SchedulerOutput` 通过 POST_OUT 通道回传给边侧，由边侧改写为 `PREFILL_LAST` 并入 `prefills_last_ready` 队列。**`batch_type` 的改写发生在云测 publish 之前**（详见 §4.3）。

---

## 3 PDSeparatedScheduler 改造（Phase 2 + Phase 3 共用结构，分两步落地）

### 3.1 队列结构改造

文件：[`vllm/v1/core/sched/pd_separated_scheduler.py`](vllm-pdmix/vllm/v1/core/sched/pd_separated_scheduler.py)

#### 现状（基线）

```python
self.chunk_prefill: list[Request] = []   # P 段尚未走完的请求
self.running: list[Request] = []          # D 段请求 (父类维护)
self.waiting: RequestQueue                # 新进请求 (父类维护)
```

#### 改造后（Phase 2 起，Phase 3 用到 `_last_ready`）

```python
# === 重命名：chunk_prefill → chunk_prefill_first ===
# 语义：P 首段已发起、但尚未在边侧拿到 P 中→P 尾的完整闭环结果的请求
self.chunk_prefill_first: list[Request] = []

# === 新增：P 尾 / D 尾就绪队列 ===
# 元素为云测回传的 SchedulerOutput（已带请求快照），边侧直接送入 executor 跑 segment_e
self.prefills_last_ready: deque[SchedulerOutput] = deque()
self.decodes_last_ready: deque[SchedulerOutput] = deque()   # Phase 4 才真正使用，本期建结构
```

> **为什么尾部就绪队列存的是 `SchedulerOutput` 而不是 `Request`？**
>
> 因为 P 尾段执行所需的元数据（KV block 表、采样参数、slot mapping 等）必须与 P 首段 / P 中段保持完全一致，否则 KV 索引会错位。最简单的做法就是云测把"原 `SchedulerOutput` + 改写后的 `batch_type=PREFILL_LAST`"原样回传，边侧直接拿来跑 segment_e，免去重建。

#### 全文件统一改名

| 旧名 | 新名 |
|------|------|
| `self.chunk_prefill` | `self.chunk_prefill_first` |
| 局部变量 `saved_chunk_prefill` | `saved_chunk_prefill_first` |
| 局部变量 `new_chunk_prefill` | `new_chunk_prefill_first` |
| 日志中的 `chunk_prefill[]` | `chunk_prefill_first[]` |

`__init__` / `_pick_prefill_batch` / `_pick_decode_batch` / `_migrate_prefill_to_running` / `_preempt_request` / `update_from_output` / `get_request_counts` / `get_num_unfinished_requests` / `finish_requests` / `reset_prefix_cache` / `make_stats` / `_handle_invalid_blocks` 全部跟改。

> **风险：** 旧测试用例 `tests/v1/core/test_pd_separated_scheduler.py`（若存在）会引用 `scheduler.chunk_prefill`。改名后需同步适配。计划在 Phase 2 实施步骤 1 中一并完成。

### 3.2 调度决策扩展（Phase 2 引入 P 首 / P 尾分支）

#### 调度阶段枚举扩展

```python
class SchedulingPhase(enum.Enum):
    PREFILL_FIRST = "prefill_first"   # 选 P 首
    PREFILL_LAST  = "prefill_last"    # 选 P 尾
    DECODE        = "decode"          # 选 D（Phase 4 再细化为 DECODE_FIRST/LAST）
```

#### `_select_scheduling_phase()` 改造

| 候选条件 | 优先级（Phase 2/3 暂用 prefill_last 优先） | 备注 |
|----------|-------------------------------------------|------|
| `prefills_last_ready` 非空 | **最高** | P 尾段已等待，优先 sample 出 token，释放 KV |
| `chunk_prefill_first` 非空 / `waiting` 非空 | 次高 | P 首段还有新工作可发 |
| `running` 非空 | 最低 | D 段（Phase 4 才真正生效） |

简化伪代码（`pd_scheduling_policy=prefill_first` 默认值下）：

```python
def _select_scheduling_phase(self) -> SchedulingPhase:
    if self.prefills_last_ready:
        return SchedulingPhase.PREFILL_LAST
    if self.chunk_prefill_first or self.waiting:
        return SchedulingPhase.PREFILL_FIRST
    if self.running:
        return SchedulingPhase.DECODE
    return SchedulingPhase.PREFILL_FIRST   # 兜底
```

> 真正的 1P1D / 2P1D 调度策略在 Phase 5 重写，本期只做"能跑通"的最简策略：**只要有 P 尾就先吃 P 尾**。理由：P 尾会释放采样后的 token、推进请求生命周期；积压 P 尾会导致 KV cache 长时间占用，反压 P 首。

#### `schedule()` 主入口

```python
def schedule(self) -> SchedulerOutput:
    phase = self._select_scheduling_phase()
    if phase == SchedulingPhase.PREFILL_LAST:
        return self._pick_prefill_last_batch()
    if phase == SchedulingPhase.PREFILL_FIRST:
        return self._pick_prefill_first_batch()
    return self._pick_decode_batch()   # Phase 4 才会真正进入
```

#### `_pick_prefill_first_batch()` （由原 `_pick_prefill_batch` 改造，Phase 2）

逻辑与原 `_pick_prefill_batch` 基本一致，仅做两处改动：

1. 把所有 `chunk_prefill` 引用改为 `chunk_prefill_first`。
2. 最终 `scheduler_output.batch_type` 写入：
   - `total_num_scheduled_tokens == 0` → `BatchType.EMPTY`
   - 否则 → `BatchType.PREFILL_FIRST`（替代原 `PURE_PREFILL`）

#### `_pick_prefill_last_batch()` （Phase 3 新增）

```python
def _pick_prefill_last_batch(self) -> SchedulerOutput:
    """从 prefills_last_ready 队列取一个云测回传的 SchedulerOutput。

    云测回传时已把 batch_type 改写为 PREFILL_LAST，并保留了原始
    KV block 分配 / sampling_params / num_scheduled_tokens 等元数据。
    边侧直接把它送给 executor 跑 segment_e + sampler 即可。
    """
    if not self.prefills_last_ready:
        return SchedulerOutput.make_empty()  # 防御性兜底
    so = self.prefills_last_ready.popleft()
    assert so.batch_type == BatchType.PREFILL_LAST
    # 注意：不要再次 super().schedule()，因为 KV block 在 P 首阶段已分配过；
    # 也不要把这些 req 加进 self.running，避免 D 段误调度。
    return so
```

#### `update_from_output` 改造

P 尾段完成后，请求要么"采样出最终 token / 进入 decode 阶段"（迁入 `self.running`），要么"prefill 全部 chunk 完成、生成第一个 token"（同样迁入 `self.running`）。这套逻辑父类 `Scheduler.update_from_output` 已经处理；本子类只需保证：

```python
def update_from_output(self, scheduler_output, model_runner_output):
    outputs = super().update_from_output(scheduler_output, model_runner_output)
    # 改名后过滤已完成请求
    self.chunk_prefill_first = [
        req for req in self.chunk_prefill_first if not req.is_finished()
    ]
    return outputs
```

P 尾路径的请求在父类 `update_from_output` 走完后会自然进入 `self.running`，不需要在子类里特别处理。

### 3.3 与 ZMQ 回传链路的衔接（Phase 3）

`prefills_last_ready` 的填充由 `EngineCore` 主循环负责（不是 `PDSeparatedScheduler` 内部主动 poll）：

```python
# vllm/v1/engine/core.py EngineCore 主循环 (rank0/edge)
def _process_engine_step(self):
    # 把 POST_OUT 通道里云测回传的 SchedulerOutput 灌进 scheduler
    if self._pp_post_subscriber is not None:
        for _seq, so in self._pp_post_subscriber.consume_new_outputs():
            assert so.batch_type in (BatchType.PREFILL_LAST, BatchType.DECODE_LAST)
            if so.batch_type == BatchType.PREFILL_LAST:
                self.scheduler.prefills_last_ready.append(so)
            else:
                self.scheduler.decodes_last_ready.append(so)
    # 然后正常 step
    ...
```

`PDSeparatedScheduler` 自己不持有任何 ZMQ socket，保持单一职责。

---

## 4 ZMQ 通道改造（Phase 2 完成双向 + 对称化）

> 设计原则：**两条通道统一使用 PUSH/PULL + queue.Queue 桥接的对称模式**。这样无论方向如何，发送端 caller 只往本进程的 `queue.Queue` 塞一个对象，pickle / send 在专门的后台线程做；接收端的 ZMQ recv 也在后台线程做，主线程只从 `queue.Queue` 取。CPU 与 GIL 与 ZMQ 的耦合降到最低，对称、好审计、易测。

### 4.1 通道布局

| 通道名 | 方向 | 绑定端 (bind) | 连接端 (connect) | 端口环境变量 | 默认端口 |
|--------|------|--------------|-----------------|-------------|---------|
| **PRE_OUT** | Edge → Cloud | 边侧 (rank0) `tcp://*:$PORT_PRE` | 云测 (rank1) `tcp://<edge_ip>:$PORT_PRE` | `VLLM_PP_PRE_OUT_ZMQ_PORT` | `5558` |
| **POST_OUT** | Cloud → Edge | 云测 (rank1) `tcp://*:$PORT_POST` | 边侧 (rank0) `tcp://<cloud_ip>:$PORT_POST` | `VLLM_PP_POST_OUT_ZMQ_PORT` | `5559` |

> 复用 Phase 1 已有的 `VLLM_PP_SCHEDULER_ZMQ_ADDR` 环境变量难以承载"双地址 + 双方向"语义，因此本期改为**两个端口 + master_addr** 推导：
> - 边侧绑定的 PRE_OUT 监听地址：`tcp://*:${VLLM_PP_PRE_OUT_ZMQ_PORT}`
> - 云测连接的 PRE_OUT 目标地址：`tcp://${master_addr}:${VLLM_PP_PRE_OUT_ZMQ_PORT}`
> - 云测绑定的 POST_OUT 监听地址：`tcp://*:${VLLM_PP_POST_OUT_ZMQ_PORT}`
> - 边侧连接的 POST_OUT 目标地址：`tcp://${cloud_addr}:${VLLM_PP_POST_OUT_ZMQ_PORT}`
>
> `master_addr` 已经在 `--master-addr` 中提供（边侧 IP）。`cloud_addr` 需要新增 `--cloud-addr` CLI 参数（云测节点 IP）。两端在启动时各自能计算出"我该 bind 什么、connect 什么"。
>
> 旧的 `VLLM_PP_SCHEDULER_ZMQ_ADDR` 保留作为单通道兼容回退，仅在 `enable_pd_separation=False` 时生效。

### 4.2 对称化的 `PUSH/PULL + 队列桥接` 抽象

文件：[`vllm/v1/engine/core.py`](vllm-pdmix/vllm/v1/engine/core.py)

现在的代码已经：
- `PPSchedulerZmqPublisher`：caller → `queue.Queue` → 后台线程 → ZMQ PUSH。已经是桥接模式 ✅
- `PPSchedulerZmqSubscriber`：ZMQ PULL → 直接写 `self._received_outputs` list（由 `PassiveScheduler` 后台线程再桥接到 `_inbox`）。**只是 Subscriber→PassiveScheduler 之间是桥接的，ZMQ→Subscriber 之间是直接 list 操作。**

本期把两端都做成**完全对称**的同一抽象 `PPSchedulerZmqChannel`：

```python
class PPSchedulerZmqChannel:
    """对称的双向 SchedulerOutput ZMQ 通道。

    每条通道都同时持有：
      - 一个发送侧 (push)：bind 或 connect 一个 PUSH socket，后台线程从
        outbound queue.Queue 拉取 SchedulerOutput → pickle → send
      - 一个接收侧 (pull)：bind 或 connect 一个 PULL socket，后台线程从
        socket recv → unpickle → 推入 inbound queue.Queue

    上层 caller 只接触两个 queue.Queue：
      - publish(so)  → outbound queue
      - consume_new_outputs() → 从 inbound queue 拉空

    构造时通过 (role, name) 决定 bind/connect 端：
      - role="bind":    bind PUSH 用于发送、bind PULL 用于接收
      - role="connect": connect PUSH 用于发送、connect PULL 用于接收

    Phase 2/3 实际用法（一条通道用单向，但代码结构对称）：
      - PRE_OUT 通道：边侧 send（bind PUSH）、云测 recv（connect PULL）
      - POST_OUT 通道：云测 send（bind PUSH）、边侧 recv（connect PULL）
      （也就是说每条通道只用了 push 或 pull 中的一边；保留双向能力是为
       了 Phase 4+ D 段以及未来错误反馈通道。）
    """

    SHUTDOWN_TIMEOUT: float = 2.0
    MAX_QUEUE_DEPTH: int = 1000

    def __init__(
        self,
        name: str,                                   # "pre_out" / "post_out"，用于日志
        send_endpoint: str | None,                    # None 表示本端不发送
        recv_endpoint: str | None,                    # None 表示本端不接收
        send_bind: bool,                              # send_endpoint 是 bind 还是 connect
        recv_bind: bool,                              # recv_endpoint 是 bind 还是 connect
    ) -> None:
        ...

    def publish(self, so: SchedulerOutput) -> None: ...
    def consume_new_outputs(self) -> list[tuple[int, SchedulerOutput]]: ...
    def shutdown(self) -> None: ...
```

**两条通道在边侧 / 云测的实例化：**

```python
# rank0 (Edge) ── 在 EngineCore.__init__ 中
self._pp_pre_channel = PPSchedulerZmqChannel(
    name="pre_out",
    send_endpoint=f"tcp://*:{pre_out_port}",  send_bind=True,
    recv_endpoint=None, recv_bind=False,      # 边侧不在 PRE_OUT 上接收
)
self._pp_post_channel = PPSchedulerZmqChannel(
    name="post_out",
    send_endpoint=None, send_bind=False,      # 边侧不在 POST_OUT 上发送
    recv_endpoint=f"tcp://{cloud_addr}:{post_out_port}", recv_bind=False,
)

# rank1 (Cloud) ── 在 PassiveEngineCoreProc.run_passive_engine_core 中
pp_pre_channel = PPSchedulerZmqChannel(
    name="pre_out",
    send_endpoint=None, send_bind=False,
    recv_endpoint=f"tcp://{edge_addr}:{pre_out_port}", recv_bind=False,
)
pp_post_channel = PPSchedulerZmqChannel(
    name="post_out",
    send_endpoint=f"tcp://*:{post_out_port}", send_bind=True,
    recv_endpoint=None, recv_bind=False,
)
```

> **绑定端选择原则：**
> - PRE_OUT 由边侧 bind（边侧作为"前端"对外暴露端口），云测 connect。
> - POST_OUT 由云测 bind，边侧 connect。
>
> 这样两端的 bind 集中在不同节点，避免端口竞争，也便于跨网段防火墙开放规则。

### 4.3 SchedulerOutput 在云测的"批型改写 + 回传"

在云测 `PassiveEngineCoreProc.step()` 内，对于从 PRE_OUT 通道接收并已调度（即将 enqueue 到 worker）的 `SchedulerOutput`，**在 enqueue 给 executor 之前**改写 `batch_type` 并 publish 到 POST_OUT 通道：

```python
# vllm/v1/engine/core.py PassiveEngineCoreProc.step (Phase 2/3 改造)
def step(self) -> bool:
    self.passive_scheduler.poll_and_classify()
    dispatched = False
    while True:
        batch = self.passive_scheduler.schedule()
        if batch.is_empty():
            break

        so = batch.scheduler_output

        # ── Phase 3 新增：先把 SchedulerOutput 回传给边侧 ──
        # 仅对会产生 P 尾 / D 尾段的批型回传
        if so.batch_type == BatchType.PREFILL_FIRST:
            so_for_edge = self._make_last_output(so, BatchType.PREFILL_LAST)
            self._pp_post_channel.publish(so_for_edge)
        elif so.batch_type == BatchType.DECODE_FIRST:
            so_for_edge = self._make_last_output(so, BatchType.DECODE_LAST)
            self._pp_post_channel.publish(so_for_edge)
        # EMPTY / PURE_* 兼容旧链路，不回传

        # 再正常下发给 executor 跑 segment_c
        for slice_info in batch.slices:
            payload = (so, slice_info) if slice_info is not None else (so,)
            self.executor.rpc_broadcast_mq.enqueue(
                (b"pp_scheduler_output", payload, {}, None)
            )
        dispatched = True
        if so.batch_type != BatchType.EMPTY:
            break
    return dispatched
```

**`_make_last_output()` 的实现策略**

最直接的实现是 `copy.copy(so)` 再改 `batch_type`：

```python
def _make_last_output(self, so, new_type) -> SchedulerOutput:
    import copy
    cloned = copy.copy(so)        # 浅拷贝，所有 list/dict/set 引用共享
    cloned.batch_type = new_type   # 改写
    return cloned
```

**为什么不深拷贝？**
- `SchedulerOutput` 是边云链路上的只读快照（边侧产生 → 云测消费 → 边侧消费完即弃），不会被任何一方原地改字段。
- pickle 时按值序列化，因此共享引用对边侧 / 云测两端进程而言反正是各自独立反序列化的副本。
- 浅拷贝足以在云测自己进程内同时拥有"自留段（PREFILL_FIRST）"和"回传段（PREFILL_LAST）"两个 batch_type 标签的两个引用。

**为什么先 publish 再 enqueue executor，而不是反过来？**
- publish 是发到 `outbound queue.Queue`，纳秒级，**非阻塞、不会被 worker 慢拖累**。
- 若反过来（先 enqueue executor，等 segment_c 执行完再回传），云测必须挂一个 callback 等 worker output，再触发 publish，链路冗长、状态多，且会让 P 尾段的就绪时间被 segment_c 的 forward 时延拉长。
- 边侧拿到 `prefills_last_ready` 入队后，**真正执行 segment_e 前会通过 PP 通信组 irecv hidden_states**（详见 §5.3）。这个 irecv 会自然等待云测 segment_c 完成。也就是说：调度信号（SchedulerOutput）通过 ZMQ 快速回传，数据信号（hidden_states）通过 PP 通信组按需等待，两者解耦。

### 4.4 端口环境变量与 envs.py 注册

文件：[`vllm/envs.py`](vllm-pdmix/vllm/envs.py)

```python
# 新增
VLLM_PP_PRE_OUT_ZMQ_PORT: int = 5558
VLLM_PP_POST_OUT_ZMQ_PORT: int = 5559

# 单通道兼容（已存在）保留：
VLLM_PP_SCHEDULER_ZMQ_ADDR: str | None = None
```

`environment_variables` 字典补：

```python
"VLLM_PP_PRE_OUT_ZMQ_PORT": lambda: int(
    os.getenv("VLLM_PP_PRE_OUT_ZMQ_PORT", "5558")
),
"VLLM_PP_POST_OUT_ZMQ_PORT": lambda: int(
    os.getenv("VLLM_PP_POST_OUT_ZMQ_PORT", "5559")
),
```

### 4.5 CLI 参数 `--cloud-addr`

文件：[`vllm/engine/arg_utils.py`](vllm-pdmix/vllm/engine/arg_utils.py)

在 parallel_group 中新增：

```python
parallel_group.add_argument(
    "--cloud-addr",
    type=str,
    default=None,
    help="Cloud node IP for edge to connect POST_OUT channel. "
         "Required when --enable-pd-separation is set.",
)
```

并写入 `ParallelConfig.cloud_addr`。校验：当 `enable_pd_separation=True` 时，`cloud_addr` 必填且非空。

> **可选简化方案：** 如果 Phase 2/3 验收环境里边侧、云测同一台机器（单机模拟双节点，PROCESS-LEVEL 隔离即可），可让 `cloud_addr` 默认 `127.0.0.1`，避免改 CLI；待真实双节点验证再开放。

---

## 5 模型执行段切分与 hidden_state 传输（Phase 2 + Phase 3）

> 本节涉及 `vllm-ascend-pdmix`。设计层面给出"worker 收到 `(SchedulerOutput, slice_info)` payload 后必须按 `batch_type` 走对应 segment 调用"的契约；具体 ascend 端代码改动由 ascend 维护方落地。

### 5.1 Worker 端的 batch_type → segment 分发

文件：`vllm-ascend-pdmix/vllm_ascend/worker/model_runner_v1.py`（参考 Phase 1 已迁入的 `_create_segment_callable`、`segment_a/c/e`）

```python
def execute_pp_scheduler_output(self, scheduler_output, slice_info=None):
    bt = scheduler_output.batch_type
    if is_edge_device():
        if bt == BatchType.PREFILL_FIRST:
            # 跑 segment_a，结束后 isend hidden_states 给云测
            out = self.segment_a_wrapper(...)
            self._isend_hidden_to_cloud(out)
            return None              # 不进入 sampler；调度器无需 ModelRunnerOutput
        elif bt == BatchType.PREFILL_LAST:
            # 跑 segment_e + sampler，先 irecv hidden_states
            intermediate = self._irecv_hidden_from_cloud()
            out = self.segment_e_wrapper(..., intermediate_tensors=intermediate)
            sampled = self.sampler(out)
            return sampled            # 返回正常 ModelRunnerOutput
        elif bt == BatchType.EMPTY:
            return self._sync_only_output(scheduler_output)
        else:
            raise NotImplementedError(f"edge: unhandled batch_type {bt}")
    else:  # cloud
        if bt in (BatchType.PREFILL_FIRST, BatchType.DECODE_FIRST):
            # 跑 segment_c，先 irecv，后 isend
            intermediate = self._irecv_hidden_from_edge()
            out = self.segment_c_wrapper(..., intermediate_tensors=intermediate)
            self._isend_hidden_to_edge(out)
            return None
        elif bt == BatchType.EMPTY:
            return self._sync_only_output(scheduler_output)
        else:
            raise NotImplementedError(f"cloud: unhandled batch_type {bt}")
```

### 5.2 segment 选择的 is_first / is_last 语义

复用 Phase 1 已迁入的 `_create_segment_callable` 签名：

| segment | 调用方 | 层范围 | `is_first_segment` | `is_last_segment` |
|---------|--------|--------|-------------------|-------------------|
| `segment_a` | 边侧 | `[0, head_k)` | True | False |
| `segment_c` | 云测 | `[head_k, N - tail_k)` | False | False |
| `segment_e` | 边侧 | `[N - tail_k, N)` | False | True |

边侧 `segment_a` 和 `segment_e` 是同一个 model 实例上不同 layer slice 的两个调用，由 batch_type 区分入口。

### 5.3 hidden_state 跨节点通信

使用 Phase 1 已建好的 **PP 通信组**（rank0 边侧 leader 与 rank `edge_npu_count` 云测 leader 在同一 PP 组）：

| 段 | 通信操作 | 通信组 | 备注 |
|----|---------|--------|------|
| 边侧 segment_a 完成 | `isend(IntermediateTensors)` to next rank in PP | `_PP` | 异步发送 hidden_states + residual |
| 云测 segment_c 入口 | `irecv(IntermediateTensors)` from prev rank in PP | `_PP` | 异步接收，wait 直到完成 |
| 云测 segment_c 完成 | `isend(IntermediateTensors)` to prev rank in PP | `_PP` | **方向反过来发回边侧** |
| 边侧 segment_e 入口 | `irecv(IntermediateTensors)` from next rank in PP | `_PP` | wait 完成后送入 segment_e |

> **复用 PP 通信组的合理性：** Phase 1 已经把 PP 通信组的成员设为 `[edge_rank0, cloud_rank0]`，组内只有这两个 rank，方向是抽象的（哪边是 "prev/next" 由 vLLM 内部约定）。对 segment_a→segment_c 用一个方向、对 segment_c→segment_e 用另一个方向即可。无需新通信组，待 Phase 6 引入双/三通道再细化。

### 5.4 Worker 与 SchedulerOutput 的 batch_type 透传

当前 `executor.rpc_broadcast_mq.enqueue((b"pp_scheduler_output", payload, ...))` 中 `payload = (scheduler_output,)` 或 `(scheduler_output, slice_info)`，worker 端反序列化后会读取 `scheduler_output.batch_type`。**Phase 2 不需要新增任何通信字段**，因为 `batch_type` 已经在 `SchedulerOutput` 里随 ZMQ 传过去了。

---

## 6 EngineCore 主循环改造（汇总，Phase 2 + Phase 3）

文件：[`vllm/v1/engine/core.py`](vllm-pdmix/vllm/v1/engine/core.py)

### 6.1 边侧 `EngineCore`（rank0）改动

```python
class EngineCore:
    def __init__(self, ...):
        ...
        # === Phase 1 旧 publisher 替换为 Phase 2 双通道 ===
        self._pp_pre_channel: PPSchedulerZmqChannel | None = None
        self._pp_post_channel: PPSchedulerZmqChannel | None = None
        pc = vllm_config.parallel_config
        if pc.enable_pd_separation:
            self._pp_pre_channel = PPSchedulerZmqChannel(
                name="pre_out",
                send_endpoint=f"tcp://*:{envs.VLLM_PP_PRE_OUT_ZMQ_PORT}",
                send_bind=True,
                recv_endpoint=None, recv_bind=False,
            )
            self._pp_post_channel = PPSchedulerZmqChannel(
                name="post_out",
                send_endpoint=None, send_bind=False,
                recv_endpoint=f"tcp://{pc.cloud_addr}:{envs.VLLM_PP_POST_OUT_ZMQ_PORT}",
                recv_bind=False,
            )
            logger.info(
                "Edge EngineCore: PRE_OUT publisher bound, POST_OUT subscriber "
                "connected to %s", pc.cloud_addr,
            )

    def step(self):
        # === Phase 3 新增：先把云测回传的 SchedulerOutput 灌进 scheduler ===
        if self._pp_post_channel is not None:
            for _seq, so in self._pp_post_channel.consume_new_outputs():
                if so.batch_type == BatchType.PREFILL_LAST:
                    self.scheduler.prefills_last_ready.append(so)
                elif so.batch_type == BatchType.DECODE_LAST:
                    self.scheduler.decodes_last_ready.append(so)
                else:
                    logger.warning(
                        "Edge EngineCore got unexpected batch_type %s on "
                        "POST_OUT channel; dropping", so.batch_type,
                    )

        # 原 schedule + execute 流程
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()

        # === Phase 2：只把 PREFILL_FIRST / DECODE_FIRST 下发给云测 ===
        # PREFILL_LAST / DECODE_LAST 是边侧自留段，不下发
        if self._pp_pre_channel is not None and scheduler_output.batch_type in (
            BatchType.PREFILL_FIRST, BatchType.DECODE_FIRST,
        ):
            self._pp_pre_channel.publish(scheduler_output)

        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        ...
```

### 6.2 云测 `PassiveEngineCoreProc`（rank1）改动

```python
@staticmethod
def run_passive_engine_core(vllm_config, ready_pipe):
    ...
    pp_pre_channel = None
    pp_post_channel = None
    pc = vllm_config.parallel_config
    if pc.enable_pd_separation:
        pp_pre_channel = PPSchedulerZmqChannel(
            name="pre_out",
            send_endpoint=None, send_bind=False,
            recv_endpoint=f"tcp://{pc.master_addr}:{envs.VLLM_PP_PRE_OUT_ZMQ_PORT}",
            recv_bind=False,
        )
        pp_post_channel = PPSchedulerZmqChannel(
            name="post_out",
            send_endpoint=f"tcp://*:{envs.VLLM_PP_POST_OUT_ZMQ_PORT}",
            send_bind=True,
            recv_endpoint=None, recv_bind=False,
        )
        logger.info(
            "Cloud PassiveEngineCore: PRE_OUT subscriber connected to %s, "
            "POST_OUT publisher bound", pc.master_addr,
        )
    ...
    # 把 pp_pre_channel 当作 PassiveScheduler 的 subscriber 源
    # 把 pp_post_channel 注入 PassiveEngineCoreProc，用于回传
    proc = PassiveEngineCoreProc(
        vllm_config, executor, pp_pre_channel,
        pp_post_channel=pp_post_channel,
        dispatch_policy=policy,
    )
    proc.run_busy_loop()
```

`PassiveEngineCoreProc.__init__` 增加 `pp_post_channel` 参数；`step()` 按 §4.3 实现"先 publish 回传，再 enqueue executor"。

### 6.3 `PassiveScheduler` 接口的小幅适配

原 `PassiveScheduler.__init__(vllm_config, pp_subscriber, ...)` 的 `pp_subscriber` 形参语义不变（"用来 consume_new_outputs"），实际传入的是 `PPSchedulerZmqChannel` 实例（duck-typed 即可，方法签名兼容）。

---

## 7 测试方案

### 7.1 单元测试

#### `tests/v1/core/test_pd_separated_scheduler.py`（如已存在则修改）

| 用例 | Phase | 断言 |
|------|-------|------|
| `test_chunk_prefill_renamed_to_chunk_prefill_first` | 2 | 旧字段不存在、新字段存在 |
| `test_pick_prefill_first_marks_batch_type_prefill_first` | 2 | 输出 `batch_type == PREFILL_FIRST` |
| `test_pick_prefill_last_pops_from_ready_queue` | 3 | `prefills_last_ready` 队列被 popleft，输出 `batch_type == PREFILL_LAST` |
| `test_select_phase_prefers_prefill_last_when_ready_nonempty` | 3 | 即使 `waiting` / `chunk_prefill_first` 非空也优先返回 `PREFILL_LAST` |
| `test_update_from_output_clears_finished_in_chunk_prefill_first` | 2 | 已 finished 的请求被清出 `chunk_prefill_first` |

#### `tests/v1/engine/test_zmq_channel.py`（新增）

| 用例 | Phase | 断言 |
|------|-------|------|
| `test_channel_publish_then_consume_inproc_tcp` | 2 | 同进程内 bind + connect 闭环，publish 一个 SchedulerOutput，另一侧 consume 拿到 |
| `test_channel_shutdown_drains_outbound` | 2 | shutdown 前 outbound 内的消息被 flush，线程 join 成功 |
| `test_channel_symmetric_both_directions` | 2 | 同一通道同时启用 send 和 recv，对发回环 |
| `test_channel_bridge_queue_full_drops_with_warning` | 2 | outbound queue 满时 publish 不阻塞，logger 有 warning |

#### `tests/v1/engine/test_passive_engine_core_proc.py`（扩展）

| 用例 | Phase | 断言 |
|------|-------|------|
| `test_step_publishes_prefill_last_to_post_channel_before_enqueue` | 3 | 给云测灌一个 `PREFILL_FIRST`，验证 `post_channel.publish` 被调用且 `batch_type == PREFILL_LAST`，且发生在 `executor.rpc_broadcast_mq.enqueue` 之前 |
| `test_step_does_not_publish_for_empty_or_pure_types` | 3 | `EMPTY` / `PURE_PREFILL` 不触发回传 |

### 7.2 集成测试（手工）

| 场景 | Phase | 步骤 | 期望日志 |
|------|-------|------|---------|
| 边侧 P 首下发 + 云测接收 | 2 | curl 单请求；观察边侧日志、云测日志 | 边侧打印 `[PD] _pick_prefill_first_batch done`; 云测 `PassiveScheduler classified batch_type=prefill_first` |
| 云测 P 中执行 + 回传 SchedulerOutput | 3 | 同上；继续观察 | 云测打印 `POST_OUT publish ... batch_type=prefill_last`；边侧打印 `prefills_last_ready append seq=...` |
| 边侧 P 尾执行 + 采样 | 3 | 同上；最终能拿到第一个 token | 边侧打印 `_pick_prefill_last_batch popped`; sampler 输出 token，curl 拿到响应起始字段 |

### 7.3 验收标准

**Phase 2 验收：**
- 边侧能产出 `batch_type == PREFILL_FIRST` 的 `SchedulerOutput` 并通过 PRE_OUT 通道发出
- 云测能通过 PRE_OUT 通道接收 `SchedulerOutput`，正确分类到 `ready_prefills`，并将其下发给 worker
- 云测 worker 执行 `segment_c`（hidden_state 接收 → 中间层 forward → hidden_state 发送）流程跑通，日志无报错
- 边侧 worker `segment_a` 完成后能 `isend` hidden_state；云测能 `irecv` 到

**Phase 3 验收：**
- 云测在 enqueue executor 前能通过 POST_OUT 通道回传 `batch_type == PREFILL_LAST` 的 `SchedulerOutput`
- 边侧能从 POST_OUT 通道接收，并把其追加到 `prefills_last_ready`
- 边侧 `PDSeparatedScheduler._select_scheduling_phase` 在 `prefills_last_ready` 非空时优先选择 `PREFILL_LAST`
- 边侧 worker 执行 `segment_e`（hidden_state 接收 → 尾部层 + LM Head → 采样）流程跑通
- 单请求 `curl` 能拿到至少一个 token 的响应（即 prefill 完整闭环 + 第一个 sampled token）

---

## 8 实施步骤（建议顺序）

| 顺序 | 任务 | Phase | 涉及文件 |
|------|------|-------|---------|
| 1 | `BatchType` 枚举扩展（新增 `PREFILL_FIRST/LAST` / `DECODE_FIRST/LAST`） | 2 | `vllm/v1/core/sched/output.py` |
| 2 | `PDSeparatedScheduler.chunk_prefill` → `chunk_prefill_first` 全文件改名 | 2 | `vllm/v1/core/sched/pd_separated_scheduler.py` + 现存测试 |
| 3 | `PDSeparatedScheduler` 新增 `prefills_last_ready` / `decodes_last_ready` 字段（仅建结构） | 2 | 同上 |
| 4 | `_pick_prefill_first_batch` 改名 + `batch_type=PREFILL_FIRST` 打标 | 2 | 同上 |
| 5 | `PassiveScheduler` 分类规则适配 `PREFILL_FIRST` / `DECODE_FIRST` → `ready_prefills` / `ready_decodes` | 2 | `vllm/v1/core/sched/passive_scheduler.py` |
| 6 | `PPSchedulerZmqChannel` 对称抽象实现（带单元测试） | 2 | `vllm/v1/engine/core.py` + 新增测试 |
| 7 | `envs.py` / `arg_utils.py` 注册 `VLLM_PP_PRE_OUT_ZMQ_PORT` / `VLLM_PP_POST_OUT_ZMQ_PORT` / `--cloud-addr` | 2 | `vllm/envs.py`, `vllm/engine/arg_utils.py`, `vllm/config/parallel.py` |
| 8 | 边侧 `EngineCore.__init__` / `step` 接入 PRE_OUT publish | 2 | `vllm/v1/engine/core.py` |
| 9 | 云测 `PassiveEngineCoreProc.run_passive_engine_core` 接入两条通道 | 2 | 同上 |
| 10 | Worker 端 `execute_pp_scheduler_output` 按 `batch_type` 分发 segment（ascend 侧落地） | 2 | `vllm-ascend-pdmix/vllm_ascend/worker/model_runner_v1.py` |
| 11 | hidden_state isend/irecv 与 segment 切换对齐 | 2 | 同上 |
| 12 | 单请求 P 首 + P 中链路联调（验收 Phase 2） | 2 | — |
| 13 | 云测 `PassiveEngineCoreProc.step` 在 enqueue 前 `_make_last_output` + POST_OUT publish | 3 | `vllm/v1/engine/core.py` |
| 14 | 边侧 `EngineCore.step` 入口处 consume POST_OUT 并填充 `prefills_last_ready` | 3 | 同上 |
| 15 | `PDSeparatedScheduler._select_scheduling_phase` 优先 `PREFILL_LAST` | 3 | `vllm/v1/core/sched/pd_separated_scheduler.py` |
| 16 | `_pick_prefill_last_batch` 实现 | 3 | 同上 |
| 17 | 边侧 worker `PREFILL_LAST` 段：irecv hidden → segment_e → sampler | 3 | `vllm-ascend-pdmix/.../model_runner_v1.py` |
| 18 | 单请求完整 P 首+P 中+P 尾闭环联调（验收 Phase 3） | 3 | — |

---

## 9 不在 Phase 2/3 范围

| 能力 | 推迟到 |
|------|--------|
| D 首 / D 中 / D 尾 完整链路 | Phase 4 |
| 1P1D / 2P1D batch 调度算法 | Phase 5 / 7 |
| PP 双通道拓展为三通道（Prefill / Decode 独立通信组） | Phase 6 |
| `prefills_last_ready` 的 backpressure（队列上限 + 反压到 PRE_OUT publish） | Phase 5 联合调度算法时一并设计 |
| `_pick_prefill_last_batch` 与 LM Head Tensor Parallel 的特殊优化 | 后续性能阶段 |
| ZMQ 通道的可观测性（消息延迟、积压告警、prometheus 指标） | Phase 5 联调期一并补 |

---

## 10 风险与注意事项

1. **`SchedulerOutput` 浅拷贝的语义**
   云测 `_make_last_output` 浅拷贝 `SchedulerOutput` 后改 `batch_type`。这是安全的当且仅当 `SchedulerOutput` 在 publish/enqueue 之后**没有任何一方对其字段做原地修改**。当前实现确实是只读消费，但后续若有 patch 在 worker 端原地改字段，需要在那里改成深拷贝或重新构造。

2. **PRE_OUT / POST_OUT 启动顺序**
   ZMQ PUSH bind 不需要等 PULL connect 才能 send（消息会先排队在 PUSH socket 的 HWM 内）。但 PULL connect 到不存在的 bind 时会安静等待。建议：
   - 边侧 EngineCore 在 `__init__` 中先 bind PRE_OUT，再连接 POST_OUT（云测的 bind 可能尚未起来 → connect 会重试，无问题）
   - 云测 PassiveEngineCore 在 `run_passive_engine_core` 中先 bind POST_OUT，再连接 PRE_OUT
   - HWM 设为 1000（与现有 publisher 一致），保证启动阶段消息不丢

3. **`prefills_last_ready` 与 `update_from_output` 的时序**
   边侧执行完 `PREFILL_LAST` 段（采样完）会进入 `update_from_output`，父类会把请求迁入 `self.running`。**但 `chunk_prefill_first` 里同请求的元数据要在 `_pick_prefill_last_batch` 时就清理掉**，否则父类 `update_from_output` 又会把它当 running 处理一次，状态错乱。
   解决：在 `_pick_prefill_last_batch` 内根据 `so.num_scheduled_tokens` 的 req_id 集合，把这些 req 从 `chunk_prefill_first` 中移除。

4. **PP 通信组方向**
   vLLM 现有 PP 是单向流水（rank0 → rank1），Phase 2 segment_a → segment_c 复用此方向；Phase 3 segment_c → segment_e 需要"反向"isend/irecv，需要确认昇腾 HCCL 在同一通信组内支持双向。如果不支持，Phase 6 的"双/三通道"会变成 Phase 3 的硬需求，需要前置规划。**建议 Phase 2 联调时先验证此点。**

5. **边侧 `chunk_prefill_first` 改名的兼容性**
   下游若有其他模块（如 metrics / dump_input / stats）通过 `getattr(scheduler, "chunk_prefill")` 访问，会失败。需要全局 grep 一遍 `chunk_prefill`，统一替换。

6. **`--cloud-addr` 在单机调试场景的便利性**
   单机模拟双节点（同主机两个进程）时，`--cloud-addr=127.0.0.1` 即可。建议在 CLI help 中标注该典型用法，避免单机调试同学误以为必须双节点。

7. **`PassiveScheduler` 现有的 16 个测试不受影响**
   PassiveScheduler 的对外契约（`poll_and_classify` / `schedule` / `ScheduledBatch`）未变化，仅分类规则把 `PREFILL_FIRST` / `DECODE_FIRST` 路由到现有 `ready_prefills` / `ready_decodes`，等价于把新枚举值并到原有桶。原有测试通过 `BatchType.PURE_PREFILL` / `PURE_DECODE` 触发，依然有效。**新增的两个细粒度类型在 PassiveScheduler 侧的路由需要新增 2 个用例**（已列在 7.1）。
