# PD batch 分离边云协同推理 — Phase 2-4 Extend 详细设计文档

> 本文档作为 Phase 2 / Phase 3 / Phase 4 的**修正性补丁**：
> Phase 2/3/4 的上层调度契约（PF/PL/DF/DL 四段独立调度）已经按设计落地，
> 但**底层 `NPUWorker.execute_model` 在边侧把"P 首 + P 尾"强行合并到一次调用**，
> 与上层"两次独立 step"的契约不一致，导致 P 尾段实际投递到 worker 时永久阻塞。
>
> 本次扩展不引入新功能，**只把执行层切回与调度层一致的契约**：
>
> 1. `worker.execute_model` 严格按 `SchedulerOutput.batch_type` 选 segment
> 2. 边侧 P 首调用做完 segment_a 后立即返回，**不**等云回包、**不**跑 segment_e
> 3. 边侧 P 尾调用先 recv 云回包，再跑 segment_e + sampler
> 4. P 首与 P 尾跨调用共享的中间状态（input_batch、attn_metadata、cudagraph context）
>    引入"挂起态"机制，由 model_runner 跨调用持有
> 5. `EngineCore.step_with_batch_queue` 在边侧按 batch_type 决定是否紧跟 `sample_tokens`
>
> D 段（DECODE_FIRST / DECODE_LAST）按完全对称的方式同步修复。
>
> 适用前提：本次修正不动 PassiveScheduler、不动 ZMQ 通道、不动 POST_OUT 改写、
> 不动 PP 通信组——只动 `worker.py` / `model_runner_v1.py` 的边侧 segment 分发、
> `core.py` 的 `step_with_batch_queue` 在边侧的批型判定。

---

## 0 术语与上下文

| 术语 | 含义 |
|------|------|
| **P 首段** | 边侧 segment_a：embedding + 头 `head_k` 层 |
| **P 中段** | 云侧 segment_b/c：中间全部层（含 KV-cache、attn、MLP） |
| **P 尾段** | 边侧 segment_e：尾 `tail_k` 层 + lm_head + sampler |
| **D 首段** | 边侧 segment_a（decode shape：每请求 1 token） |
| **D 中段** | 云侧 segment_b/c（decode shape） |
| **D 尾段** | 边侧 segment_e + sampler（decode shape） |
| **HeadState** | 本次新增的边侧"P/D 首段挂起态"：跨两次 `execute_model` 调用持有 input_batch、attn_metadata、cudagraph 上下文等 |
| `_pp_send_work` | 边侧持有的上一轮 `isend_tensor_dict` Work 句柄列表，下一次 `execute_model` 入口处 `.wait()` |

### 当前基线（Phase 2/3/4 已合入后）

- 上层 `PDSeparatedScheduler` 已正确产出 `PREFILL_FIRST / PREFILL_LAST / DECODE_FIRST / DECODE_LAST` 四类独立的 `SchedulerOutput`
- `EngineCore.step()` / `step_with_batch_queue()` 已按 batch_type 把头段通过 PRE_OUT 发给云、尾段通过 POST_OUT 收回
- 云侧 `PassiveEngineCoreProc._maybe_publish_post_out` 已实现 `PREFILL_FIRST → PREFILL_LAST` / `DECODE_FIRST → DECODE_LAST` 改写
- 边侧 `EngineCore._drain_pd_channel_inbox` 已把回传段分别入 `prefills_last_ready` / `decodes_last_ready`
- 云侧 `NPUWorker.execute_model` 对称简单：`edge_cloud_broadcast_recv → model_runner.execute_model → isend_tensor_dict`，工作正常
- Fix A（云侧 sample_tokens 短路返回 EMPTY）已合入并需要保留

### 当前出错点（本次修复目标）

`vllm-ascend-pdmix/vllm_ascend/worker/worker.py:567-582` 边侧分支：

```python
if is_edge_device():
    if get_pp_group().world_size == 2:
        self._pp_send_work = get_pp_group().isend_tensor_dict(output.tensors)  # ① P 首 send
    tensor_dict, ... = edge_cloud_broadcast_recv()                              # ② 阻塞等云回包
    ...
    torch.npu.synchronize()
    output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)  # ③ P 尾 forward
    return output
```

**无视 `scheduler_output.batch_type`**，把 ①②③ 串在一次调用里。后果在 §1.2 详述。

---

## 1 问题分析

### 1.1 上下层契约对照

| 调度层（上层意图） | 执行层（worker 实际行为） |
|--------------------|--------------------------|
| 第 N 次 step 下发 `PREFILL_FIRST` | worker 跑 segment_a → isend → **等回包** → segment_e → 返回 |
| 第 N+1 次 step 下发 `PREFILL_LAST` | worker 又跑 segment_a → isend → **再次等回包**（永久阻塞） |

上层把 P 首 / P 尾当**两段独立 batch**，配合 PRE_OUT / POST_OUT 各一条 ZMQ。
底层却**完全无视 `batch_type`**，每次都按"a → 等回 → e"流水线跑。
两次调用都试图配对一遍 PP 通信——第一次能凑上，第二次没人对端。

### 1.2 卡死链路（PREFILL 为例，DECODE 同构）

**Step N（上层意图 = P 首）**

1. 边 EngineCore: schedule → 产出 `SO_PF` → ZMQ.publish(`SO_PF`) → executor.execute_model
2. 边 rank=0 worker: 进入 `worker.execute_model(SO_PF)`
   - segment_a ✓
   - **isend segment_a → 云** ✓
   - **571 行 `edge_cloud_broadcast_recv()` → 阻塞等云回 segment_c**
3. 云 PassiveEngineCore: 收 ZMQ `SO_PF` → 推 local MQ
4. 云 worker: 进入 `worker.execute_model(SO_PF)` → recv segment_a → segment_b/c → isend segment_c
5. 云 PassiveEngineCore: POST_OUT.publish(`SO_PL`)（PF 改写为 PL）
6. 边 worker 571 行解锁 → segment_e → 返回 ModelRunnerOutput（含真 sampled tokens）
7. 边 EngineCore: `future.result()` 取的是 sample_tokens 的 future（参见 §1.3），**ModelRunnerOutput 被丢**

**Step N+1（上层意图 = P 尾）**

8. 边 EngineCore: `_drain_pd_channel_inbox` 拿到 `SO_PL` → schedule → 产出 `SO_PL` → executor.execute_model
9. 边 rank=0 worker: **再次**进入 `worker.execute_model(SO_PL)`
   - segment_a ✗（重复且 batch_type 是 PL，本不该跑 segment_a）
   - **isend segment_a → 云**（云这次没人接：PL 不走 PRE_OUT，云 PassiveEngineCore 不会再推 local MQ）
   - **571 行 `edge_cloud_broadcast_recv()` → 永久阻塞**
10. 90 秒后 `response_mqs[0].dequeue` 超时 → 用户看到 `TimeoutError: RPC call to sample_tokens timed out`

### 1.3 timeout 报点指向 sample_tokens 的原因

`EngineCore.step_with_batch_queue` 在边侧每次 step 同时入队两条 RPC（[vllm-pdmix/vllm/v1/engine/core.py:905-923](vllm-pdmix/vllm/v1/engine/core.py#L905-L923)）：

```python
exec_future   = self.model_executor.execute_model(scheduler_output, non_block=True)
...
future        = self.model_executor.sample_tokens(grammar_output, non_block=True)
batch_queue.appendleft((future, scheduler_output, exec_future))
```

后续 953 行 `future.result()` 等的是 **sample_tokens** 的 future。
`response_mqs[0]` 是 FIFO，sample_tokens 的回包必须排在 execute_model 回包之后。
worker 卡 90s 在 execute_model 内 → response_mq 没人写 → sample_tokens 的 dequeue 超时 → **报错指向 sample_tokens，真实卡点在 execute_model**。

### 1.4 二级副作用

| # | 副作用 | 影响 |
|---|--------|------|
| 1 | Step N 的 sample_tokens 拿到 EMPTY，真 sampled tokens 被 `exec_future` 持有但被丢 | 即使 timeout 不发生，输出 token 也不对 |
| 2 | POST_OUT 通道形同虚设：边侧 PL 入队后一调度就死锁 | Phase 2/3 设计中"云改写 PF→PL 回传"的语义没用上 |
| 3 | 唯一"看起来工作"的配置 = 关掉 POST_OUT、让上层永远只下发 PF | 等价于退化为"非分离的纯 PP"，PD batch 分离的设计目标落空 |
| 4 | `is_pooling_model or not model_executed` 分支对 PF 也调 sample_tokens | sample_tokens 对 segment_a 输出（非 logits）无意义 |

---

## 2 设计目标

1. **执行层与调度层契约一致**：每次 `worker.execute_model` 只做一段，按 `SchedulerOutput.batch_type` 派发
2. **P 首立即返回**：发完 segment_a 就返回，**不**阻塞等云回包
3. **P 尾独立调用**：先 recv 云回包，再跑 segment_e + sampler
4. **跨调用状态完整保持**：P 首段已经构造好的 input_batch / attn_metadata / cudagraph 上下文，P 尾段直接复用
5. **D 段对称修复**：DECODE_FIRST / DECODE_LAST 走同一套机制
6. **不破坏标准 PP 路径**：非边云模式 (`enable_edge_cloud=False`) 走原 worker.execute_model 路径
7. **保留 Fix A**：云侧 sample_tokens 短路返回 EMPTY 不变
8. **回归非边云模式**：standard PP / TP / 单节点 worker_busy_loop 行为不变

非目标：
- 不动 PassiveScheduler / PDSeparatedScheduler 已有契约
- 不动 ZMQ 通道（PRE_OUT / POST_OUT / pp_scheduler_zmq）
- 不动 PP / TP 通信组的构建
- 不引入新的 batch_type / SchedulingPhase 枚举

---

## 3 总体执行链路（修复后）

```
┌──────────────────────────── 边侧 rank0 ────────────────────────────┐
│  Step N    (上层意图 = P 首)                                       │
│  EngineCore.step_with_batch_queue:                                 │
│    ① schedule → SO_PF                                              │
│    ② ZMQ.publish(SO_PF)         (PRE_OUT 通道)                     │
│    ③ execute_model(SO_PF)  → worker → segment_a → isend → return   │
│    ④ batch_type == PF → 不入 sample_tokens（仅 exec_future）       │
│    ⑤ HeadState 已挂起，等下一次 P 尾调用复用                       │
│                                                                    │
│  Step N+1  (上层意图 = P 尾，从 prefills_last_ready 来)            │
│  EngineCore.step_with_batch_queue:                                 │
│    ① _drain_pd_channel_inbox → SO_PL                               │
│    ② schedule → SO_PL                                              │
│    ③ 不 publish 给云（PL 是回传段，不再 PRE_OUT）                  │
│    ④ execute_model(SO_PL) → worker → broadcast_recv → segment_e    │
│                                       → sampler → 返回 MRO         │
│    ⑤ batch_type == PL → 入 sample_tokens（兜底，正常返回 EMPTY）   │
│       — 真 sampled tokens 已在 ④ 的 MRO 里                         │
└────────────────────────────────────────────────────────────────────┘
                  │ (期间) ZMQ POST_OUT 通道把 PF 改写为 PL 回边
                  ▼
┌──────────────────────────── 云侧 rank1 ────────────────────────────┐
│  对称简单：保持现状不动                                              │
│  worker.execute_model(SO_PF):                                       │
│    edge_cloud_broadcast_recv → segment_b/c → isend → return         │
│  PassiveEngineCoreProc.step():                                      │
│    POST_OUT.publish(SO_PL)                                          │
└────────────────────────────────────────────────────────────────────┘
```

D 段链路完全同构：把 PF/PL 换成 DF/DL，调用栈结构相同。

---

## 4 NPUWorker.execute_model 改造（核心）

文件：[`vllm-ascend-pdmix/vllm_ascend/worker/worker.py`](vllm-ascend-pdmix/vllm_ascend/worker/worker.py)

### 4.1 当前实现的结构

`execute_model` 当前混在一起的逻辑可以拆分成 4 个动作：

| 动作 | 当前行号 | 说明 |
|------|---------|------|
| (A) 等上轮 isend 完成 | 505-508 | `self._pp_send_work` |
| (B) 收上游 | 517-546 | 按角色：云侧 `broadcast_recv`、非首 rank `irecv` |
| (C) forward | 551-552 | `model_runner.execute_model(...)` |
| (D) 发下游 / 自循环再跑 | 562-600 | 边侧路径在 567-582 自循环跑 segment_e |

### 4.2 改造后的执行表

边侧加上 batch_type 派发后，(B)(C)(D) 的具体内容由 `batch_type` 决定：

| batch_type | (A) 等 isend | (B) 收上游 | (C) forward | (D) 发下游 / 后处理 | 返回 |
|------------|--------------|------------|-------------|--------------------|------|
| `PREFILL_FIRST` / `DECODE_FIRST`（边侧 P/D 首） | ✓ | × | segment_a | `isend` 给云；HeadState 挂起 | **EMPTY_MODEL_RUNNER_OUTPUT_HEAD** |
| `PREFILL_LAST` / `DECODE_LAST`（边侧 P/D 尾） | ✓ | `edge_cloud_broadcast_recv()` | segment_e | sampler；HeadState 释放 | **ModelRunnerOutput**（含真 sampled tokens） |
| 云侧（任何 batch_type） | ✓ | `edge_cloud_broadcast_recv()` | segment_b/c | `isend` 给边 | None / EMPTY |
| 非边云模式（兜底） | 同当前 | 同当前 | 同当前 | 同当前 | 同当前 |

注意：返回的 "EMPTY_MODEL_RUNNER_OUTPUT_HEAD" 不是新枚举，仍是 [model_runner_output.py 中的 `EMPTY_MODEL_RUNNER_OUTPUT`](vllm-pdmix/vllm/v1/outputs/model_runner_output.py)，只是命名上强调"这一次 step 上层不该把它当作最终结果"。

### 4.3 派发骨架（伪代码）

```python
def execute_model(self, scheduler_output, layer_slice_info=None):
    self._wait_prev_pp_send()                          # (A)

    # 非边云模式直接走原路径
    if not self._edge_cloud_enabled:
        return self._execute_model_legacy(
            scheduler_output, layer_slice_info,
        )

    bt = scheduler_output.batch_type

    if is_cloud_device():
        # 云侧不分首尾，单段执行
        return self._execute_model_cloud(scheduler_output)

    # ↓ 以下都是边侧
    if bt in (BatchType.PREFILL_FIRST, BatchType.DECODE_FIRST):
        return self._execute_model_edge_head(scheduler_output)

    if bt in (BatchType.PREFILL_LAST, BatchType.DECODE_LAST):
        return self._execute_model_edge_tail(scheduler_output)

    # 其它 batch_type（EMPTY / 单机 PURE_PREFILL 等）走兜底
    return self._execute_model_edge_legacy(scheduler_output, layer_slice_info)
```

四个子函数职责：

```python
def _execute_model_edge_head(self, scheduler_output):
    """边侧 P/D 首：segment_a → isend(含 head_token) → 挂起 HeadState → 立即返回 EMPTY"""
    intermediate = self.model_runner.execute_model(
        scheduler_output, intermediate_tensors=None,
    )
    assert isinstance(intermediate, IntermediateTensors)
    # 把 head_token 嵌入 intermediate tensors，供云侧原样回填到回传 hidden
    token = scheduler_output.head_token
    intermediate.tensors["_head_token"] = torch.tensor(
        list(bytearray(token, "utf-8")), dtype=torch.uint8, device="npu",
    )
    if get_pp_group().world_size == 2:
        self._pp_send_work = get_pp_group().isend_tensor_dict(intermediate.tensors)
    self.model_runner.suspend_head_state(scheduler_output)   # § 5.3
    return EMPTY_MODEL_RUNNER_OUTPUT

def _execute_model_edge_tail(self, scheduler_output):
    """边侧 P/D 尾：broadcast_recv → resume HeadState → segment_e + sampler"""
    tensor_dict, comm_handles, comm_postprocess = edge_cloud_broadcast_recv()
    intermediate = AsyncIntermediateTensors(tensor_dict, comm_handles, comm_postprocess)
    torch.npu.synchronize()
    # resume 内部做 control-plane vs data-plane head_token 一致性校验
    self.model_runner.resume_head_state(scheduler_output, intermediate)   # § 5.3
    output = self.model_runner.execute_model(scheduler_output, intermediate)
    assert isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput))
    return output

def _execute_model_cloud(self, scheduler_output):
    """云侧：保持当前对称实现，未变化"""
    tensor_dict, comm_handles, comm_postprocess = edge_cloud_broadcast_recv()
    intermediate = AsyncIntermediateTensors(tensor_dict, comm_handles, comm_postprocess)
    output = self.model_runner.execute_model(scheduler_output, intermediate)
    if isinstance(output, IntermediateTensors):
        if get_pp_group().world_size == 2:
            self._pp_send_work = get_pp_group().isend_tensor_dict(output.tensors)
    return output  # ModelRunnerOutput / None / EMPTY

def _execute_model_edge_legacy(self, scheduler_output, layer_slice_info):
    """兜底：非 PF/PL/DF/DL 的边侧批型（EMPTY 同步、PURE_PREFILL 等）走原路径"""
    return self._execute_model_legacy(scheduler_output, layer_slice_info)
```

### 4.4 与 layer-slicing 的关系

当前 worker 函数同时承担"边云 segment 分发"和"layer-slicing 分片调度"两件事。
本次改造**不**动 layer-slicing 路径——非边云模式的 `_execute_model_legacy` 完整保留 510-600 的所有逻辑。
边云模式与 layer-slicing **互斥**（边云已经按 segment 切分了，再 layer-slice 会冲突），在 `__init__` 处加 assert 防止误配。

---

## 5 NPUModelRunner 跨调用 HeadState（核心）

文件：[`vllm-ascend-pdmix/vllm_ascend/worker/model_runner_v1.py`](vllm-ascend-pdmix/vllm_ascend/worker/model_runner_v1.py)

### 5.1 现状问题

`_edge_cloud_forward_edge` 当前按 `intermediate_tensors is None` 决定跑 segment_a 还是 segment_e。
P 首调 forward 时为 None → 跑 segment_a；P 尾调 forward 时非 None → 跑 segment_e。
**但同一次 P 首 / P 尾**之间，model_runner 没有任何"中间态"需要保留——因为之前的实现里两次 forward 都在同一个 `worker.execute_model` 内部连续完成。

**改造后**两次 forward 落在**不同的 `worker.execute_model` 调用**里，期间 worker_busy_loop 会回到循环顶部处理其它消息（包括 EMPTY 类同步、check_health、可能的另一段 PD 批等）。所以要把**第一次调用结束时**已经准备好的下列状态显式挂起，**第二次调用开始时**再 resume：

| 字段 | 含义 | 是否必须跨调用保留 |
|------|------|------------------|
| `self.input_batch` 的 `req_ids / num_scheduled_tokens / sampling_params` | scheduler_output 中元数据派生 | **是** — P 尾 sampler 要用到 |
| `attn_metadata` | attention 元数据 | **是** — segment_e 要用 |
| `positions` / `logits_indices` | 位置/采样切片 | **是** |
| `cudagraph_stats` / `batch_desc` | 当前 batch 的 cudagraph 上下文 | **是** — segment_e 走 graph 时要用 |
| `kv_connector_output` | KV 传输完成标志 | **是** — sampler 后回传 |
| `_layerwise_intermediate` | layer-slicing 中间 | 否（与本次互斥） |

### 5.2 HeadState 数据类设计

新增数据类（同 `ExecuteModelState`，但语义不同）：

```python
@dataclass
class HeadState:
    """P/D 首段挂起态，跨两次 execute_model 调用持有。

    与 ExecuteModelState 的区别：
    - ExecuteModelState 是 last-rank 标准 PP 用的"算完 logits 等 sample_tokens"挂起态
    - HeadState 是边云模式下"做完 segment_a 等下一次 segment_e"挂起态
    """
    head_token: str                # ★ 跨通道配对键（见 §5.4）
    scheduler_output: SchedulerOutput
    req_ids: list[str]
    num_scheduled_tokens: int
    attn_metadata: Any
    positions: torch.Tensor
    logits_indices: torch.Tensor
    cudagraph_stats: Any
    batch_desc: Any
    kv_connector_output: Any
    sampling_metadata: Any
    # segment_a 输出的 hidden_states 不必保留——已经 isend 出去了
    # cloud 回传的 hidden_states 由 P 尾的 intermediate_tensors 参数传入
```

`NPUModelRunner` 增加字段：

```python
self._pending_head_states: dict[str, HeadState] = {}   # head_token → HeadState
```

### 5.3 `suspend_head_state` / `resume_head_state`

```python
def suspend_head_state(self, scheduler_output: SchedulerOutput) -> None:
    """P/D 首段结束时调用。把当前 input_batch 衍生的元数据全部打包挂起。

    要求：
    1. 调用此方法时 self.input_batch / self._latest_attn_metadata 等
       必须是 P 首 forward 刚刚使用过的状态。
    2. scheduler_output.head_token 已由 EngineCore 在调度时预分配（§5.4）。
    """
    token = scheduler_output.head_token
    assert token, "head_token must be pre-assigned by EngineCore before PF/DF dispatch"
    assert token not in self._pending_head_states, (
        f"head_token={token} already suspended; previous tail did not consume it"
    )
    head_state = HeadState(
        head_token=token,
        scheduler_output=scheduler_output,
        req_ids=list(self.input_batch.req_ids),
        num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
        attn_metadata=self._latest_attn_metadata,
        positions=self._latest_positions,
        logits_indices=self._latest_logits_indices,
        cudagraph_stats=self._latest_cudagraph_stats,
        batch_desc=self._latest_batch_desc,
        kv_connector_output=self.kv_connector_output,
        sampling_metadata=self._latest_sampling_metadata,
    )
    self._pending_head_states[token] = head_state

def resume_head_state(
    self,
    scheduler_output: SchedulerOutput,
    intermediate_tensors: IntermediateTensors,
) -> HeadState:
    """P/D 尾段开始时调用。按 head_token 取出 HeadState 并做跨通道一致性校验。

    控制面（ZMQ POST_OUT）与数据面（PP irecv_tensor_dict）是两条独立通道，
    2P1D 场景下必须显式校验它们携带的是同一个 head_token，否则静默错配。
    """
    token_ctrl = scheduler_output.head_token
    assert token_ctrl, "PL/DL scheduler_output must carry head_token from cloud"

    # ① 从 PP 数据面提取 token
    token_tensor = intermediate_tensors.tensors.get("_head_token")
    assert token_tensor is not None, (
        "intermediate_tensors missing '_head_token'; cloud worker must embed it"
    )
    token_data = bytes(token_tensor.cpu().numpy().tolist())
    token_pp = token_data.decode("utf-8")

    # ② 跨通道一致性断言
    assert token_ctrl == token_pp, (
        f"Control-plane vs data-plane head_token mismatch: "
        f"ZMQ={token_ctrl}, PP={token_pp}"
    )

    # ③ 按 token 从挂起池取出
    head_state = self._pending_head_states.pop(token_ctrl, None)
    assert head_state is not None, (
        f"No suspended HeadState for head_token={token_ctrl}; "
        f"tail dispatched without matching head or already consumed"
    )

    # ④ batch_type 配对校验
    expected_pair = self._expected_tail_batch_type(
        head_state.scheduler_output.batch_type,
    )
    assert scheduler_output.batch_type == expected_pair, (
        f"HeadState batch_type mismatch: head was "
        f"{head_state.scheduler_output.batch_type}, "
        f"tail is {scheduler_output.batch_type}"
    )

    # ⑤ 把挂起态重新挂回 model_runner 当前状态
    self.input_batch.restore_from(head_state)
    self._latest_attn_metadata = head_state.attn_metadata
    self._latest_positions = head_state.positions
    self._latest_logits_indices = head_state.logits_indices
    self._latest_cudagraph_stats = head_state.cudagraph_stats
    self._latest_batch_desc = head_state.batch_desc
    self.kv_connector_output = head_state.kv_connector_output
    self._latest_sampling_metadata = head_state.sampling_metadata
    return head_state

@staticmethod
def _expected_tail_batch_type(head_bt: BatchType) -> BatchType:
    return {
        BatchType.PREFILL_FIRST: BatchType.PREFILL_LAST,
        BatchType.DECODE_FIRST:  BatchType.DECODE_LAST,
    }[head_bt]
```

### 5.4 PL SchedulerOutput 与 HeadState 的协作协议

#### 5.4.1 两条数据通路分离

P 尾在 worker 侧同时消费两份云侧返回的数据，但走两条独立链路：

| 数据 | 通路 | 接收方 | 用途 |
|------|------|--------|------|
| **PL SchedulerOutput** | ZMQ POST_OUT（云 PassiveEC → 边 EngineCore） | EngineCore 调度循环 | 触发"该跑 P 尾了" |
| **hidden_states (tail)** | NPU PP `irecv_tensor_dict`（云 worker → 边 worker） | `edge_cloud_broadcast_recv()` | 作为 segment_e 的输入张量 |

PL 不带张量、只是调度信号；hidden 不带调度信息、只是张量。两者在 worker 侧汇合成一次完整的 P 尾。

#### 5.4.2 head_token 端到端贯穿

2P1D 场景下，控制面（ZMQ）与数据面（PP）**可能乱序**，必须用全局唯一的 `head_token` 做端到端配对：

```
Edge EngineCore (step N)
  │  ① 分配 head_token = uuid4().hex
  │  ② 写入 SO_PF.head_token
  │  ③ ZMQ PRE_OUT.publish(SO_PF)  ───────────► Cloud PassiveEC
  │                                              Cloud worker
  │  ④ execute_model(SO_PF)                      ⑤ recv hidden_a
  │     segment_a ──► isend(hidden_a + _head_token) ⑥ segment_b/c
  │     suspend_head_state(token)                ⑦ isend(hidden_e + _head_token)
  │                                              ⑧ POST_OUT.publish(SO_PL + token)
  │  ⑨ _drain_pd_channel_inbox → SO_PL (含 token)
  │  ⑩ execute_model(SO_PL)
  │        ⑪ broadcast_recv → hidden_e (含 _head_token)
  │        ⑫ resume_head_state(SO_PL, hidden_e)
  │              assert SO_PL.head_token == hidden_e._head_token
  │        ⑬ segment_e + sampler
```

**三个 Embedding 点**：
1. **SchedulerOutput**：EngineCore 在生成 PF/DF 时预分配 `head_token`，随 PRE_OUT 发给云；云 PassiveEngineCore 在 POST_OUT 回传 PL/DL 时**原样回填**。
2. **intermediate_tensors**：边 worker 在 segment_a 的 `isend_tensor_dict` payload 里附加 `_head_token` 张量（`torch.tensor(bytearray(token, "utf-8"), dtype=torch.uint8)`）；云 worker 在 segment_e 的 `isend_tensor_dict` 里**原样回填**同一 `_head_token`。
3. **model_runner**：`suspend_head_state` 按 token 把 HeadState 存入 `_pending_head_states`；`resume_head_state` 从 PL scheduler_output 取控制面 token、从 `intermediate_tensors` 取数据面 token，做一致性断言后配对。

#### 5.4.3 为什么必须是 head_token（而非 req_ids）

| # | 原因 | 在 1P1D | 在 2P1D |
|---|------|---------|---------|
| 1 | `_pending_head_states` 同时挂 ≥2 条 | 不会 | **会** |
| 2 | req_ids 唯一性依赖上层调度器不变量 | 退化可忽略 | 关键安全属性 |
| 3 | 控制面（ZMQ）× 数据面（PP）跨通道乱序检测 | 队列长度=1，不会乱 | **必须检测** |
| 4 | 配合未来 chunked-prefill / 抢占 / 多云路由 | 不涉及 | 直接相关 |

#### 5.4.4 异常情形

| 情形 | 处理 |
|------|------|
| PL 到达但 `_pending_head_states` 中无对应 token | assert → 云侧错发或 edge 重启后残留 |
| control-plane token 与 data-plane token 不一致 | assert → 跨通道乱序，立即报错，绝不静默 |
| HeadState 存在但久未到 PL（云侧挂了） | 加 TTL（如 30s），超时则 abort 并清理 HeadState |
| PL 的 req_ids 是 HeadState 的子集（preemption） | V2 优化：按子集裁剪后跑 segment_e；本期先 assert 等长 |
| PL 的 req_ids 与 HeadState 完全不交 | assert → 协议 bug |

### 5.5 `_edge_cloud_forward_edge` 改造

不再按 `intermediate_tensors is None` 派发，而是按调用方传入的 segment 角色：

```python
def _edge_cloud_forward_edge(
    self,
    num_tokens_padded: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor | None,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None,
    use_graph: bool,
    forward_context,
    *,
    segment: Literal["a", "e"],     # ★ 新增：显式声明本次跑哪一段
    **model_kwargs,
):
    if segment == "a":
        return self._forward_segment_a(
            num_tokens_padded, input_ids, positions, inputs_embeds,
            use_graph, forward_context, **model_kwargs,
        )
    elif segment == "e":
        assert intermediate_tensors is not None
        return self._forward_segment_e(
            num_tokens_padded, positions, intermediate_tensors,
            use_graph, forward_context, **model_kwargs,
        )
    raise ValueError(f"unexpected segment={segment}")
```

`execute_model` 顶层判断 `batch_type` 后传 `segment="a"` 或 `"e"`。
两个 `_forward_segment_*` 函数从当前 `_edge_cloud_forward_edge` 提取实现，**不**改算法本身。

### 5.6 cudagraph 与 input_batch 的边界

P 首和 P 尾跑的是**两个不同 layer 范围**的 forward，但**同一个 `scheduler_output` 实例**（云回传时只改 `batch_type`，其它字段不变）。所以：

- `input_batch` 的 `req_ids / num_scheduled_tokens / sampling_params` **不变** → resume 时直接 restore
- `attn_metadata` 已经按 P 首阶段的 layer 范围构造好 → P 尾段可以直接复用（层范围不同，但 page table / seq lens / block table 一致）
- `cudagraph_stats` / `batch_desc` 在两段间不变（segment_e 单独有自己的 cudagraph wrapper `seg_e_wrapper`）
- `_update_full_graph_params_if_needed` 在 P 尾 forward 入口仍要按 segment_e 的 layer_indices 调用一次

风险：P 首和 P 尾**中间**如果穿插了其它批的 forward（按 §6 的调度策略，理论上 PF 之后下一次 schedule 优先 PL，不应该插入其它批；但 EMPTY 同步可能插入），需要在 worker_busy_loop 层面保证 HeadState 期间不被其它批污染——见 §7。

---

## 6 EngineCore.step_with_batch_queue 改造（边侧）

文件：[`vllm-pdmix/vllm/v1/engine/core.py`](vllm-pdmix/vllm/v1/engine/core.py)

### 6.1 现状问题回顾

`step_with_batch_queue` 在 911-923 行无条件给非空批配一对 `(exec_future, sample_tokens future)`：

```python
exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
...
if self.is_pooling_model or not model_executed:
    future = exec_future
else:
    if not scheduler_output.pending_structured_output_tokens:
        future = self.model_executor.sample_tokens(grammar_output, non_block=True)
    ...
batch_queue.appendleft((future, scheduler_output, exec_future))
```

**P 首段的 segment_a 出的是中间 hidden_states，不是 logits**。对它调 sample_tokens 没意义，
还会把"真 sampled tokens 在 exec_future 里"的问题永久化。

### 6.2 改造点：按 batch_type 决定是否入 sample_tokens

```python
# 是否需要紧跟 sample_tokens
def _needs_sample_tokens(self, scheduler_output: SchedulerOutput) -> bool:
    bt = scheduler_output.batch_type
    if not self._is_edge_cloud:
        return True   # 非边云：标准 PP 行为
    # 边云模式下，只有 P/D 尾段才走 sampler
    return bt in (BatchType.PREFILL_LAST, BatchType.DECODE_LAST)
```

`step_with_batch_queue` 内：

```python
if self.is_pooling_model or not model_executed:
    future = cast(Future[ModelRunnerOutput], exec_future)
elif not self._needs_sample_tokens(scheduler_output):
    # 边云 P/D 首段：exec_future 直接当 future，sample_tokens 跳过
    future = cast(Future[ModelRunnerOutput], exec_future)
else:
    if not scheduler_output.pending_structured_output_tokens:
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        future = self.model_executor.sample_tokens(grammar_output, non_block=True)
    else:
        deferred_scheduler_output = scheduler_output
```

### 6.3 `step()` 路径

`step()` 路径（单批、PP=1 路径）当前是 "execute_model → 如果 None 才 sample_tokens"。
本次改造后边侧 P 首 worker 返回 EMPTY_MODEL_RUNNER_OUTPUT（非 None） → 不进 sample_tokens 分支，行为自然正确。
**`step()` 路径无需额外改动**。

### 6.4 batch_queue 容量与 P 首/尾顺序

`batch_queue_size = pp_size` = 2（边云模式 PP=2）。改造后：

- Step N 调度 PF → batch_queue 入 `(exec_future_PF, SO_PF, exec_future_PF)`
- Step N+1 调度 PL → 在入队之前先 pop → 取 Step N 的 result → 处理完
- 然后 Step N+1 入 `(sample_future_PL, SO_PL, exec_future_PL)`

容量 2 仍然够用（worst case 队里 1 个 PF + 1 个 PL 同时挂着）。
PF 段的 exec_future 在 worker 端是"isend 完即返回 EMPTY"，秒级返回；不会真的占用 90s。

---

## 7 worker_busy_loop 与 HeadState 的并发安全

文件：[`vllm-pdmix/vllm/v1/executor/multiproc_executor.py`](vllm-pdmix/vllm/v1/executor/multiproc_executor.py)

### 7.1 现状

`worker_busy_loop` 串行处理 `rpc_broadcast_mq` 与 `local_rpc_broadcast_mq`，云侧已恢复"收到 cross-node execute_model 就切到 local 模式"的隐式 barrier（已回退 Fix B）。
边侧 leader 不存在 local_rpc_broadcast_mq，每条消息按 FIFO 处理。

### 7.2 修复后是否需要改 worker_busy_loop？

**结论：不需要**。理由：

1. 边侧 leader 仍然是 FIFO 处理 execute_model：PF 先入队、PL 后入队（中间间隔一次 step）
2. PF 调用现在很快返回（isend 即走），不再阻塞 worker_busy_loop
3. PL 调用进入后阻塞在 `broadcast_recv` 等云回包，这是预期的，配对完成后释放
4. 中间可能穿插的批型只能是 EMPTY 类同步消息（不进 model_runner 的 forward），不污染 HeadState

### 7.3 防御性保险

为防止意外的批型穿插污染 HeadState，在 `worker._execute_model_edge_legacy`（兜底分支）入口加 assert：

```python
def _execute_model_edge_legacy(self, scheduler_output, layer_slice_info):
    assert self.model_runner._head_state is None, (
        f"HeadState is suspended (head batch_type="
        f"{self.model_runner._head_state.scheduler_output.batch_type}), "
        f"cannot run legacy batch_type={scheduler_output.batch_type} "
        f"in between head/tail pair"
    )
    return self._execute_model_legacy(scheduler_output, layer_slice_info)
```

EMPTY 批根本不会进 `forward` 路径（在 model_runner 入口 1990-2003 的 `if not scheduler_output.total_num_scheduled_tokens` 早退），所以不会触发这条 assert。

---

## 8 Fix A 的位置与作用

文件：[`vllm-ascend-pdmix/vllm_ascend/worker/model_runner_v1.py:2377-2391`](vllm-ascend-pdmix/vllm_ascend/worker/model_runner_v1.py#L2377-L2391)

Fix A（云侧 sample_tokens 短路）**保留不变**。理由：

- `EngineCore.sample_tokens` 通过 `collective_rpc(unique_reply_rank=0)` 广播给所有 worker
- 云侧 worker 仍然会从 cross-node MQ 收到 sample_tokens 这条 RPC
- Fix A 让云侧 worker 直接返回 EMPTY，避免误触发 PP 通信

本次改造让边侧 PF 段不调 sample_tokens，但**PL 段仍然调**——Fix A 仍然是云侧 sample_tokens 的安全网。

---

## 9 D 段对称处理

D 段沿用同一套机制，**不**需要额外代码：

- `_execute_model_edge_head` 同时处理 `PREFILL_FIRST` 和 `DECODE_FIRST`
- `_execute_model_edge_tail` 同时处理 `PREFILL_LAST` 和 `DECODE_LAST`
- HeadState 配对表 `_expected_tail_batch_type` 覆盖两对映射
- EngineCore `_needs_sample_tokens` 同时识别 `PREFILL_LAST` / `DECODE_LAST`

差别仅在 attention shape：decode 是 `[batch_size, 1, hidden_size]`，prefill 是 `[batch_size, seq_len, hidden_size]`。这部分已经在 Phase 4 model_runner 侧解决，本次改造不涉及。

---

## 10 改造文件清单

| 文件 | 改动类型 | 大致行数 | 说明 |
|------|---------|---------|------|
| `vllm-ascend-pdmix/vllm_ascend/worker/worker.py` | 重构 `execute_model` | +120 / -60 | 拆出 4 个子函数，按 batch_type 派发；isend 时嵌入 `_head_token` |
| `vllm-ascend-pdmix/vllm_ascend/worker/model_runner_v1.py` | 新增 HeadState + suspend/resume | +80 | `_edge_cloud_forward_edge` 拆成 segment_a / segment_e 两段；`_pending_head_states` dict |
| `vllm-pdmix/vllm/v1/engine/core.py` | `step_with_batch_queue` 加 batch_type 判定 + head_token 预分配 | +30 | `_needs_sample_tokens` 帮助方法；PF/DF 调度时预分配 `head_token` |
| `vllm-pdmix/vllm/v1/core/scheduler.py` (或 SchedulerOutput 定义处) | 新增字段 | +2 | `SchedulerOutput` 新增 `head_token: str \| None = None` |
| `vllm-pdmix/vllm/v1/engine/core.py` (PassiveEngineCoreProc) | POST_OUT 回填 head_token | +5 | `_maybe_publish_post_out` 把收到的 PF/DF head_token 原样写入 PL/DL |
| `vllm-pdmix/vllm/v1/executor/multiproc_executor.py` | 不动 | 0 | 已回退 Fix B，保持原 worker_busy_loop |
| `tests/v1/engine/test_step_with_batch_queue_edge_cloud.py` | 新增 | +150 | mock executor 验证 PF 不入 sample_tokens、PL 入 sample_tokens |
| `tests/ascend/worker/test_worker_segment_dispatch.py` | 新增 | +200 | mock model_runner 验证 PF/PL/DF/DL/cloud 四类 batch 派发正确 |
| `tests/ascend/worker/test_head_state_lifecycle.py` | 新增 | +200 | suspend/resume 配对、跨通道 token  mismatch assert、并发 2P1D 场景 |

---

## 11 实施步骤

按依赖顺序：

1. **`model_runner_v1.py`**：先落 HeadState 数据类 + suspend/resume + `_edge_cloud_forward_edge` 拆 segment_a / segment_e
   - 测试：`test_head_state_lifecycle.py` 单测全绿
2. **`worker.py`**：重构 `execute_model` 加 4 个派发分支
   - 测试：`test_worker_segment_dispatch.py` 单测全绿
3. **`core.py`**：`step_with_batch_queue` 加 `_needs_sample_tokens`
   - 测试：`test_step_with_batch_queue_edge_cloud.py` 单测全绿
4. **集成验证**：起边云服务，发单条 curl 请求
   - 期望：原 `TimeoutError: RPC call to sample_tokens timed out` 消失
   - 期望：用户能拿到 sampled token（prefill 一段 + 至少一轮 decode）
5. **回归**：跑非边云模式 standard PP / TP 测试，确认无回归

---

## 12 验证

```bash
# 单元测试
cd d:/Kisella/vllm_dev/vllm-pdmix
.venv/bin/python -m pytest tests/v1/engine/test_step_with_batch_queue_edge_cloud.py -v

cd d:/Kisella/vllm_dev/vllm-ascend-pdmix
.venv/bin/python -m pytest tests/ascend/worker/test_worker_segment_dispatch.py \
                            tests/ascend/worker/test_head_state_lifecycle.py -v

# 端到端
# (1) 起云侧
.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model <model> \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"cloud",...}}' \
    ...

# (2) 起边侧
.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model <model> \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge",...}}' \
    ...

# (3) curl
curl http://<edge_ip>:8000/v1/completions \
    -d '{"model":"<model>","prompt":"hello","max_tokens":16}'
```

期望：
- 边侧日志按顺序出现 `Send intermediate tensors to cloud` (P 首) → `Received intermediate tensors from edge` (云 P 中) → `Send intermediate tensors to edge` (云回包) → 边侧 segment_e 完成
- HTTP 返回 200 + 完整 generated text
- 无 `TimeoutError`

---

## 13 风险与回滚

### 13.1 风险

| 风险 | 缓解 |
|------|------|
| HeadState 字段不全，P 尾段读到旧/缺失的元数据 | 单测 `test_head_state_lifecycle.py` 覆盖所有字段；首版集成时在 resume 后加日志比对 |
| cudagraph context 跨调用失效 | 边云模式实测 segment_e 已经有独立的 `seg_e_wrapper`，cudagraph capture/replay 独立；若有问题先关 graph (`enable_decode_graph=False`) 验证 |
| ascend 仓引入的 worker.py 改动需要严格架构 review (per AGENTS.md) | 提交 PR 时按 Conventional Commits + `git commit -s`，PR 描述说明设计文档链接 |
| 真 sampled tokens 来源切换（从 sample_tokens future 改为 exec_future） | `core.py` 修改后 P 尾段 `future = exec_future`，已在伪代码体现；单测覆盖 |

### 13.2 滚动升级与兼容性

**本次 head_token 协议变更是前向兼容性破坏**：
- `SchedulerOutput` 新增 `head_token` 字段
- PRE_OUT / POST_OUT ZMQ 消息需携带 `head_token`
- intermediate_tensors 张量字典新增 `_head_token` 字段

**升级要求**：
- 云侧与边侧必须**同时升级**，不允许混跑（旧边 + 新云 或 新边 + 旧云 都会因缺失 head_token 触发 assert）
- 若需灰度，建议先升级云侧（云侧被动消费 head_token，旧边不发送也能跑——但会失去跨通道校验），再升级边侧；或统一通过配置开关 `enable_head_token_pairing=False` 关闭校验做兼容过渡

### 13.3 回滚策略

如果集成验证失败，**第一步只回滚 `core.py` 的 `_needs_sample_tokens` 判定**——退回到"PF 也调 sample_tokens"路径。这条对正确性没影响（只是 sample_tokens 拿 EMPTY），可帮助快速隔离是 worker 派发问题还是 sample_tokens 链路问题。

如果是 worker 派发本身有问题，整体回滚 worker.py / model_runner_v1.py 的改动，回到当前 timeout 状态（已知失败状态），但 Fix A 保留以防止云侧崩溃。

---

## 14 与 Phase 2 / 3 / 4 文档的关系

- Phase 2 / 3 / 4 已经规划并落地了**调度层**的"四段独立"契约
- 本文档 Phase 2-4 Extend 修复**执行层**与该契约的不一致
- 未来 Phase 5 / 6 优化（吞吐/异步/recv_object 同步消除等）以本文档落地后的执行层为前提

---

## 15 `prefill_last_pending` 缓冲队列设计（请求生命周期修复）

### 15.1 问题现象

在 §13 的 `prefill_inflight_limit` 合入后，单条请求的调度顺序仍错误：

```
step1: 调度 PREFILL_FIRST  (PF)   ✓
step2: 调度 PURE_DECODE    (DF)   ✗ — 请求尚未完成完整 prefill，不应进 decode
```

根因在 `_pick_prefill_first_batch`（`pd_separated_scheduler.py:173-183`）：

```python
new_running = [
    req for req in self.running if not req.is_prefill_chunk
]
self.running = saved_running + new_running   # ←★ PF 做完直接进 running
```

父类 `Scheduler.schedule()` 按标准 prefill 语义判断：`is_prefill_chunk == False` = prefill 结束。但在 PD 分离模式下，prefill 的完整生命周期是 **PF → cloud → PL**，PF 做完只算"首段完成"，尾段还没回来。请求提前进入 `running` → decode 调度器误以为它是 decode-ready。

### 15.2 设计方案

新增缓冲队列 `prefill_last_pending`，职责单一：**存放"PF 首段已做完、等 PL 尾段回包"的请求**。该队列不参与任何调度决策。

#### 状态机

```
新请求 ──► waiting
            │
            ▼
        ┌─────────────────────────┐
        │ _pick_prefill_first_batch │
        │ (super().schedule 调度 PF)  │
        └─────────────────────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
chunk_prefill_first   prefill_last_pending
(还需继续 PF)          (PF 做完，等 PL)
            │
            ▼
        ┌─────────────────────────┐
        │ _pick_prefill_last_batch  │
        │ (从 prefills_last_ready pop)│
        └─────────────────────────┘
            │
            ▼
        ┌─────────────────────────┐
        │ update_from_output(PL)    │
        │ 从 prefill_last_pending 移除 │
        │ 加入 running                │
        └─────────────────────────┘
            │
            ▼
        running ──► _pick_decode_batch / DF/DL
```

#### 四个修改点

| # | 位置 | 修改内容 |
|---|------|---------|
| 1 | `PDSeparatedScheduler.__init__` | 新增 `self.prefill_last_pending: list[Request] = []` |
| 2 | `_pick_prefill_first_batch` | PF 做完后，`is_prefill_chunk == False` 的请求不进 `running`，进 `prefill_last_pending` |
| 3 | `_pick_prefill_last_batch` | PL pop 时，从 `chunk_prefill_first` **和** `prefill_last_pending` 同时移除对应 req_ids |
| 4 | `update_from_output` | `batch_type == PREFILL_LAST` 时，把该 batch 的 req_ids 从 `prefill_last_pending` 移到 `running` |

#### 队列语义对照

| 队列 | 语义 | 是否参与调度 |
|------|------|-------------|
| `waiting` | 全新请求，未做任何 prefill | ✓（PF 调度源） |
| `chunk_prefill_first` | chunked prefill 首段，还需继续 PF | ✓（PF 调度源） |
| `prefill_last_pending` | PF 首段已做完，等 PL 回包 | ✗（纯缓冲） |
| `running` | 已完成完整 prefill（PF+PL），可 decode | ✓（DF/DL 调度源） |
| `prefills_last_ready` | cloud POST_OUT 回传的 PL SchedulerOutput | ✓（PL 调度源） |

### 15.3 与 `prefill_inflight_limit` 的协同

两者独立但互补：

| 机制 | 解决的问题 | 触发时机 |
|------|-----------|---------|
| `prefill_inflight_limit` | 防止**多个 PF batch** 同时在飞 | `_schedule_pd_separated` phase 选择时 |
| `prefill_last_pending` | 防止**单个请求**在 PF 未完时误入 decode | `_pick_prefill_first_batch` 结果处理时 |

单条请求的标准调度序列（limit=1）：

```
step1: waiting ──PF──► prefill_last_pending   (count=1)
step2: prefill_last_pending 缓冲中，running 为空
       → _select_scheduling_phase 尝试 DECODE，但 running 为空
       → fallback 到 EMPTY（或等 PL 回包后调度 PL）
stepN: cloud POST_OUT ──PL──► update_from_output
       → 从 prefill_last_pending 移除，加入 running   (count=0)
stepN+1: running ──DF/DL──► running             (正常 decode)
```

### 15.4 边界情形

| 情形 | 处理 |
|------|------|
| PL 回包时 req 已不在 `prefill_last_pending`（如被 abort） | `update_from_output` 用 `req_id in set` 过滤，只移存在的；不在的不影响 running |
| `prefill_last_pending` 中 req 被 preempt | 同 `chunk_prefill_first` 处理：preempt 后回 `waiting`，PL 回包时找不到则忽略 |
| 多请求 batch 部分在 pending、部分不在 | `update_from_output` 只移存在的，不存在的视为已 abort |
| `running` 为空且 `prefill_last_pending` 非空 | `_select_scheduling_phase` 不应选 DECODE（因为 running 为空）→ 自然 fallback 到 EMPTY |
| chunked-prefill：PF 没做完（`is_prefill_chunk == True`）| 进 `chunk_prefill_first`，不进 `prefill_last_pending`，下次继续 PF |

### 15.5 D 段不需要对称队列

Decode 请求本来就在 `running` 里持续存活，DF → DL 只是"同一请求的不同 step 形态"，不需要 pending 缓冲。PL 完成后请求进入 `running`，后续 DF/DL 循环与标准 decode 一致。

### 15.6 文件变更

| 文件 | 改动类型 | 大致行数 | 说明 |
|------|---------|---------|------|
| `vllm-pdmix/vllm/v1/core/sched/pd_separated_scheduler.py` | 新增缓冲队列 + 生命周期调整 | +25 / -3 | `prefill_last_pending` + 四处修改点 |
