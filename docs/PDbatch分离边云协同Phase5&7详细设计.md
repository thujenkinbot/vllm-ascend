# PDbatch 分离边云协同 Phase5 & Phase7 详细设计文档

参考：[PDbatch分离分布式边云协同推理调度算法设计.md](PDbatch分离分布式边云协同推理调度算法设计.md)

---

# 1. 背景

当前 Phase2-4 Extend 已完成单请求基础链路：

```text
P首(edge) -> P中(cloud) -> P尾(edge) -> D首(edge) -> D中(cloud) -> D尾(edge) -> ...
```

并已具备以下基础能力：

- P首 / P尾 / D首 / D尾 独立 batch type；
- 边侧只执行 head/tail segment；
- 云侧只执行 middle segment；
- `head_token` 对齐控制面 `SchedulerOutput` 和数据面 `IntermediateTensors`；
- `prefill_last_pending` 管理 P首完成但 P尾未完成的请求生命周期；
- `prefill_inflight_count` 限制 P batch 在飞数；
- `decode_inflight_count` 限制 D batch 在飞数；
- EMPTY 不再跨边云发送；
- P尾/D尾 不再发往云侧；
- 单请求链路已打通并正常返回。

Phase5 和 Phase7 的目标是在此基础上实现更高吞吐的 **PD 掩盖调度**：

```text
D中 -> P中(掩盖 D尾 + D首 + 通信时延) -> D中
```

进一步支持：

- Phase5：边侧 2P1D 调度状态机；
- Phase7：云侧 P/D 交替调度状态机。

---

# 2. 总体目标

## 2.1 Phase5 目标：边侧支持 2P1D 调度

Phase5 负责边侧调度能力增强，使边侧能够提前下发第二个 Prefill batch，形成：

```text
P1首 -> P2首 -> P1尾 -> P2尾
```

同时 Decode 仍然满足：

```text
D首 -> D尾
```

系统约束：

```text
prefill_inflight_count <= prefill_inflight_limit
decode_inflight_count <= 1
```

其中：

- `prefill_inflight_limit = 1` 时，退化为 1P1D；
- `prefill_inflight_limit = 2` 时，启用 2P1D；
- `decode_inflight_limit` 固定为 1；
- P尾完成后才能把请求从 `prefill_last_pending` 迁移到 `running`；
- D首只能从 `running` 中取请求；
- P首/D首 才能发往云侧；
- P尾/D尾/EMPTY 均不发往云侧。

## 2.2 Phase7 目标：云侧支持 PD 掩盖调度

Phase7 负责云侧调度能力增强，使云侧在 ready queue 中同时存在 P中 和 D中 时，按照状态机进行 P/D 交替调度：

```text
EXPECT_EXECUTE_PREFILL -> EXPECT_EXECUTE_DECODE -> EXPECT_EXECUTE_PREFILL -> ...
```

核心目标：

- 云侧尽量交替执行 P中 / D中；
- 用 P中 掩盖两个 D中 之间的边云通信和边侧计算气泡；
- 当期望类型不存在时允许 fallback；
- 只调度 P首/D首 对应的云侧 middle segment；
- 不接收、不处理 EMPTY / P尾 / D尾。

---

# 3. Phase5：边侧 2P1D 调度详细设计

## 3.1 边侧请求生命周期

边侧请求生命周期如下：

```text
waiting[]
   |
   | P首调度
   v
chunk_prefill_first[] 或 prefill_last_pending[]
   |
   | P尾完成
   v
running[]
   |
   | D首调度
   v
decode in-flight
   |
   | D尾完成
   v
running[] 或 finished
```

各队列语义：

| 队列 | 含义 | 是否可参与调度 |
|---|---|---|
| `waiting[]` | 尚未开始 P首 的新请求 | 可调度 P首 |
| `chunk_prefill_first[]` | 已开始 chunk prefill，但还有后续 P首 chunk 未完成 | 可调度 chunk P首 |
| `prefill_last_pending[]` | P首已完成，P尾未完成 | 不参与调度 |
| `running[]` | P尾已完成，可以调度 D首 | 可调度 D首 |
| `prefills_last_ready[]` | 云侧返回的 P尾控制面 | 可调度 P尾 |
| `decodes_last_ready[]` | 云侧返回的 D尾控制面 | 可调度 D尾 |

关键约束：

```text
prefill_last_pending[] 不允许参与 D首调度。
running[] 只保存已经完成 P尾、具备 decode 条件的请求。
```

## 3.2 边侧 in-flight 计数

Phase5 引入两个核心计数：

```python
prefill_inflight_count
prefill_inflight_limit
decode_inflight_count
decode_inflight_limit = 1
```

语义：

| 字段 | 含义 |
|---|---|
| `prefill_inflight_count` | 当前已经下发 P首，但尚未完成 P尾 的 prefill batch 数量 |
| `prefill_inflight_limit` | 允许同时在飞的 prefill batch 数量，1 表示 1P1D，2 表示 2P1D |
| `decode_inflight_count` | 当前已经下发 D首，但尚未完成 D尾 的 decode batch 数量 |
| `decode_inflight_limit` | 固定为 1 |

计数变化：

| 调度动作 | 计数变化 |
|---|---|
| 调度 P首 | `prefill_inflight_count += 1` |
| 完成 P尾 | `prefill_inflight_count -= 1` |
| 调度 D首 | `decode_inflight_count += 1` |
| 完成 D尾 | `decode_inflight_count -= 1` |

计数不变量：

```text
0 <= prefill_inflight_count <= prefill_inflight_limit
0 <= decode_inflight_count <= 1
```

## 3.3 边侧状态机

根据 `prefill_inflight_count` 定义边侧状态：

| 状态 | 条件 | 含义 |
|---|---|---|
| `IDLE` | `prefill_inflight_count == 0` | 当前无 P batch 在飞 |
| `LOW` | `prefill_inflight_count == 1` | 当前有 1 个 P batch 在飞 |
| `HIGH` | `prefill_inflight_count == prefill_inflight_limit` 且 limit 为 2 | 当前 P batch 在飞数已满 |

当 `prefill_inflight_limit == 1` 时：

```text
IDLE <-> LOW
```

当 `prefill_inflight_limit == 2` 时：

```text
IDLE <-> LOW <-> HIGH
```

## 3.4 边侧调度优先级

### 3.4.1 IDLE 状态

条件：

```text
prefill_inflight_count == 0
```

调度优先级：

| 优先级 | Batch Type | 请求来源 | 状态变化 |
|---|---|---|---|
| 1 | P首 / chunk0首 | `waiting[]` | `IDLE -> LOW` |
| 2 | D尾 | `decodes_last_ready[]` | `IDLE -> IDLE` |
| 3 | D首 | `running[]` | `IDLE -> IDLE` |
| 4 | Empty | - | `IDLE -> IDLE` |

说明：

- IDLE 时优先启动新的 Prefill；
- 若无 Prefill，可调度 Decode；
- D尾优先于 D首，因为 D尾完成可以释放 `decode_inflight_count`；
- Empty 仅本地返回，不发送云侧。

### 3.4.2 LOW 状态

条件：

```text
prefill_inflight_count == 1
```

当 `prefill_inflight_limit == 1` 时，LOW 已经是 P 在飞上限；

当 `prefill_inflight_limit == 2` 时，LOW 仍可继续下发第二个 P首。

调度优先级：

| 优先级 | Batch Type | 请求来源 | 状态变化 |
|---|---|---|---|
| 1 | P尾 / chunk(i)尾 | `prefills_last_ready[]` | `LOW -> IDLE` |
| 2 | chunk(i>0)首 | `chunk_prefill_first[]` | `LOW -> HIGH` |
| 3 | P首 | `waiting[]` | `LOW -> HIGH` |
| 4 | D尾 | `decodes_last_ready[]` | `LOW -> LOW` |
| 5 | D首 | `running[]` | `LOW -> LOW` |
| 6 | Empty | - | `LOW -> LOW` |

说明：

- P尾优先级高，因为 P尾完成后可以释放 P 在飞槽位；
- 2P1D 下，LOW 状态允许提前调度第二个 P首；
- D尾优先于 D首；
- D首必须受 `decode_inflight_count < 1` 限制。

### 3.4.3 HIGH 状态

条件：

```text
prefill_inflight_count == prefill_inflight_limit
```

调度优先级：

| 优先级 | Batch Type | 请求来源 | 状态变化 |
|---|---|---|---|
| 1 | P尾 / chunk(i)尾 | `prefills_last_ready[]` | `HIGH -> LOW` |
| 2 | D尾 | `decodes_last_ready[]` | `HIGH -> HIGH` |
| 3 | D首 | `running[]` | `HIGH -> HIGH` |
| 4 | Empty | - | `HIGH -> HIGH` |

说明：

- HIGH 状态禁止继续调度 P首；
- 必须等待 P尾完成释放 prefill slot；
- Decode 仍可在 `decode_inflight_count < 1` 时调度；
- D尾仍然优先于 D首。

## 3.5 边侧调度伪代码

```python
def schedule_edge():
    drain_post_out()

    if prefills_last_ready:
        return pick_prefill_last()

    if decodes_last_ready:
        return pick_decode_last()

    state = get_prefill_state()

    if state == IDLE:
        if has_prefill_work():
            return pick_prefill_first()
        if can_schedule_decode_first():
            return pick_decode_first()
        return empty()

    if state == LOW:
        if prefill_inflight_count < prefill_inflight_limit:
            if has_prefill_work():
                return pick_prefill_first()
        if can_schedule_decode_first():
            return pick_decode_first()
        return empty()

    if state == HIGH:
        if can_schedule_decode_first():
            return pick_decode_first()
        return empty()
```

其中：

```python
def has_prefill_work():
    return bool(waiting or chunk_prefill_first)

def can_schedule_decode_first():
    return bool(running) and decode_inflight_count < decode_inflight_limit
```

## 3.6 P首调度规则

P首调度来源：

```text
waiting[]
chunk_prefill_first[]
```

P首调度动作：

1. 临时将 `running` 替换为 `chunk_prefill_first`；
2. 临时清空 `waiting` 外的 decode 请求来源；
3. 调用父类 scheduler 生成 prefill SchedulerOutput；
4. 非空时标记：

```python
scheduler_output.batch_type = BatchType.PREFILL_FIRST
scheduler_output.head_token = uuid4().hex
prefill_inflight_count += 1
```

5. P首完成后：
   - 未完成 prefill 的请求进入 `chunk_prefill_first[]`；
   - 已完成 P首但未 P尾的请求进入 `prefill_last_pending[]`；
   - 不进入 `running[]`。

## 3.7 P尾调度规则

P尾来源：

```text
prefills_last_ready[]
```

P尾调度动作：

1. 从 `prefills_last_ready[]` pop 一个 `PREFILL_LAST`；
2. 校验 batch_type；
3. 不从 `prefill_last_pending[]` 删除请求；
4. 只允许从 `chunk_prefill_first[]` 移除对应请求；
5. 等 `update_from_output(PREFILL_LAST)` 后再迁移请求。

P尾完成后：

```python
prefill_inflight_count -= 1
prefill_last_pending[] -> running[]
```

## 3.8 D首调度规则

D首来源：

```text
running[]
```

D首不允许从以下队列取请求：

```text
waiting[]
chunk_prefill_first[]
prefill_last_pending[]
```

D首调度动作：

1. 确认：

```python
running
decode_inflight_count < 1
```

2. 临时清空：

```python
waiting
skipped_waiting
chunk_prefill_first
```

3. 调用父类 scheduler，仅从 `running[]` 构造 decode SchedulerOutput；
4. 非空时标记：

```python
scheduler_output.batch_type = BatchType.DECODE_FIRST
scheduler_output.head_token = uuid4().hex
decode_inflight_count += 1
```

5. 补齐 `scheduled_cached_reqs.all_token_ids`。

补齐原因：

PD 分离下 P尾不走父类 `super().schedule()`，导致 `prev_step_scheduled_req_ids` 可能残留 P首 req_id。第一个 D首构造 cached req 时，父类可能误判该请求在上一轮已经调度，从而不填充 `all_token_ids`。但边侧 model_runner 在 async scheduling 下，如果 `req_index is None` 且 `num_output_tokens > 0`，需要从 `all_token_ids` 恢复 output tokens。

因此 D首需要防御性补齐：

```python
if num_output_tokens > 0 and req_id not in cached_reqs.all_token_ids:
    cached_reqs.all_token_ids[req_id] = self.requests[req_id].all_token_ids.copy()
```

## 3.9 D尾调度规则

D尾来源：

```text
decodes_last_ready[]
```

D尾调度动作：

1. 从 `decodes_last_ready[]` pop 一个 `DECODE_LAST`；
2. 校验 batch_type；
3. 执行边侧 tail segment 和 sampler；
4. `update_from_output(DECODE_LAST)` 后：

```python
decode_inflight_count -= 1
```

D尾完成后：

- 请求若未结束，仍在 `running[]`，可参与下一轮 D首；
- 请求若结束，从 `running[]` 清理。

## 3.10 Phase5 正确性不变量

Phase5 必须满足以下不变量：

```text
1. waiting[] 中请求只能调度 P首。
2. prefill_last_pending[] 中请求不能调度 D首。
3. running[] 中请求必须已经完成 P尾。
4. D首只能从 running[] 构造。
5. P尾/D尾/EMPTY 不允许发送云侧。
6. PREFILL_FIRST/DECODE_FIRST 必须携带 head_token。
7. PREFILL_LAST/DECODE_LAST 必须通过 head_token 匹配 HeadState。
8. prefill_inflight_count 只在 P首/P尾变化。
9. decode_inflight_count 只在 D首/D尾变化。
10. has_requests() 必须计入 prefill_last_pending[] 和 chunk_prefill_first[]。
```

---

# 4. Phase7：云侧 PD 掩盖调度详细设计

## 4.1 云侧职责

云侧只执行 middle segment：

```text
P中
D中
```

云侧不执行：

```text
P首
P尾
D首
D尾
EMPTY
```

云侧输入来源：

```text
PRE_OUT channel
```

只允许接收：

```text
PREFILL_FIRST
DECODE_FIRST
```

云侧输出：

```text
POST_OUT channel
```

只允许返回：

```text
PREFILL_FIRST -> PREFILL_LAST
DECODE_FIRST  -> DECODE_LAST
```

## 4.2 云侧 ready queue

云侧 PassiveScheduler 维护以下 ready queue：

| 队列 | 来源 | 含义 |
|---|---|---|
| `ready_prefills[]` | `PREFILL_FIRST` | 可执行 P中 |
| `ready_decodes[]` | `DECODE_FIRST` | 可执行 D中 |
| `ready_pdmixes[]` | legacy `PD_MIX` | 兼容旧路径 |
| `ready_empties[]` | 不再使用 | EMPTY 不接收不处理 |

云侧接收规则：

| batch type | 云侧行为 |
|---|---|
| `PREFILL_FIRST` | 放入 `ready_prefills[]` |
| `DECODE_FIRST` | 放入 `ready_decodes[]` |
| `PREFILL_LAST` | 丢弃并记录错误 |
| `DECODE_LAST` | 丢弃并记录错误 |
| `EMPTY` | 直接丢弃 |
| `PURE_PREFILL` | legacy 路径，放入 `ready_prefills[]` |
| `PURE_DECODE` | legacy 路径，放入 `ready_decodes[]` |
| `PD_MIX` | legacy 路径，放入 `ready_pdmixes[]` |

## 4.3 云侧状态机

Phase7 引入云侧调度状态：

```python
CloudSchedulingState.EXPECT_EXECUTE_PREFILL
CloudSchedulingState.EXPECT_EXECUTE_DECODE
```

状态含义：

| 状态 | 含义 |
|---|---|
| `EXPECT_EXECUTE_PREFILL` | 本轮优先调度 P中 |
| `EXPECT_EXECUTE_DECODE` | 本轮优先调度 D中 |

初始状态建议：

```python
EXPECT_EXECUTE_PREFILL
```

原因：

- Prefill 数量通常少于 Decode；
- 优先调度 P中 有利于提前填充 P pipeline；
- 2P1D 下边侧会提前准备 P中，云侧优先消费 P中 更容易形成 PD 掩盖。

## 4.4 云侧状态转移规则

状态转移只在成功调度期望 batch 时发生。

### 4.4.1 EXPECT_EXECUTE_PREFILL

优先级：

| 优先级 | Batch | 来源 | 状态变化 |
|---|---|---|---|
| 1 | P中 | `ready_prefills[]` | `EEP -> EED` |
| 2 | D中 | `ready_decodes[]` | `EEP -> EEP` |
| 3 | PD_MIX | `ready_pdmixes[]` | `EEP -> EEP` |
| 4 | Empty | - | `EEP -> EEP` |

说明：

- 期望执行 P中；
- 如果 P中 ready，则调度 P中，并切换到期望 D中；
- 如果 P中 不 ready，但 D中 ready，则 fallback 调度 D中，但状态保持 EEP；
- 状态保持的意义是：下一轮仍然优先尝试 P中，避免 P 被 D 持续饿死。

### 4.4.2 EXPECT_EXECUTE_DECODE

优先级：

| 优先级 | Batch | 来源 | 状态变化 |
|---|---|---|---|
| 1 | D中 | `ready_decodes[]` | `EED -> EEP` |
| 2 | P中 | `ready_prefills[]` | `EED -> EED` |
| 3 | PD_MIX | `ready_pdmixes[]` | `EED -> EED` |
| 4 | Empty | - | `EED -> EED` |

说明：

- 期望执行 D中；
- 如果 D中 ready，则调度 D中，并切换到期望 P中；
- 如果 D中 不 ready，但 P中 ready，则 fallback 调度 P中，但状态保持 EED；
- 状态保持的意义是：下一轮仍然优先尝试 D中，避免 Decode 被 P 持续占用。

## 4.5 云侧调度伪代码

```python
def schedule_cloud():
    poll_and_classify()

    if state == EXPECT_EXECUTE_PREFILL:
        if ready_prefills:
            batch = pop_prefill()
            state = EXPECT_EXECUTE_DECODE
            return batch

        if ready_decodes:
            batch = pop_decode()
            state = EXPECT_EXECUTE_PREFILL
            return batch

        if ready_pdmixes:
            return pop_pdmix()

        return empty()

    if state == EXPECT_EXECUTE_DECODE:
        if ready_decodes:
            batch = pop_decode()
            state = EXPECT_EXECUTE_PREFILL
            return batch

        if ready_prefills:
            batch = pop_prefill()
            state = EXPECT_EXECUTE_DECODE
            return batch

        if ready_pdmixes:
            return pop_pdmix()

        return empty()
```

注意：

上面的状态转移有两种可选策略：

## 策略 A：只在命中期望类型时切换

```text
EEP fallback D中 后仍保持 EEP
EED fallback P中 后仍保持 EED
```

优点：

- 更符合“期望调度”的语义；
- 可避免某一类 batch 因短暂不 ready 被长期饿死；
- 与参考文档中“若本轮成功调度期望类型，则状态转移，否则状态不变”一致。

## 策略 B：只要调度了 P/D 就切换

```text
调度 P中 后切 EED
调度 D中 后切 EEP
```

优点：

- 实现简单；
- 更接近严格 P/D 交替。

推荐采用 **策略 A**。

## 4.6 云侧切片调度

云侧 P中 可以按层切分 slice 执行，用于增加 P中 数量，使 P中 更好覆盖 D中 之间的气泡。

配置：

```text
VLLM_LAYER_SLICE_SIZE
```

切片规则：

| batch | 是否切片 |
|---|---|
| `PREFILL_FIRST` / P中 | 可切片 |
| `PURE_PREFILL` | 可切片 |
| `PD_MIX` | 可切片 |
| `DECODE_FIRST` / D中 | 不切片 |
| `PURE_DECODE` | 不切片 |
| `EMPTY` | 不处理 |

P中切片示例：

```text
P中(slice0) -> P中(slice1) -> P中(slice2)
```

Decode 不切片：

```text
D中
```

目标：

```text
让多个 P中 slice 填充两个 D中 之间的通信和边侧计算气泡。
```

## 4.7 云侧 POST_OUT 生成规则

云侧调度并执行完成后，需要向边侧返回 tail SchedulerOutput。

映射关系：

| 云侧收到 | 云侧执行 | 云侧返回 |
|---|---|---|
| `PREFILL_FIRST` | P中 | `PREFILL_LAST` |
| `DECODE_FIRST` | D中 | `DECODE_LAST` |
| `EMPTY` | 不执行 | 不返回 |
| `PREFILL_LAST` | 不执行 | 不返回 |
| `DECODE_LAST` | 不执行 | 不返回 |

返回时必须保留：

```text
head_token
num_scheduled_tokens
scheduled_new_reqs
scheduled_cached_reqs
sampling metadata
KV metadata
```

并且数据面 `IntermediateTensors` 中必须携带：

```text
_head_token
```

用于边侧 tail segment 恢复 HeadState。

## 4.8 云侧正确性不变量

Phase7 必须满足：

```text
1. 云侧只执行 P中 / D中。
2. 云侧不执行 P尾 / D尾 / EMPTY。
3. PREFILL_FIRST 必须返回 PREFILL_LAST。
4. DECODE_FIRST 必须返回 DECODE_LAST。
5. POST_OUT 返回的 SchedulerOutput 必须保留 head_token。
6. 数据面 IntermediateTensors 必须携带 _head_token。
7. PassiveScheduler 不允许把 PREFILL_LAST / DECODE_LAST 放入 ready queue。
8. EMPTY 不进入 ready queue，不 enqueue worker。
9. 云侧状态机只根据 P中/D中 调度结果更新。
10. P中切片不能破坏 head_token 和 SchedulerOutput 对齐关系。
```

---

# 5. Phase5 + Phase7 组合流水

## 5.1 1P1D 流水

当：

```text
prefill_inflight_limit = 1
```

典型流水：

```text
Edge: P1首
Cloud: P1中
Edge: P1尾
Edge: D1首
Cloud: D1中
Edge: D1尾
Edge: D2首
Cloud: D2中
Edge: D2尾
```

云侧可能出现：

```text
D中 -> idle -> D中
```

因为两个 P batch 之间无法提前准备下一个 P中。

## 5.2 2P1D 流水

当：

```text
prefill_inflight_limit = 2
```

典型流水：

```text
Edge: P1首
Edge: P2首
Cloud: P1中
Cloud: P2中
Edge: P1尾
Edge: P2尾
Edge: D1首
Cloud: D1中
Edge: D1尾
```

在更稳定的多请求场景中，目标流水是：

```text
Cloud: D中 -> P中 -> D中 -> P中 -> D中 -> P中
Edge:        D尾+D首       D尾+D首
```

即：

```text
D中 -> P中(掩盖通信 + D尾 + D首) -> D中
```

---

# 6. 配置设计

## 6.1 边侧配置

建议配置：

```python
pd_prefill_inflight_limit: int = 1
```

含义：

| 值 | 调度模式 |
|---|---|
| 1 | 1P1D |
| 2 | 2P1D |

后续可扩展：

```python
pd_decode_inflight_limit: int = 1
```

当前保持固定：

```python
decode_inflight_limit = 1
```

## 6.2 云侧配置

建议配置：

```python
pd_cloud_dispatch_policy: str = "expect_alternation"
```

可选：

| 值 | 含义 |
|---|---|
| `expect_alternation` | EEP/EED 状态机 |
| `prefill_first` | 总是优先 P中 |
| `decode_first` | 总是优先 D中 |
| `pdmix_first` | legacy 兼容 |

Phase7 推荐默认：

```text
expect_alternation
```

---

# 7. 日志与调试信息

## 7.1 边侧日志

建议每次调度打印：

```text
[PD] StepN
phase
waiting[]
chunk_prefill_first[]
prefill_last_pending[]
running[]
prefills_last_ready[]
decodes_last_ready[]
prefill_inflight_count / prefill_inflight_limit
decode_inflight_count / decode_inflight_limit
```

关键事件日志：

```text
PREFILL_FIRST scheduled, prefill_inflight +1
PREFILL_LAST done, prefill_inflight -1, pending -> running
DECODE_FIRST scheduled, decode_inflight +1
DECODE_LAST done, decode_inflight -1
```

## 7.2 云侧日志

建议每次调度打印：

```text
[PD-CLOUD] state=EXPECT_EXECUTE_PREFILL
ready_prefills=[]
ready_decodes=[]
ready_pdmixes=[]
picked=PREFILL_FIRST
next_state=EXPECT_EXECUTE_DECODE
```

异常日志：

```text
Received PREFILL_LAST on cloud, dropped
Received DECODE_LAST on cloud, dropped
Received EMPTY on cloud, dropped
Missing head_token for PREFILL_FIRST/DECODE_FIRST
```

---

# 8. 测试设计

## 8.1 Phase5 边侧测试

### 测试 1：1P1D

配置：

```text
prefill_inflight_limit = 1
```

期望：

```text
P1首 -> 等 P1尾 -> D首
```

不能出现：

```text
P1首 -> P2首
```

### 测试 2：2P1D

配置：

```text
prefill_inflight_limit = 2
```

期望：

```text
P1首 -> P2首 -> P1尾 -> P2尾
```

并保证：

```text
prefill_inflight_count <= 2
```

### 测试 3：D首只能从 running 取

构造：

```text
waiting 有请求
prefill_last_pending 有请求
running 为空
```

期望：

```text
不能调度 D首
```

### 测试 4：P尾后迁移 running

流程：

```text
P首完成 -> prefill_last_pending
P尾完成 -> running
```

期望：

```text
running 中出现该请求
```

### 测试 5：decode_inflight 限制

流程：

```text
D首 -> decode_inflight_count = 1
再次尝试 D首
```

期望：

```text
不能继续调度 D首
必须等 D尾完成
```

## 8.2 Phase7 云侧测试

### 测试 1：EEP 优先 P中

输入：

```text
ready_prefills = [P1]
ready_decodes = [D1]
state = EEP
```

期望：

```text
调度 P1
state -> EED
```

### 测试 2：EED 优先 D中

输入：

```text
ready_prefills = [P1]
ready_decodes = [D1]
state = EED
```

期望：

```text
调度 D1
state -> EEP
```

### 测试 3：EEP fallback D中

输入：

```text
ready_prefills = []
ready_decodes = [D1]
state = EEP
```

期望：

```text
调度 D1
state 保持 EEP
```

### 测试 4：EED fallback P中

输入：

```text
ready_prefills = [P1]
ready_decodes = []
state = EED
```

期望：

```text
调度 P1
state 保持 EED
```

### 测试 5：云侧丢弃非法 batch

输入：

```text
EMPTY
PREFILL_LAST
DECODE_LAST
```

期望：

```text
不进入 ready queue
不 enqueue worker
不返回 POST_OUT
```

---

# 9. 风险与注意事项

## 9.1 `prev_step_scheduled_req_ids` 风险

PD 分离下 P尾/D尾 不走父类 scheduler，会导致父类 `prev_step_scheduled_req_ids` 与真实边侧执行 step 不完全一致。

风险：

```text
DECODE_FIRST scheduled_cached_reqs 缺 all_token_ids
```

解决：

```text
D首构造后补齐 cached_reqs.all_token_ids
```

## 9.2 HeadState 匹配风险

2P1D 下可能同时存在多个 P首 in-flight：

```text
P1首
P2首
P1尾
P2尾
```

不能用 req_id 顺序做唯一匹配，必须使用：

```text
head_token
```

匹配：

```text
SchedulerOutput.head_token == IntermediateTensors["_head_token"]
```

## 9.3 Tail segment 重复 update 风险

P尾/D尾 与 P首/D首 拆成两次 execute_model 后，不能无脑重复 `_update_states()`。

策略：

```text
如果 input_batch.req_ids 已经等于 tail scheduler_output req_ids，则跳过 _update_states。
否则允许 _update_states 重建 input_batch。
```

## 9.4 EMPTY / Tail 误发云侧风险

边侧只允许发送：

```text
PREFILL_FIRST
DECODE_FIRST
```

禁止发送：

```text
EMPTY
PREFILL_LAST
DECODE_LAST
```

云侧即使收到，也必须丢弃。

---

# 10. 实施步骤建议

## Phase5 实施步骤

1. 完善 `PDSeparatedScheduler` 状态机；
2. 支持 `prefill_inflight_limit = 2`；
3. 确保 `prefill_inflight_count` 只在 P首/P尾变化；
4. 确保 `decode_inflight_count` 只在 D首/D尾变化；
5. 封装 `_pick_decode_first_batch()`；
6. D首只从 `running[]` 构造；
7. P尾完成后从 `prefill_last_pending[]` 迁移到 `running[]`；
8. 补齐 D首 `cached_reqs.all_token_ids`；
9. 增加 1P1D / 2P1D 调度单测。

## Phase7 实施步骤

1. 云侧 `PassiveScheduler` 增加 EEP/EED 状态；
2. 实现 `expect_alternation` 调度策略；
3. `PREFILL_FIRST` route 到 `ready_prefills[]`；
4. `DECODE_FIRST` route 到 `ready_decodes[]`；
5. 云侧只调度 P中/D中；
6. 云侧只返回 PL/DL；
7. EMPTY / PL / DL 在云侧全部丢弃；
8. 增加云侧状态机单测；
9. 验证 PD 掩盖流水。

---

# 11. 验收标准

Phase5 验收：

```text
1. 单请求 P首/P尾/D首/D尾 正常返回。
2. prefill_inflight_limit=1 时不会连续调度两个 P首。
3. prefill_inflight_limit=2 时允许 P1首 -> P2首。
4. D首不会从 prefill_last_pending 调度。
5. D首不会超过 decode_inflight_limit=1。
6. P尾完成后请求进入 running。
7. P尾/D尾/EMPTY 不发送云侧。
```

Phase7 验收：

```text
1. 云侧只收到 P首/D首。
2. 云侧不处理 EMPTY/P尾/D尾。
3. 云侧状态机按 EEP/EED 交替调度。
4. P中/D中 ready 同时存在时能交替执行。
5. P中 缺失时 D中 可 fallback。
6. D中 缺失时 P中 可 fallback。
7. POST_OUT 正确返回 P尾/D尾。
8. 2P1D 下云侧能形成 D中 -> P中 -> D中 的掩盖流水。
```
