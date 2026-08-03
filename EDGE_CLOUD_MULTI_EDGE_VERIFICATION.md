# Edge-Cloud 多边协同（2 edge + 1 cloud）分步验证指南

本特性让 **2 个 edge（各 1 卡 910B）共享 1 个 cloud（多卡 TP）**，一份云权重
时分轮转服务两个 edge。本文档按"验证点"组织，**每个验证点对应一组 commit，
通过即表示该子系统已 OK**。在 NPU 多节点环境逐点验证，失败时按点定位问题块，
不要跨点混调。

## 目标拓扑

```
edge_0 (1卡, hostA)   edge_1 (1卡, hostB)        cloud (4卡TP, hostC)
 head+tail 段           head+tail 段              middle 段（一份权重）
     │ hidden_0             │ hidden_1
     ▼                      ▼
   PP pair 0 ◀──────── cloud 首卡 ────────▶ PP pair 1
                   （归属 2 个 PP pair）
```

- 每 edge 单 DP 单卡；cloud 多卡 TP（一份权重）。
- 不做跨 edge batch 合并，cloud 串行时分轮转。
- rank 布局：rank 0/1 = edge_0/edge_1；rank 2.. = cloud（首卡 = rank num_edges）。

## 环境准备

- 3 节点互通，模型路径一致，CANN/HCCL 配好。
- 两仓库都在 `feat/edge-cloud-multi-edge` 分支，**checkout 到要验证的验证点对应的 commit**。
- 角色：edge 进程不带 `--headless`（is_edge_node=True）；cloud 进程带 `--headless`。
- env：
  - edge_0：`VLLM_ASCEND_EDGE_CLOUD_EDGE_IDX=0`
  - edge_1：`VLLM_ASCEND_EDGE_CLOUD_EDGE_IDX=1`
  - cloud：不设 EDGE_IDX；`VLLM_ASCEND_EDGE_CLOUD_MASTER_ADDRS=hostA,hostB`（D 阶段起用）

## 通用启动模板

```bash
# edge_i (i=0,1)
VLLM_ASCEND_EDGE_CLOUD_EDGE_IDX=<i> vllm serve <model> \
  --enable-edge-cloud --num-edges 2 \
  --edge-npu-count 2 --cloud-npu-count 4 \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 --data-parallel-size 1 \
  --master-addr <edge_i_host> --master-port 29501 --port 800<i+1>

# cloud
VLLM_ASCEND_EDGE_CLOUD_MASTER_ADDRS=<hostA>,<hostB> vllm serve <model> \
  --enable-edge-cloud --headless --num-edges 2 \
  --edge-npu-count 2 --cloud-npu-count 4 \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 --data-parallel-size 1 \
  --master-addr <hostA> --master-port 29501 --port 8000
```

---

## 验证点 A — 配置层

**commit**：vllm `5b05cc5` + vllm-ascend `54b1641`
**子系统**：`num_edges` 配置端到端（CLI → 字段 → rank 布局）

**OK 标志**：
1. `num_edges=1`（默认，不加 `--num-edges`）：现有 1:1 edge-cloud e2e **行为完全不变**（这是回归红线）。
2. `num_edges=2`：vllm serve 启动到配置加载阶段，日志可见 `num_edges=2`、`world_size=6`（2+4）、edge `tp=1` / cloud `tp=4`。

**怎么验证**：
- 先跑 A.1：任意现有 edge-cloud 用例，确认零变化。
- 再加 `--num-edges 2` 启动，看 startup 日志的 parallel config 打印。

**预期可失败（不算 bug）**：此阶段 distributed init / PP 建组尚未完整（B 才补），所以 `num_edges=2` 可能在 init 阶段报错或卡住——只要配置打印正确即算 A 通过。

---

## 验证点 B — cloud 多 PP group 建组

**commit**：vllm `af08b89` + vllm-ascend `0d48dbc`
**子系统**：N 个 per-edge PP GroupCoordinator 建立 + hidden channel warmup

**OK 标志**（cloud 日志，3 节点同时启动、cloud 先 ready）：
1. `[edge-cloud] multi-edge: built 2 per-edge PP groups`
2. `[edge-cloud] warmed up hidden channels [prefill_1, ...] (both directions)` 出现 **2 次**（每对 PP 各一次）
3. 无 `new_group` / HCCL 报错、不卡死（init 在 1~2 分钟内完成）

**怎么验证**：3 节点启动，盯 cloud 的 init 日志到 `worker_monitor` ready。

**失败排查**：
| 现象 | 查 |
|---|---|
| `new_group` 报错/卡死 | `_init_multi_edge_pp_groups`：所有 rank 是否按相同顺序对相同 group_ranks 调 `create_alternate_groups/create_hidden_channel_groups`（collective 必须全 rank 同步） |
| warmup 卡死 | cloud 首卡是否**串行**预热 2 edge（`warmup_edge_cloud_hidden_channels(pp_group=gc)` 逐个）；两 edge 是否各自只在自己的 pair 上 warmup |
| edge 侧报 PP group 未初始化 | edge 是否能经 `get_pp_group_for_edge(edge_id)` 拿到 world_size=2 的 pair |

---

## 验证点 C — 按 edge_id 通信

**commit**：vllm-ascend `bcf9e30`
**子系统**：edge/cloud 的 send/recv 按 `SO.edge_id` 选对的 PP pair

**OK 标志**：
- 单 edge（先只起 edge_0）发一个 prefill 请求，cloud 日志显示收到 `edge_id=0` 的 hidden，且 hidden 经正确的 PP pair（pair 0）收发，无串扰。
- edge_0 收到 cloud 回推的 middle，完成 tail，产出 token。

**怎么验证**：只起 edge_0 + cloud（edge_1 不起，避开 D 的多 edge 接入），对 edge_0 发一个 completion 请求，看是否正常返回。
> 说明：此阶段 cloud 控制流仍按单 edge（D 才改多 edge 调度），所以用**单 edge** 验证通信路由正确性。

**失败排查**：
| 现象 | 查 |
|---|---|
| cloud 收不到 hidden / edge_id 缺失 | `_ensure_pd_head_token` 里的 `_stamp_edge_id`（env `VLLM_ASCEND_EDGE_CLOUD_EDGE_IDX` 是否设） |
| 选错 PP pair | `NPUWorker._ec_pp_group`（edge 用 `_my_edge_id`，cloud 用 `SO.edge_id`） |
| hidden 串到另一 pair | `edge_cloud_*` 函数是否都传了 `pp_group`（漏传会回落 `get_pp_group()` = singleton，短路） |

---

## 验证点 D — cloud 服务多 edge

**commit**：vllm-ascend（D：握手 ZMQ fan-in + 调度 round-robin）
**子系统**：cloud 同时接入 2 edge 控制流 + round-robin 轮转服务

**OK 标志**：
- cloud 日志 `PD-separation cloud channels: 2 edge(s)`（建了 2 个 PPSchedulerZmqChannel）
- 2 edge 各发请求，cloud 日志显示 `edge_id=0/1` 交替处理（round-robin），结果各回各 edge

**怎么验证**：3 节点（edge_0 + edge_1 + cloud）启动，两 edge 各发一个 completion 请求，看是否各正确返回、cloud 日志交替出现两个 edge_id。

**失败排查**：
| 现象 | 查 |
|---|---|
| cloud 只连 1 edge | `VLLM_ASCEND_EDGE_CLOUD_MASTER_ADDRS`（逗号分隔 2 个地址）+ `--num-edges 2` |
| edge 报 ZMQ bind 冲突（EADDRINUSE）| `VLLM_ASCEND_EDGE_CLOUD_EDGE_IDX`（edge_0=0、edge_1=1，端口偏移 edge_idx*2）|
| 请求/响应串到错的 edge | `_maybe_publish_post_out` 是否按 `SO.edge_id` 选 channel；`_stamp_edge_id` 是否在每个 SO 上 |
| 跨 edge prefill 状态串（layerwise）| `_active_sliced_prefill` 跨 edge 互斥（MVP 时分轮转天然不并发，极端交错可能需 NPU 调优）|

---

## 验证点 E — KV 隔离

**commit**：vllm-ascend（E：cloud 侧 block_id 偏移，方案 B）
**子系统**：两 edge 的 KV block ID 不撞 cloud 单 pool（数据正确性，**最关键**）

**实现**（cloud 侧偏移）：cloud 在 `step()` 的 **worker 副本**上把 block IDs `+= edge_id * (cloud_num_blocks // num_edges)`；echo 回 edge 的**原件**保持 local block IDs，所以 edge tail 段索引自己的小 KV pool 无需还原。一处 hook、零还原、零跨节点 num_blocks 协调。覆盖三个字段：`scheduled_new_reqs[].block_ids`、`scheduled_cached_reqs.new_block_ids`、`new_block_ids_to_zero`（-1 空块保留）。`cloud_num_blocks` 从 env `VLLM_ASCEND_EDGE_CLOUD_CLOUD_NUM_BLOCKS` 读。

**OK 标志**：2 edge **同时**跑相同 prompt，输出与单 edge baseline **逐 token 一致**（无静默损坏）。

**怎么验证**：
1. cloud 启动看日志 profile 出的 `num_blocks`，设 `VLLM_ASCEND_EDGE_CLOUD_CLOUD_NUM_BLOCKS=<该值>` 重启 cloud。
2. 2 edge + cloud，两 edge **同时**发相同 prompt，逐 token 对比输出。

**失败排查**：
| 现象 | 查 |
|---|---|
| cloud 报 block 越界（assert）| `VLLM_ASCEND_EDGE_CLOUD_CLOUD_NUM_BLOCKS` 是否 = cloud 实际 profile 的 num_blocks |
| 输出损坏/乱码 | 偏移字段是否漏（三个都要覆盖）；`SO.edge_id` 是否盖戳 |
| 单 edge 也偏移（应短路）| `num_edges<=1` 或 `cloud_num_blocks==0` 时 `_offset_worker_so_block_ids` 应直接 return |

**NPU TODO**：`cloud_num_blocks` 目前用 env（手设 profile 值）。可改为惰性 worker RPC（`get_kv_cache_config`）自动同步，去掉手设。

---

## 端到端（D + E 通过后）

1. 2 edge + 1 cloud，并发请求，输出正确。
2. `num_edges=1` 全量回归通过。
3.（P3）杀 edge_1，edge_0 继续服务。

## 故障定位速查

| 现象 | 先查验证点 |
|---|---|
| `num_edges` 不识别 / 配置错 | A |
| distributed init 卡死 / `new_group` 报错 | B |
| warmup 卡死 | B |
| 单 edge prefill 收不到 hidden | C |
| cloud 只服务一个 edge / 请求串 | D |
| 输出乱码、损坏、token 错 | E |
