# PD batch分离边云协同推理 — Phase 1 详细设计文档

> 本文档基于《PDbatch分离分布式边云协同推理设计说明书》中 4.3 节 Phase 1 的两个功能点进行细化设计。参考实现来源于 `vllm-pdmix` 仓（GPU 通用逻辑）与 `vllm-ascend-pdmix` 仓（昇腾 NPU 适配逻辑）。

---

## 0 术语与上下文

| 术语 | 含义 |
|------|------|
| 边侧 (Edge) | 推理入口节点，执行 Embedding + 首若干 Transformer 层 + 尾若干 Transformer 层 + LM Head |
| 云测 (Cloud) | 算力扩展节点，执行中间 Transformer 层 |
| P 首 / P 尾 | Prefill 的首段（边侧）和尾段（边侧） |
| D 首 / D 尾 | Decode 的首段（边侧）和尾段（边侧） |
| P 中 / D 中 | Prefill/Decode 的中间段（云测） |
| TP 不均等 | 边侧和云测的 NPU/GPU 数量不一致（如边 2 NPU、云 4 NPU） |
| `PPMissingLayer` | vLLM 占位层，不加载权重、forward 直接透传输入 |
| `LayerShardLoader` | 边云协同模型层切分器，把非本地层替换为 `PPMissingLayer` |

**当前 PD batch 分离版本现状（vllm_dev）：**
- PP=2，rank0（边侧）为 leader，rank1（云测）为 passive。
- 模型加载沿用原生 PP 逻辑：rank0 加载前半层，rank1 加载后半层。
- 没有 `enable_edge_cloud` 开关，没有 `edge_head_tail_layers` 配置。
- 两节点 TP 数量相等。

**目标：** 把 `vllm-pdmix` / `vllm-ascend-pdmix` 中的 Phase 1 能力迁移到 PD batch 分离版本，使其具备：
1. 边侧加载首尾层、云测加载中间层（**复用 PDmix 已有实现**）。
2. 边侧与云测 TP 不均等划分（**复用 PDmix 已有实现**）。
3. 新增 `--enable-pd-separation` 作为 `--enable-edge-cloud` 的子特性，开启 PD batch 分离调度。

**核心设计原则：**
- `--enable-edge-cloud` = 边云协同基础能力（模型层切分 + 不均等 TP + 通信组布局）。
- `--enable-pd-separation` = PD batch 分离调度特性，**必须依赖 `--enable-edge-cloud`**。
- 开启 `--enable-pd-separation` 时，**无需也不应配置 `--pipeline-parallel-size`**，内部隐式固定为 2。

---

## 0.5 配置参数关系与校验规则

### 新增参数

| 参数 | 类型 | 所属配置 | 含义 |
|------|------|----------|------|
| `--enable-edge-cloud` | bool | `ParallelConfig` | 启用边云协同基础能力 |
| `--enable-pd-separation` | bool | `ParallelConfig` | 启用 PD batch 分离调度（`--enable-edge-cloud` 的子特性） |
| `--edge-npu-count` | int | `ParallelConfig` | 边侧 NPU 数量 |
| `--cloud-npu-count` | int | `ParallelConfig` | 云测 NPU 数量 |
| `edge_cloud_config` | dict | `additional_config` | 昇腾专属：层切分策略、角色、模式等 |

### 参数校验逻辑

```python
def _validate_edge_cloud(self):
    # --enable-pd-separation 必须同时开启 --enable-edge-cloud
    if self.enable_pd_separation and not self.enable_edge_cloud:
        raise ValueError("--enable-pd-separation requires --enable-edge-cloud")

    if self.enable_edge_cloud:
        if self.edge_npu_count <= 0 or self.cloud_npu_count <= 0:
            raise ValueError("edge_npu_count and cloud_npu_count must be positive")
        if self.edge_npu_count >= self.cloud_npu_count:
            raise ValueError("edge_npu_count must be less than cloud_npu_count")
        if self.data_parallel_size != 1:
            raise ValueError("data_parallel_size must be 1 in edge-cloud mode")

        # 无论 enable_pd_separation 是否开启，pipeline_parallel_size 都由内部管理
        if self.pipeline_parallel_size != 1:
            # 如果用户显式配置了 --pipeline-parallel-size，给出友好提示
            logger.warning(
                "--pipeline-parallel-size is ignored in edge-cloud mode. "
                "Internally fixed to 2."
            )
        self.pipeline_parallel_size = 2

        self.world_size = self.edge_npu_count + self.cloud_npu_count
        self.tensor_parallel_size = (
            self.edge_npu_count if self.is_edge_node else self.cloud_npu_count
        )
```

### 三种模式的互斥与切换

| 模式 | 条件 | `pipeline_parallel_size` | 模型加载 | 调度执行 |
|------|------|--------------------------|----------|----------|
| 原生 PP | `enable_edge_cloud=False` | 用户配置 | `get_pp_indices` 切层 | 标准 EngineCore |
| 边云协同（PDmix） | `enable_edge_cloud=True, enable_pd_separation=False` | 内部固定 2 | `LayerShardLoader` 切层 | 单 batch 同步顺序执行 |
| 边云 PD 分离 | `enable_edge_cloud=True, enable_pd_separation=True` | 内部固定 2 | `LayerShardLoader` 切层（**复用 PDmix**） | PD batch 分离调度 |

**CLI 启动示例：**

```bash
# 模式 A：原生 PP（与边云无关）
vllm serve model --pipeline-parallel-size 2 --tensor-parallel-size 2

# 模式 B：边云协同（PDmix 原有流程，单 batch 同步顺序执行）
vllm serve model \
    --enable-edge-cloud --edge-npu-count 2 --cloud-npu-count 4 \
    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge","edge_head_tail_layers":1}}'

# 模式 C：边云 PD 分离（本版本目标，多 batch 异步流水线）
vllm serve model \
    --enable-edge-cloud --enable-pd-separation \
    --edge-npu-count 2 --cloud-npu-count 4 \
    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge","edge_head_tail_layers":1}}'
```

> **注意：** 模式 B 和模式 C 在 Phase 1 的模型加载层、通信组层、进程布局层**完全一致**，都复用 PDmix 已有实现。区别仅在于调度执行层（模式 B 单 batch 串行，模式 C 多 batch 异步分离）。

---

## 1 功能点 1：边侧加载首尾层、云测加载中间层

> 参考：`vllm-ascend-pdmix` 中的 `EdgeCloudConfig`、`EdgeCloudLayerPlan`、`LayerShardLoader` 以及 `model_runner_v1.py` 中的 `_load_model_edge_cloud`。

### 1.1 子任务拆分

| 子任务 | 描述 | 涉及仓 |
|--------|------|--------|
| 1.1.1 配置层扩展 | 在 `vllm` 和 `vllm-ascend` 中增加边云协同配置入口 | vllm + vllm-ascend |
| 1.1.2 模型加载改造 | 模型构建时全量构建，加载权重时按角色 shard | vllm-ascend |
| 1.1.3 模型执行改造 | 支持按 segment（首段 / 中段 / 尾段）forward | vllm-ascend |
| 1.1.4 Patch 模型层 | 为目标模型（如 Qwen3.5）添加 `forward_edge_cloud_segment` 方法 | vllm-ascend |

### 1.2 详细设计

#### 1.2.1 配置层扩展（子任务 1.1.1）

**vllm 仓（通用配置）：**

在 `vllm/config/parallel.py` 的 `ParallelConfig` 中新增 CLI 级开关：

```python
enable_edge_cloud: bool = False
enable_pd_separation: bool = False
edge_npu_count: int = 0
cloud_npu_count: int = 0
is_edge_node: bool = False
```

在 `vllm/engine/arg_utils.py` 中注册 CLI 参数：
- `--enable-edge-cloud`（bool）
- `--enable-pd-separation`（bool）
- `--edge-npu-count`（int）
- `--cloud-npu-count`（int）

校验规则见上方 **0.5 配置参数关系与校验规则**。

**vllm-ascend 仓（昇腾专属配置）：**

在 `vllm_ascend/ascend_config.py` 中新增 `EdgeCloudConfig` 类，从 `additional_config` 中解析。逻辑与 PDmix 完全一致，**直接复用**：

```python
class EdgeCloudConfig:
    enabled: bool
    role: str
    mode: str
    edge_head_tail_layers: int | list[int]
    enable_decode_graph: bool
    hidden_dtype: str
```

校验逻辑：
- `role` 必须与 `--headless` 推导出的 `is_edge_node` 一致。
- `mode == "head_tail"` 时，`head_k > 0` 且 `tail_k > 0` 且 `head_k + tail_k < total_layers`。
- `mode == "embedding_only"` 时，强制 `head_k = tail_k = 0`。

CLI 启动示例（模式 C）：
```bash
# 边侧
--enable-edge-cloud --enable-pd-separation --edge-npu-count 2 --cloud-npu-count 4 \
--additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge","edge_head_tail_layers":1}}'

# 云测
--enable-edge-cloud --enable-pd-separation --edge-npu-count 2 --cloud-npu-count 4 \
--additional-config '{"edge_cloud_config":{"enabled":true,"role":"cloud","edge_head_tail_layers":1}}'
```

#### 1.2.2 模型加载改造（子任务 1.1.2）

**核心思想：** Edge-Cloud 模式在运行时借用 PP 通信组做跨节点 hidden state 传输，但在**模型构建阶段**必须绕过原生 PP 的层切分逻辑，先构建出完整模型树，再由 `LayerShardLoader` 按角色决定哪些层保留、哪些层替换为 `PPMissingLayer`。

**vllm-ascend 仓 `vllm_ascend/model_loader/layer_shard_loader.py`：**

**直接复用 PDmix 已有实现，迁移到目标仓。** 关键类：

```python
@dataclass
class EdgeCloudLayerPlan:
    role: str            # "edge" | "cloud"
    total_layers: int
    k: int | list[int] | tuple[int, int]
    mode: str = "head_tail"

    @property
    def head_tail_k(self) -> tuple[int, int]:
        ...

    def get_local_layers(self) -> set[int]:
        if self.role == "edge":
            return set(range(self.head_k)) | set(range(self.total_layers - self.tail_k, self.total_layers))
        return set(range(self.head_k, self.total_layers - self.tail_k))
```

`LayerShardLoader.apply_sharding(model, layer_plan, compilation_config)` 逻辑：
1. 获取 `language_model.model.layers`。
2. 遍历所有层，不在 `local_layers` 中的替换为 `PPMissingLayer()`。
3. 若 `role == "cloud"`，把 `embed_tokens`、`norm`、`lm_head` 也替换为 `PPMissingLayer`。
4. 清理 `compilation_config.static_forward_context` 中指向被释放层的条目（防止图编译引用已删除模块）。
5. 校验：本地层必须是真实模块，非本地层必须是 `PPMissingLayer`。

**vllm-ascend 仓 `model_runner_v1.py` 中模型加载流程：**

新增 `_load_model_edge_cloud(self)` 方法：

```python
def _load_model_edge_cloud(self) -> None:
    # 1. 临时 patch get_pp_indices，使模型构建时不按 PP 切分
    orig_get_pp_indices = dist_utils.get_pp_indices
    dist_utils.get_pp_indices = lambda n, r, s: (0, n)
    try:
        model = initialize_model(self.vllm_config)
    finally:
        dist_utils.get_pp_indices = orig_get_pp_indices

    # 2. 应用 LayerShardLoader
    layer_plan = EdgeCloudLayerPlan(
        role=self.edge_cloud_cfg.role,
        total_layers=len(transformer_model.layers),
        k=[self.head_k, self.tail_k],
        mode=self.edge_cloud_cfg.mode,
    )
    LayerShardLoader.apply_sharding(model, layer_plan, self.vllm_config.compilation_config)

    # 3. 加载权重（只加载本地层权重，PPMissingLayer 的权重为空）
    model_loader.load_weights(model, self.vllm_config.model_config)
    model.to(device)
    process_weights_after_loading(model, ...)
```

> **重要：** 在 PD batch 分离版本中，rank0（边侧）和 rank1（云测）各自运行在独立的进程中。模型加载时，`vllm_config.parallel_config.is_edge_node` 已经决定了该进程的角色，因此 `layer_plan.role` 可以据此正确设置。

#### 1.2.3 模型执行改造（子任务 1.1.3）

**核心思想：** 把一次完整的 `forward` 拆成多段调用（segment），每段只执行自己本地持有的层。边侧执行首段和尾段，云测执行中段。

**完全复用 PDmix 已有实现。** `model_runner_v1.py` 中的 `_create_segment_callable`、`_wrap_segment_if_needed`、segment 划分逻辑均直接迁移。

**Segment 定义：**

```python
def _create_segment_callable(
    self, model, start_layer, end_layer,
    is_first_segment=None, is_last_segment=None
):
    def _segment_forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        return model.forward_edge_cloud_segment(
            start_layer, end_layer,
            input_ids, positions,
            intermediate_tensors, inputs_embeds,
            is_first_segment=is_first_segment,
            is_last_segment=is_last_segment,
            **kwargs,
        )
    return _segment_forward
```

**边侧和云测的 segment 划分：**

假设模型有 `N` 层，`head_k` 为首层数，`tail_k` 为尾层数：

| 节点 | Segment | 层范围 | is_first_segment | is_last_segment |
|------|---------|--------|------------------|-----------------|
| 边侧 | segment_a | `[0, head_k)` | True | False |
| 云测 | segment_c | `[head_k, N - tail_k)` | False | False |
| 边侧 | segment_e | `[N - tail_k, N)` | False | True |

**执行流程（以 Prefill 为例）：**

1. **边侧 P 首**：调用 `segment_a_wrapper(input_ids, positions)` → 输出 `IntermediateTensors(hidden_states, residual)`。
2. **边→云 hidden state 传输**：通过 PP 通信组（或边云专用通道）发送 `IntermediateTensors`。
3. **云测 P 中**：接收 `IntermediateTensors`，调用 `segment_c_wrapper(..., intermediate_tensors=...)` → 输出新的 `IntermediateTensors`。
4. **云→边 hidden state 回传**：发送 `IntermediateTensors` 回边侧。
5. **边侧 P 尾**：接收 `IntermediateTensors`，调用 `segment_e_wrapper(..., intermediate_tensors=...)` → 输出最终 `hidden_states`。
6. **采样**：边侧执行 `lm_head` + `sampler`，生成 logits 和 next token。

> **与 PD batch 分离的衔接：** 当前 PD batch 分离版本已经通过 ZMQ 把 `SchedulerOutput` 从 rank0 发送到 rank1。Phase 1 不改动这一调度链路，只改造**模型层加载**和**准备 segment callable**。真正的跨段 hidden state 传输（步骤 2、4）属于 Phase 2/3/4 的范畴。

#### 1.2.4 Patch 模型层（子任务 1.1.4）

**vllm-ascend 仓 `vllm_ascend/patch/models/`：**

**直接复用 PDmix 已有 patch，迁移到目标仓。** 为目标模型（如 Qwen3.5）添加 `forward_edge_cloud_segment` 方法。该方法与标准 `forward` 的区别：

```python
def forward_edge_cloud_segment(
    self,
    start_layer: int,
    end_layer: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    is_first_segment: bool | None = None,
    is_last_segment: bool | None = None,
    **extra_layer_kwargs,
) -> torch.Tensor | IntermediateTensors:
    # is_first_segment=True: 执行 embed_input_ids，初始化 residual=None
    # is_first_segment=False: 从 intermediate_tensors 中取出 hidden_states 和 residual
    # 执行 layers[start_layer:end_layer]
    # is_last_segment=False: 返回 IntermediateTensors(hidden_states, residual)
    # is_last_segment=True: 执行 norm，返回 hidden_states
```

同时需要为 MoE 模型 patch `update_physical_experts_metadata` 和 `set_moe_parameters`，使其跳过 `PPMissingLayer` 层，正确统计物理专家数量。

> **优先级：** Phase 1 先支持 Qwen3.5/Qwen3.5-MoE（已有参考实现），后续再扩展到其他模型架构。

### 1.3 验收标准

- `vllm serve` 在边侧和云测均能成功拉起。
- 日志中 `LayerShardLoader` 打印的 kept_layers / skipped_layers 符合预期：
  - 边侧 kept = `[0]` + `[N-1]`（当 `edge_head_tail_layers=1`）
  - 云测 kept = `[1, ..., N-2]`
- `nvidia-smi` / `npu-smi info` 观察到的显存占用符合不均等预期（云测 > 边侧）。
- 模型加载后 `_load_model_edge_cloud` 中的校验通过，无 `RuntimeError`。

---

## 2 功能点 2：边侧与云测 TP 不均等划分

> 参考：`vllm-pdmix` 中的 `parallel.py`、`parallel_state.py`、`arg_utils.py`；`vllm-ascend-pdmix` 中的 `patch_multiproc_executor.py`、`parallel_state.py`。

### 2.1 子任务拆分

| 子任务 | 描述 | 涉及仓 |
|--------|------|--------|
| 2.1.1 并行配置改造 | `ParallelConfig` 支持 `enable_edge_cloud` 及相关校验 | vllm |
| 2.1.2 分布式通信组改造 | 按边云模式重新划分 TP / PP / DCP / PCP / DP / EP 组 | vllm |
| 2.1.3 Executor / Worker 改造 | `MultiprocExecutor` 和 `WorkerProc` 支持不均等 rank 布局 | vllm + vllm-ascend |
| 2.1.4 KV Cache 与 BlockTable 适配 | 处理边云模式下 CP 相关内存分配 | vllm-ascend |

### 2.2 详细设计

#### 2.2.1 并行配置改造（子任务 2.1.1）

**vllm 仓 `vllm/config/parallel.py`：**

直接复用 PDmix 已有字段，新增 `enable_pd_separation`：

```python
enable_edge_cloud: bool = False
enable_pd_separation: bool = False
edge_npu_count: int = 0
cloud_npu_count: int = 0
is_edge_node: bool = False
```

`__post_init__` 关键逻辑（与 PDmix 一致，仅增加 `enable_pd_separation` 校验）：

```python
if self.enable_pd_separation and not self.enable_edge_cloud:
    raise ValueError("--enable-pd-separation requires --enable-edge-cloud")

if self.enable_edge_cloud:
    if self.edge_npu_count <= 0 or self.cloud_npu_count <= 0:
        raise ValueError("edge_npu_count and cloud_npu_count must be positive")
    if self.edge_npu_count >= self.cloud_npu_count:
        raise ValueError("edge_npu_count must be less than cloud_npu_count")
    if self.data_parallel_size != 1:
        raise ValueError("data_parallel_size must be 1 in edge-cloud mode")

    # pipeline_parallel_size 由边云模式内部管理，用户无需配置
    if self.pipeline_parallel_size != 1:
        logger.warning(
            "--pipeline-parallel-size is ignored in edge-cloud mode. "
            "Internally fixed to 2."
        )
    self.pipeline_parallel_size = 2

    self.world_size = self.edge_npu_count + self.cloud_npu_count
    self.tensor_parallel_size = (
        self.edge_npu_count if self.is_edge_node else self.cloud_npu_count
    )
```

**`local_world_size` 属性（复用 PDmix）：**

```python
@property
def local_world_size(self) -> int:
    if self.enable_edge_cloud:
        return self.edge_npu_count if self.is_edge_node else self.cloud_npu_count
    return self.world_size // self.nnodes_within_dp
```

这决定了每个节点启动多少个 worker 进程。

> **复用说明：** 以上逻辑与 `vllm-pdmix` 仓完全一致，无需改动。`enable_pd_separation` 仅作为一个布尔标记，在后续调度层（EngineCore/PassiveScheduler）中判断是否需要启用 PD batch 分离调度。

#### 2.2.2 分布式通信组改造（子任务 2.1.2）

**vllm 仓 `vllm/distributed/parallel_state.py`：**

**完全复用 PDmix 已有实现。** 当 `enable_edge_cloud=True` 时（无论 `enable_pd_separation` 是否开启），在 `initialize_model_parallel` 中跳过标准均匀 rank 布局，改用边云专用布局：

```python
if parallel_config.enable_edge_cloud:
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    edge_npu_count = parallel_config.edge_npu_count
    is_edge = rank < edge_npu_count
    _IS_EDGE_DEVICE = is_edge

    # TP 组：边侧所有 rank 一组，云测所有 rank 一组
    tp_edge_ranks = list(range(edge_npu_count))
    tp_cloud_ranks = list(range(edge_npu_count, world_size))
    _TP = init_model_parallel_group(
        [tp_edge_ranks, tp_cloud_ranks],
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )

    # PP 组：只把边侧 rank0 和云测 rank0 放入同一 PP 组
    pp_group_ranks = [0, edge_npu_count]
    pp_other_ranks = [[r] for r in range(world_size) if r not in (0, edge_npu_count)]
    _PP = init_model_parallel_group(
        [pp_group_ranks] + pp_other_ranks,
        get_world_group().local_rank,
        backend,
        group_name="pp",
    )

    # DCP / PCP / DP / EP：每个 rank 各自独立成组（单 rank 组）
    all_ranks = list(range(world_size))
    for name, store in (("dcp", _DCP), ("pcp", _PCP), ("dp", _DP), ("ep", _EP)):
        store = init_model_parallel_group(
            [[r] for r in all_ranks],
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=(name in ("dcp",)),
            group_name=name,
        )
```

**关键差异：**
- **TP 组**：边侧 `edge_npu_count` 个 rank 组成一个 TP 组；云测 `cloud_npu_count` 个 rank 组成另一个 TP 组。两组大小不等。
- **PP 组**：只有边侧 leader（global rank 0）和云测 leader（global rank `edge_npu_count`）在同一个 PP 组中。其他 rank 各自为单 rank PP 组（这些 rank 不参与跨节点 PP 通信，只在本节点内做 TP all-reduce）。
- **DCP/PCP/DP/EP**：全部降级为单 rank 组，因为边云协同目前不支持跨节点的 CP 或 DP。

**`is_edge_device()` 全局标记：**

```python
_IS_EDGE_DEVICE: bool | None = None

def is_edge_device() -> bool:
    return _IS_EDGE_DEVICE is True
```

在 `initialize_model_parallel` 中根据 `rank < edge_npu_count` 设置，供后续模型执行逻辑判断当前是否在边侧。

#### 2.2.3 Executor / Worker 改造（子任务 2.1.3）

**完全复用 PDmix 已有实现。** 以下为 `vllm-pdmix` / `vllm-ascend-pdmix` 中的已有逻辑，Phase 1 直接迁移，无改动。

**vllm 仓 `vllm/v1/executor/multiproc_executor.py`：**

在 `MultiprocExecutor._init_executor` 中：

1. 当 `enable_edge_cloud=True` 时，**不再断言** `world_size == TP * PP * PCP`（因为边云模式下这个等式不成立）。
2. 计算 `global_start_rank`：
   ```python
   if self.parallel_config.enable_edge_cloud:
       global_start_rank = (
           0 if self.parallel_config.is_edge_node else self.parallel_config.edge_npu_count
       )
   else:
       global_start_rank = self.local_world_size * self.parallel_config.node_rank_within_dp
   ```
3. 创建 worker 时，`global_rank = global_start_rank + local_rank`。

**`is_driver_worker` 判断：**

```python
def _is_driver_worker(self, rank: int) -> bool:
    if self.parallel_config.enable_edge_cloud:
        return rank == (
            0 if self.parallel_config.is_edge_node else self.parallel_config.edge_npu_count
        )
    return rank % self.parallel_config.tensor_parallel_size == 0
```

边侧的 driver worker 是 rank 0，云测的 driver worker 是 rank `edge_npu_count`。

**`_get_output_rank`：**

```python
def _get_output_rank(self) -> int:
    if self.parallel_config.enable_edge_cloud:
        return 0
    return super()._get_output_rank()
```

输出永远从边侧 rank 0 收集（因为 LM Head 在边侧）。

**vllm-ascend 仓 `vllm_ascend/patch/platform/patch_multiproc_executor.py`：**

`AscendMultiprocExecutor` 继承 `MultiprocExecutor`，复写上述方法（逻辑与 vllm-pdmix 中一致，只是把 GPU 概念替换为 NPU）。

**Worker MessageQueue 初始化：**

当 `VLLM_PP_NON_LEADER_ENGINE_CORE`（即云测 passive EngineCore）时，worker 需要维护两套 MQ：
- **local_mq**：用于与本地 passive EngineCore 握手（接收 `SchedulerOutput`）。
- **cross_mq**：用于与边侧 worker 做实际的 PP 通信（发送/接收 hidden state）。

```python
def _init_message_queues(self, input_shm_handle, vllm_config):
    if envs.VLLM_PP_NON_LEADER_ENGINE_CORE:
        # Local MQ (passive enginecore handshake)
        self.local_rpc_broadcast_mq = MessageQueue.create_from_handle(input_shm_handle, self.local_rank)
        self.local_worker_response_mq = MessageQueue(1, 1)
        # Cross-node MQ (actual PP work with rank0)
        self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(...)
        self.worker_response_mq, self.peer_response_handles = (
            get_inner_dp_world_group().create_single_reader_mq_broadcasters(reader_rank_in_group=0)
        )
    else:
        super()._init_message_queues(input_shm_handle, vllm_config)
```

> **注意：** 在 PD batch 分离版本中，rank1（云测）已经具备 `VLLM_PP_NON_LEADER_ENGINE_CORE` 逻辑。Phase 1 只需要把 `AscendMultiprocExecutor` 的 `global_start_rank` 和 `is_driver_worker` 逻辑迁移过来，使其支持不均等 rank 布局。

#### 2.2.4 KV Cache 与 BlockTable 适配（子任务 2.1.4）

**复用 PDmix 已有实现。** 以下为 `vllm-ascend-pdmix` 中的已有逻辑，Phase 1 直接迁移。

**vllm-ascend 仓 `vllm_ascend/worker/block_table.py`：**

在边云模式下，由于 `pcp_world_size` 和 `dcp_world_size` 均为 1（单 rank 组），`BlockTable` 的逻辑基本无需修改。唯一需要注意的是：当 `kernel_sizes` 与 `block_size` 不匹配时，已有 fallback 逻辑会自动回退到不分片：

```python
# block_table.py 中已有逻辑
if selected_kernel_size is None:
    self.block_size = block_size
    self.logical_block_size = block_size
    self.blocks_per_phys_block = 1
    self.use_hybrid_blocks = False
```

**`model_runner_v1.py` 中的 KV Cache 分配：**

`get_kv_cache_config` 和 `determine_num_available_blocks` 需要基于本地实际参与计算的层数来分配 KV cache。由于边侧只持有首尾层，云测只持有中间层，KV cache 的 `num_kv_cache_groups` 和 `num_blocks` 计算逻辑应继续使用 vLLM 原生的 `get_layers_from_vllm_config`，但需要确保：
- 边侧的 KV cache 只覆盖首尾层的 attention 模块。
- 云测的 KV cache 只覆盖中间层的 attention 模块。

在已有实现中，`LayerShardLoader.apply_sharding` 已经把非本地层替换为 `PPMissingLayer`，而 `PPMissingLayer` 的 `self.layer_name` 等属性不存在，因此 KV cache 的自动 profiling 只会为本地真实层分配空间（因为 profile run 只执行本地 segment）。

### 2.3 验收标准

- `vllm serve` 在边侧（2 NPU）和云测（4 NPU）均能成功拉起。
- `initialize_model_parallel` 日志打印的通信组符合预期：
  - TP edge ranks = `(0, 1)`
  - TP cloud ranks = `(2, 3, 4, 5)`
  - PP group ranks = `(0, 2)`
- `npu-smi info` 或 `nvidia-smi` 显示边侧 2 卡、云测 4 卡均被占用。
- 进程列表中 worker 数量正确：边侧 2 个 worker，云测 4 个 worker。
- `_is_driver_worker` 判断正确：边侧 rank0 为 driver，云测 rank2 为 driver。

---

## 3 Phase 1 实施顺序

Phase 1 的核心原则是：**模型加载层和通信组层完全复用 PDmix 已有实现；仅需新增 `enable_pd_separation` 配置标记和校验逻辑。**

| 顺序 | 任务 | 涉及仓 | 说明 |
|------|------|--------|------|
| 1 | `ParallelConfig` 新增 `enable_pd_separation` + CLI 注册 | vllm | 新增参数和校验规则 |
| 2 | `parallel_state.py` 边云通信组布局 | vllm | **复用 PDmix**，直接迁移 |
| 3 | `MultiprocExecutor` / `AscendMultiprocExecutor` 不均等 rank 布局 | vllm + vllm-ascend | **复用 PDmix**，直接迁移 |
| 4 | `EdgeCloudConfig` + `EdgeCloudLayerPlan` + `LayerShardLoader` | vllm-ascend | **复用 PDmix**，直接迁移 |
| 5 | `model_runner_v1.py` 中 `_load_model_edge_cloud` + segment callable | vllm-ascend | **复用 PDmix**，直接迁移 |
| 6 | `qwen3_5_edge_cloud.py` patch | vllm-ascend | **复用 PDmix**，直接迁移 |
| 7 | `enable_pd_separation` 与 EngineCore/PassiveScheduler 的衔接 | vllm | 在调度层判断 `if parallel_config.enable_pd_separation:` 时启用 PD 分离调度（Phase 1 只做标记接入，不实现调度逻辑） |

**说明：**
- 子任务 1~6 均为**代码迁移**（从 `vllm-pdmix` / `vllm-ascend-pdmix` 复制到目标仓），无逻辑改动。
- 子任务 7 为**标记接入**：在 `EngineCore` / `PassiveScheduler` 等调度执行代码中，读取 `parallel_config.enable_pd_separation`，为后续 Phase 2~7 的调度逻辑提供开关。Phase 1 本身不实现任何调度逻辑。
- 所有复用代码需要检查与目标仓版本的兼容性（如函数签名、导入路径变化），但算法和逻辑不变。

---

## 4 不在 Phase 1 范围（后续阶段处理）

| 能力 | 所属阶段 | 原因 |
|------|----------|------|
| Hidden state 跨节点传输（PP 通信） | Phase 2 / 3 / 4 | Phase 1 只负责模型层加载正确，不打通执行流程 |
| SchedulerOutput 从云测回传边侧 | Phase 3 / 4 | Phase 1 只改造 PassiveScheduler 接收端，不涉及新的 ZMQ 回传链路 |
| P 首 / P 尾 / D 首 / D 尾 队列与调度算法 | Phase 5 | Phase 1 仅在调度层接入 `enable_pd_separation` 标记，不实现具体调度策略 |
| PP 双通道拓展为三通道 | Phase 6 | Phase 1 保持现有双通道通信组 |
| 1P1D / 2P1D batch 调度算法 | Phase 5 / 7 | Phase 1 不涉及请求级调度 |
| 多模型架构支持（LLaMA、DeepSeek 等） | Phase 2+ | Phase 1 先支持 Qwen3.5，后续再扩展 patch |

---

## 5 风险与注意事项

1. **复用代码的兼容性**
   - `vllm-pdmix` / `vllm-ascend-pdmix` 与目标仓（vllm_dev）的版本基线可能不同。迁移时需要检查函数签名、导入路径、类结构是否有差异。建议逐文件对比后再复制。

2. **`enable_pd_separation` 与 `enable_edge_cloud` 的依赖关系**
   - 必须在 `__post_init__` 中严格校验 `enable_pd_separation` 要求 `enable_edge_cloud`。如果用户只配置了 `--enable-pd-separation` 而没有 `--enable-edge-cloud`，应给出明确的错误提示。

3. **与现有 PP=2 逻辑的冲突**
   - 当前 PD batch 分离版本已经硬编码了 `pipeline_parallel_size=2` 的 PP 逻辑（rank0 前半、rank1 后半）。启用 `enable_edge_cloud` 后，需要确保 `get_pp_indices` 在模型构建阶段被临时替换，否则原生 PP 逻辑会在 `initialize_model` 时就切掉一半层。

4. **TP all-reduce 与 PPMissingLayer**
   - `PPMissingLayer` 的 forward 是 `return input`，不会产生梯度，也不会参与 TP all-reduce。但由于 TP 组内所有 rank 都持有相同的层结构（只是某些层是 `PPMissingLayer`），TP all-reduce 的 collective 操作仍然可以正常执行（对空张量/no-op 是安全的）。

5. **Compilation Config 清理**
   - `LayerShardLoader._clean_compilation_config` 必须正确移除被释放层的 `static_forward_context`，否则图编译阶段会尝试访问已删除模块导致 `AttributeError`。

6. **MoE 模型的专家统计**
   - `set_moe_parameters` 和 `update_physical_experts_metadata` 在遍历 `model.layers` 时必须跳过 `PPMissingLayer`，否则会因为访问 `layer.mlp.experts` 而报错。已有 patch（`qwen3_5_edge_cloud.py`）已处理。

7. **World Size 校验**
   - 标准 vLLM 在多处断言 `world_size == TP * PP * PCP * DP`。边云模式下 `world_size = edge_npu_count + cloud_npu_count`，而 `TP` 是节点内局部值（边侧 TP = edge_npu_count，云测 TP = cloud_npu_count），全局不等式不成立。所有相关断言都需要加上 `not enable_edge_cloud` 条件保护。
