# PD/batch分离分布式边云协同推理设计说明书

边云协同将大预言模型（LLM）的不同层分散部署在边侧（Edge/Master）和云测（Cloud/Slave）,通过边云协同完成推理
## 1 角色划分
边侧（Edge）执行模型的首层（Head）和尾层（Tail）,包括Embedding、首若干Transformer层、尾若干Transformer层、LM Head 等。默认情况为首1层尾1层
云测（Cloud）执行模型的中间层（Layers），即边侧首层之后，尾层之前的全部Transformer层。
## 2 执行过程
### 2.1 Prefill 阶段
1. Edge 首层执行：边侧执行head部分（embedding + 首层），生成中间hidden state
2. 边云数据传输：边侧通过HCCL isend 将 hidden state 异步发送给云测；云测通过irecv异步接收
3. Cloud 中间层执行：云测执行中间层，生成输出hidden state
4. 云边数据回传: 云测将结果 hidden state 发送回边侧。
5. Edge 尾层执行：边侧执行 tail 部分（尾层 + lm_head）,生成logits
### 2.2 Decode 阶段
流程与Prefill 类似，但处理的是单个token的迭代生成：
- Edge执行head -> 发送hidden -> CLoud 执行layers -> 返回 hidden -> Edge执行tail -> 采样

## 3 基于PDmix的边云协同推理版本

vllm-pdmix仓和vllm-ascend-pdmix仓已基于PDmix实现了边云协同推理，vllm服务启动命令参考如下：   

**边侧(rank0)：**  
vllm server Qwen3.6-27B \
    --host 0.0.0.0 \
    --port 8060 \
    --master-addr 76.76.26.194 \
    --master-port 29501 \
    --served-model-name qwen3.6 \
    --trust-remote-code \
    --nnodes 2 \
    --node-rank 0 \
    --enable-edge-cloud \
    --edge-npu-count 2 \
    --cloud-npu-count 4 \
    --additional-config '{"edge_cloud_config":{"enble":true, "role":"edge", "edge_head_tail_layers":1}}' \
    --compilation-config '{"cudagraph_mode":"NONE", "cudagraph_capture_sizes":[2,4,6,8,10,12,14,16,18,20,22,24,32,36,40]}' 

**云侧(rank1)：**  
vllm server Qwen3.6-27B \
    --host 0.0.0.0 \
    --port 8060 \
    --master-addr 76.76.26.194 \
    --master-port 29501 \
    --served-model-name qwen3.6 \
    --trust-remote-code \
    --nnodes 2 \
    --node-rank 1 \
    --enable-edge-cloud \
    --edge-npu-count 2 \
    --cloud-npu-count 4 \
    --additional-config '{"edge_cloud_config":{"enble":true, "role":"cloud", "edge_head_tail_layers":1}}' \
    --compilation-config '{"cudagraph_mode":"NONE", "cudagraph_capture_sizes":[2,4,6,8,10,12,14,16,18,20,22,24,32,36,40]}' 

功能点实现：
1、边侧加载首尾层
2、边侧与云测tp不均等划分，边侧加载npu少，云测加载npu多

## 4 基于PD batch分离的边云协同推理版本
### 4.1 PD batch分离已实现功能
vllm仓与vllm-ascend仓（已同步vllm-pdmix仓和vllm-ascend-pdmix仓）已基于PP=2实现了PDbatch严格分离下发的部分功能：   
1、rank0(边侧)Prefiil/Decode batch 严格分离下发  
2、rank0(边侧)通过ZMQ将SchedulerOutput发送给rank1(云测)  
3、云测拉起PasiveEngineCore和PasiveEngineCore, 在收到SchedulerOutput后下发给Excutor执行  
4、PP 双通道通信组，prefill与decode的hidden state的传输使用不同的通信组，互不影响。  

### 4.2 PD batch分离到边云协同推理的整体差距
但离真正实现边云协同推理仍有较大差距，主要差距如下：  
1、现在rank0(边侧) 和 rank1(云测) 的模型层加载仍基于原有pp双机逻辑（即rank0加载模型前半，rank1加载模型后半），需要参考现有pdmix边云协同推理版本，实现边侧加载首尾层、云测加载中间层功能  
2、由于云测加载模型和执行推理的压力大，需要支持边侧与云测tp不均等划分，可参考pdmix边云协同推理版本`--edge-npu-count`、`--cloud-npu-count`功能的实现  
3、边侧下发的batch类型现在只区分Prefiil和Decode, 需要拓展为"Prefiil first"/"Prefiil last"/"Decode first"/"Decode last"四种类型，即P首/P尾/D首/D尾
4、边侧PDSeparatedScheduler需要新增队列prefill_last_ready[]和decode_last_ready[]，接收云测做完中间层后发来的SchedulerOutput，边侧可以从这两个队列中取出请求来执行P尾/D尾  
5、边侧需要一个调度算法，决定每轮step执行P首/P尾/D首/D尾中的哪一个；同理云测也需要一个调度算法，决定每轮step执行P中/D中的哪一个  
6、rank1(云测)做完中间层后也需要将SchedulerOutput发送给rank0(边侧)，同理也需要云测到边侧的hidden state的传输  

### 4.3 PD batch分离边云协同推理开发计划
|阶段|功能点实现|PDmix边云协同推理功能参考|测试验收|已实现|
|:---:|:---:|:---:|:---:|:---:|
|phase 1|边侧加载首尾层、云测加载中间层|"edge_head_tail_layers"|vllm服务成功拉起，边云模型层数加载正确|✅|
|phase 1|边侧与云测tp不均等划分|`--edge-npu-count`、`--cloud-npu-count`|vllm服务成功拉起，NPU加载正确|✅|
|phase 2|边侧P首请求下发执行成功并发送SchedulerOutput与hidden state|-|流程正确，正确输出日志信息|✅|
|phase 2|云测P中请求下发执行成功并发送SchedulerOutput与hidden state|-|流程正确，正确输出日志信息|✅|
|phase 3|边侧P尾请求下发执行成功并发送SchedulerOutput与hidden state|-|流程正确，正确输出日志信息|✅|
|phase 4|边侧D首请求下发执行成功、云测D中请求下发执行成功、边侧D尾请求下发执行成功|-|流程正确，正确输出日志信息|✅|
|phase 4|单请求执行过程打通|-|打curl能够正常输出|✅|
|phase 5|边侧实现1P1D batch调度算法|-|benchmark打多请求能够正常执行，无异常卡死、报错、精度问题|✅|
|phase 5|云侧实现PD batch 穿插调度算法|-|benchmark打多请求能够正常执行，无异常卡死、报错、精度问题|✅|
|phase 6|PP双通道通信组拓展为三通道通信组，传输Prefill hidden state使用双通道，传输Decode hidden state使用单通道|-|benchmark打多请求能够正常执行，无异常卡死、报错、精度问题|❌|
|phase 7|边侧实现2P1D batch调度算法|-|benchmark打多请求能够正常执行，无异常卡死、报错、精度问题，吞吐性能相比1P1D有所提升|✅|