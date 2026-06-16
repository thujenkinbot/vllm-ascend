# PDbatch分离分布式边云协同推理调度算法设计
# 1 PD掩盖
PDbatch分离分布式边云协同推理，一个请求的完整处理流程为：Prefill -> Decode -> 若干个Decode -> 结束  
边侧只加载首层和尾层，云侧加载transformer中间层，云测为推理计算的主力，我们通常希望云测能够吃满计算资源，边侧按需调度。  
但由于边云协同推理的特性，云侧做执行先后两个batch之间必然存在计算气泡，以decode为例，计算气泡为边到云传输时延 + 边D尾执行时间+D首执行时间+云到边传输时延。这导致边云协同推理的吞吐性能相比普通推理场景下降。  
> D中 -> 边到云传输时延 -> D尾 -> D首 -> 云到边传输时延 -> D中  

为了提高边云协同推理的吞吐性能，我们采用一种叫**PD掩盖的调度策略**，即在云测执行两个D中之间，插入一个P中，以隐藏计算气泡，提高吞吐性能。
> D中 -> P中(边到云传输时延 -> D尾 -> D首 -> 云到边传输时延) -> D中  

通常情况下，P中的数量远远小于D中的数量，故在云侧通过层切分slice来增加P中的数量，当P中数量与D中数量整体接近且做P中时间能够完全覆盖计算气泡时，吞吐性能最佳。
综上，为了实现最好的PD掩盖流水，云测通常需要Prefill/Decode 来回切换调度，边侧调度的目的主要是能够做到支持云测的穿插调度
&nbsp;
# 2 1P1D VS 2P1D
1P1D是指，同一时间内系统再跑的Prefill batch 和 Decode batch 都为1个。当云侧按层切分P中数量与D中数量整体接近时，能够掩盖绝大部分的通信气泡。
1P1D调度的缺陷在于，在两个Prefill batch 之间，必须等待前一个Prefill batch 执行完成，才能下发下一个Prefill batch。当下个Prefill batch 的P中未就绪时云侧就会面临无P可做的局面，只能连着调度D中，这部分的计算气泡无法掩盖。
> P中 -> 边到云传输时延 -> P尾 -> P首 -> 云到边传输时延 -> P中  
> D中 -> P中(batch最后一个) -> D中 -> D中（无P可做）-> ... -> P中（下一个batch就绪）

由于两个P batch 之间也存在通信时延，当通信时延很大时，1P1D调度在两个P batch 之间必然产生无法掩盖的气泡。
为了将两个P batch 之间的通信气延隐藏起来，我们采用2P1D调度。2P1D调度下，边侧会提前下发第二个Prefill batch，使下个P中的就绪时间提前，从而实现最完美的PD掩盖流水。
&nbsp;


# 3 边侧调度状态机设计
|状态|系统Prefill batch 在飞数|说明|限制|
|:---:|:---:|:---:|:---:|
|IDLE|0|无Prefill batch /chunk prefill batch 在飞。初始状态|无P尾可下发|
|LOW|1|1个Prefill batch 或 chunk prefill batch 在飞|-|
|HIGH|2|1个Prefill batch 或 chunk prefill batch 在飞|不可下发P首|
- `prefill_inflight_count` 为系统PrefPrefill batch /chunk prefill batch 在飞数。
- `prefill_inflight_limit` 为系统Prefill batch /chunk prefill batch 在飞数上限。
- `decode_inflight_count` 为系统Decode batch 在飞数, 固定限制最大值为1。
- `prefill_inflight_limit == 1`时，实现1P1D调度, 状态机状态有`IDLE`、`LOW`。
- `prefill_inflight_limit == 2`时，实现2P1D调度, 状态机状态有`IDLE`、`LOW`、`HIGH`。

# 4 边侧各状态的调度优先级
**IDLE**
|batch type|请求来源|优先级(数字小优先级高)|状态机变化|
|:---:|:---:|:---:|:---:|
|P首|waiting[]|1|IDLE -> LOW|
|chunk0首|waiting[]|1|IDLE -> LOW|
|D首|running[]|2|IDLE|
|D尾|decodes_last_ready[]|3|IDLE|
|Empty|-|4|IDLE|
- chunk0首 表示请求的首个chunk prefill的首

&nbsp;  
**LOW**
|batch type|请求来源|优先级(数字小优先级高)|状态机变化|
|:---:|:---:|:---:|:---:|
|chunk(i>0)首|chunk_prefill_first[]|1|LOW -> HIGH|
|P首|waiting[]|2|LOW -> HIGH|
|P尾 / chunk(i)尾|prefills_last_ready[]|3|LOW ->IDLE|
|D首|running[]|4|LOW|
|D尾|decodes_last_ready[]|5|LOW|
|Empty|-|6|LOW|
- chunk(i>0)首 表示请求的非首个chunk prefill的首
- chunk(i)尾 表示请求chunk prefill的尾

&nbsp;  
**HIGH**
|batch type|请求来源|优先级(数字小优先级高)|状态机变化|
|:---:|:---:|:---:|:---:|
|P尾 / chunk(i)尾|prefills_last_ready[]|1|HIGH ->LOW|
|D首|running[]|2|HIGH|
|D尾|decodes_last_ready[]|3|HIGH|
|Empty|-|4|HIGH|

# 5 云侧调度状态机设计
|状态|说明|状态机变化|
|:---:|:---:|:---:|
|EXPECT_EXECUTE_PREFILL|本轮期望调度P中|若本轮成功调度P中，则状态转移，否则状态不变|
|EXPECT_EXECUTE_DECODE|本轮期望调度D中|若本轮成功调度D中，则状态转移，否则状态不变|

# 6 云侧各状态的调度优先级
**EXPECT_EXECUTE_PREFILL**
|batch type|请求来源|优先级|状态机变化|
|:---:|:---:|:---:|:---:|
|P中|ready_prefills[]|1|EEP -> EED|
|D中|ready_decodes[]|2|EEP|
|empty|-|3|EEP|

&nbsp;  
**EXPECT_EXECUTE_DECODE**
|batch type|请求来源|优先级|状态机变化|
|:---:|:---:|:---:|:---:|
|D中|ready_decodes[]|1|EED -> EEP|
|P中|ready_prefills[]|2|EED|
|empty|-|3|EEP|
