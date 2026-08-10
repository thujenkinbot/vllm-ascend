# Multi-edge shared-cloud MVP

This prototype lets two independent one-NPU edge engines share one eight-NPU
cloud tensor-parallel model replica. The cloud executes one edge scheduling
step at a time using round-robin arbitration. It does not merge batches across
edges.

This is a semantic backport of the topology, per-edge PP groups, edge routing,
round-robin scheduling, and KV isolation from `feat/edge-cloud-multi-edge`.
The v0.23 passive engine is intentionally replaced by a small control adapter
for the v0.20 executor used on this branch.

## Scope and safety

- Static topology: two one-NPU edges and one eight-NPU cloud node.
- All three nodes must start together so HCCL groups can be created.
- Synchronous scheduling (`--no-async-scheduling`), eager execution, and
  `max_num_seqs=1` are required.
- Use the same model, layer split, block override, master address, and master
  port on every node.
- The control endpoint uses cloudpickle. Bind it only on a trusted inference
  network and block access from untrusted clients.
- Multimodal model configurations may start for text-only requests. Image,
  audio, and video request paths have not been validated in this MVP.
- Dynamic edge join/leave, failure isolation, cross-edge batching, speculative
  decoding, LoRA, and prefix caching are outside this MVP.

The global rank layout is:

```text
rank 0       edge 0
rank 1       edge 1
ranks 2..9   cloud TP=8
```

Each edge has an independent PP pair with cloud rank 2. Cloud KV block IDs are
shifted by `edge_id * (cloud_num_blocks / num_edges)`. The worker obtains the
cloud block count from its allocated KV configuration. The optional
`VLLM_ASCEND_EDGE_CLOUD_CLOUD_NUM_BLOCKS` value can assert that the allocation
matches an expected total. Cloud-side request IDs are also prefixed with the
edge identity to isolate worker request state. Each edge scheduler is limited
to half the cloud KV capacity; any remainder is left unused.

## Launch

Replace the addresses, model, and block count below. Start all commands close
together; edge 0 is the cloud-initialization authority.

Cloud node:

```bash
VLLM_ASCEND_EDGE_CLOUD_CONTROL_PORT=5568 \
vllm serve MODEL --headless \
  --enable-edge-cloud --num-edges 2 --cloud-npu-count 8 \
  --distributed-executor-backend mp \
  --nnodes 3 --node-rank 2 --master-addr EDGE0_IP --master-port 29501 \
  --num-gpu-blocks-override 4096 --no-enable-prefix-caching \
  --no-async-scheduling --enforce-eager --max-num-seqs 1 \
  --additional-config '{"edge_cloud_config":{"enabled":true,"role":"cloud","mode":"embedding_only","edge_head_tail_layers":0}}'
```

Edge 0:

```bash
VLLM_ASCEND_EDGE_CLOUD_EDGE_IDX=0 \
VLLM_ASCEND_EDGE_CLOUD_CLOUD_ADDR=CLOUD_IP \
VLLM_ASCEND_EDGE_CLOUD_CONTROL_PORT=5568 \
vllm serve MODEL \
  --enable-edge-cloud --num-edges 2 --cloud-npu-count 8 \
  --distributed-executor-backend mp \
  --nnodes 3 --node-rank 0 --master-addr EDGE0_IP --master-port 29501 \
  --num-gpu-blocks-override 4096 --no-enable-prefix-caching \
  --no-async-scheduling --enforce-eager --max-num-seqs 1 \
  --additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge","mode":"embedding_only","edge_head_tail_layers":0}}'
```

Edge 1 uses a different HTTP port if it runs on the same management host:

```bash
VLLM_ASCEND_EDGE_CLOUD_EDGE_IDX=1 \
VLLM_ASCEND_EDGE_CLOUD_CLOUD_ADDR=CLOUD_IP \
VLLM_ASCEND_EDGE_CLOUD_CONTROL_PORT=5568 \
vllm serve MODEL \
  --enable-edge-cloud --num-edges 2 --cloud-npu-count 8 \
  --distributed-executor-backend mp \
  --nnodes 3 --node-rank 1 --master-addr EDGE0_IP --master-port 29501 \
  --num-gpu-blocks-override 4096 --no-enable-prefix-caching \
  --no-async-scheduling --enforce-eager --max-num-seqs 1 \
  --additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge","mode":"embedding_only","edge_head_tail_layers":0}}'
```

## Prototype checks

1. Send one deterministic request to each edge separately and compare tokens
   with the existing one-edge baseline.
2. Send requests to both edges concurrently and verify both complete.
3. Log `SchedulerOutput.edge_id` on the cloud and verify 0/1 dispatches.
4. Send the same prompt concurrently and compare every generated token with
   baseline, exercising KV namespace isolation.
5. Confirm cloud memory contains one TP=8 logical model replica, not one replica
   per edge.
