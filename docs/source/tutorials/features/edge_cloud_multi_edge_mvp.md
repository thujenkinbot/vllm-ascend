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

## Privacy-preserving usage reporting

The optional edge usage reporter runs as ASGI middleware in each edge API
server. It reads vLLM's final request metadata after the response has
completed and sends a best-effort accounting event to a private Higress
route. The middleware does not read the inference request or response body.

Each report is an HTTP `POST` with a small `application/json` body. It contains
only accounting metadata; prompt text, generated text, token IDs, embeddings,
and user identifiers are not included:

```json
{
  "event_id": "8f9a...",
  "edge_id": "edge-0",
  "model": "qwen3.5-27b",
  "usage": {
    "prompt_tokens": 101,
    "completion_tokens": 37,
    "total_tokens": 138
  },
  "duration_ms": 1234,
  "status_code": 200
}
```

The request also carries these headers:

| Header | Meaning |
| --- | --- |
| `x-api-key` | Per-edge Higress Key Auth credential |
| `x-ai-edge-id` | Stable edge identity for correlation |
| `x-ai-event-id` | Random edge-generated ID for correlation |

Token counts deliberately use JSON numbers. Higress AI Statistics v2.2.3
does not convert numeric strings extracted from headers into Prometheus
counters.

Reports use a bounded in-memory queue. Sending has a two-second timeout and
two retries. A full queue, an unreachable gateway, or a rejected report is
logged and dropped without delaying or failing inference. This MVP reports
normally completed requests only. It does not persist events across process
failure and is intended for observability or prototype accounting, not
financial settlement. Chat Completions and Completions are supported; the
Responses API is outside this MVP because it does not currently publish the
same final usage metadata to middleware.

### Edge configuration

Add these variables and the middleware argument to edge 0. Use the exact
value passed to `--served-model-name` as the report model.

```bash
export VLLM_ASCEND_USAGE_REPORT_URL="https://HIGRESS_HOST/internal/edge-usage"
export VLLM_ASCEND_USAGE_REPORT_API_KEY="EDGE_0_SECRET"
export VLLM_ASCEND_USAGE_REPORT_EDGE_ID="edge-0"
export VLLM_ASCEND_USAGE_REPORT_MODEL="qwen3.5-27b"

vllm serve MODEL \
  ... \
  --middleware vllm_ascend.entrypoints.openai.usage_reporter.UsageReporterMiddleware
```

Edge 1 uses a different credential and edge identity:

```bash
export VLLM_ASCEND_USAGE_REPORT_URL="https://HIGRESS_HOST/internal/edge-usage"
export VLLM_ASCEND_USAGE_REPORT_API_KEY="EDGE_1_SECRET"
export VLLM_ASCEND_USAGE_REPORT_EDGE_ID="edge-1"
export VLLM_ASCEND_USAGE_REPORT_MODEL="qwen3.5-27b"

vllm serve MODEL \
  ... \
  --middleware vllm_ascend.entrypoints.openai.usage_reporter.UsageReporterMiddleware
```

`VLLM_ASCEND_USAGE_REPORT_EDGE_ID` may be omitted; it then defaults to
`edge-${VLLM_ASCEND_EDGE_CLOUD_EDGE_IDX}`. Do not load the middleware in the
headless cloud command.

### Higress demo configuration

Create a private route named `edge-usage` for
`POST /internal/edge-usage`. Its backend only needs to return a 2xx response;
the example sink can run on the cloud management network:

```bash
uvicorn usage_sink:app \
  --app-dir /vllm-workspace/vllm-ascend/examples/edge_cloud \
  --host 0.0.0.0 --port 18080
```

Point the Higress route at this service. The sink receives only the accounting
metadata shown above and does not store it. It must return `200` with a JSON
body. A `204 No Content` response does not trigger the AI Statistics response
body callback in Higress v2.2.3, so token metrics would not be finalized.

Configure the Higress Key Auth plugin globally with a unique credential per
edge:

```yaml
global_auth: false
consumers:
- credential: EDGE_0_SECRET
  name: edge-0
- credential: EDGE_1_SECRET
  name: edge-1
keys:
- x-api-key
in_query: false
in_header: true
```

On the `edge-usage` route, configure Key Auth authorization separately:

```yaml
allow:
- edge-0
- edge-1
```

The complete Key Auth and AI Statistics resources are available in
`examples/edge_cloud/higress_usage_plugins.yaml`. Replace both placeholder
credentials before applying it.

Key Auth adds `x-mse-consumer`, giving Higress a trusted per-edge dimension
instead of trusting the caller-supplied `x-ai-edge-id` alone. Apply the
AI Statistics plugin to the same route with these attributes:

```yaml
attributes:
- key: consumer
  value_source: request_header
  value: x-mse-consumer
  apply_to_log: true
- key: model
  value_source: request_body
  value: model
  apply_to_log: true
- key: input_token
  value_source: request_body
  value: usage.prompt_tokens
  apply_to_log: true
- key: output_token
  value_source: request_body
  value: usage.completion_tokens
  apply_to_log: true
- key: total_token
  value_source: request_body
  value: usage.total_tokens
  apply_to_log: true
- key: edge_inference_duration_ms
  value_source: request_body
  value: duration_ms
  apply_to_log: true
- key: event_id
  value_source: request_body
  value: event_id
  apply_to_log: true
- key: inference_status_code
  value_source: request_body
  value: status_code
  apply_to_log: true
```

Add the AI Statistics filter state to the Higress access-log format if it is
not already present:

```yaml
'{"ai_log":"%FILTER_STATE(wasm.ai_log:PLAIN)%"}'
```

The special `model`, `input_token`, and `output_token` attributes feed the AI
Statistics token counters. Key Auth supplies their `ai_consumer` dimension.
For example, Prometheus can compare input usage by edge with:

```promql
sum by (ai_consumer) (
  increase(route_upstream_model_consumer_metric_input_token{ai_route="edge-usage"}[5m])
)
```

Use the equivalent `output_token` metric for generated tokens. The gateway's
built-in service-duration metric measures the short reporting request. Use
the `edge_inference_duration_ms` access-log attribute for actual inference
latency.

Do not set `disable_openai_usage: true` on this AI Statistics rule. In Higress
v2.2.3 that option also disables metric emission for the custom numeric token
attributes.

### Higress web dashboard

The Docker all-in-one image without the `-o11y` suffix does not contain
Prometheus, Loki, or Grafana. For a local demo, run the official O11Y image and
enable the suite:

```bash
docker run -d --rm --name higress-ai \
  -e O11Y=on \
  -v /path/to/higress-data:/data \
  -p 8001:8001 -p 8080:8080 -p 8443:8443 \
  higress-registry.cn-hangzhou.cr.aliyuncs.com/higress/all-in-one:latest-o11y
```

For a Kubernetes deployment, enable it with
`--set global.o11y.enabled=true`. Log in to Higress Console, open **AI
Dashboard**, and use the built-in **Consumer Usage** table. It groups input,
output, and total token counters by `ai_consumer`; with the Key Auth
configuration above those values are `edge-0`, `edge-1`, and so on.

The Token Per Second panels use Prometheus `irate` and remain at **No data**
until usage changes across at least two scrape samples. With the all-in-one
default 15-second scrape interval, allow about 30 seconds after the first real
reports, then refresh the dashboard.

Use HTTPS or a private trusted network between edge and Higress. Do not add
prompt text, generated text, token IDs, embeddings, user identifiers, or
prompt hashes to the reporting headers.

## Prototype checks

1. Send one deterministic request to each edge separately and compare tokens
   with the existing one-edge baseline.
2. Send requests to both edges concurrently and verify both complete.
3. Log `SchedulerOutput.edge_id` on the cloud and verify 0/1 dispatches.
4. Send the same prompt concurrently and compare every generated token with
   baseline, exercising KV namespace isolation.
5. Confirm cloud memory contains one TP=8 logical model replica, not one replica
   per edge.
6. Send one request through each edge and verify the Higress access log records
   different `consumer` values and the expected token totals.
7. Capture the edge-to-Higress request and verify that the JSON contains only
   accounting metadata and no inference request or response content.
8. Open Higress Console's **AI Dashboard** and verify the **Consumer Usage**
   table contains separate rows for every edge.
