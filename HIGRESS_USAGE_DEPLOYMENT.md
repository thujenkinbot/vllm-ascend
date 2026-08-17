# Higress 边侧用量统计独立部署验证

本文说明如何在一台已经运行 Higress `latest-o11y` 容器的服务器上，
不依赖真实 vLLM 请求，独立验证边侧 Token 用量统计链路。

## 1. 验证目标

验证链路如下：

```text
模拟 Usage 请求
  -> Higress :8080/internal/edge-usage
  -> Key Auth：将 API Key 映射为 edge-0、edge-1
  -> AI Statistics：从 JSON 中提取 Token 数
  -> Usage Sink：返回 200 {}
  -> Envoy Prometheus 指标
  -> Prometheus
  -> Grafana AI Gateway Dashboard
```

这条链路只传输用量元数据，不传输原始提示词、生成内容、Embedding
或用户标识。

## 2. 前提条件

- Higress 使用 `all-in-one:latest-o11y` 镜像。
- 启动容器时设置了 `O11Y=on`。
- 宿主机可以访问 Higress Console 和 Gateway 端口。
- 示例假设：
    - 容器名为 `higress-ai`。
    - Console 映射到宿主机 `8001` 端口。
    - HTTP Gateway 映射到宿主机 `8080` 端口。
    - HTTPS Gateway 映射到宿主机 `8443` 端口。

### 2.1 创建 Higress O11Y 容器

首次部署时，先创建持久化数据目录：

```bash
sudo mkdir -p /opt/higress/data
```

拉取镜像：

```bash
docker pull \
  higress-registry.cn-hangzhou.cr.aliyuncs.com/higress/all-in-one:latest-o11y
```

创建并启动容器：

```bash
docker run -d \
  --name higress-ai \
  --restart unless-stopped \
  -e O11Y=on \
  -p 8001:8001 \
  -p 8080:8080 \
  -p 8443:8443 \
  -v /opt/higress/data:/data \
  higress-registry.cn-hangzhou.cr.aliyuncs.com/higress/all-in-one:latest-o11y
```

端口用途：

| 宿主机端口 | 容器端口 | 用途 |
| --- | --- | --- |
| `8001` | `8001` | Higress Console |
| `8080` | `8080` | HTTP Gateway |
| `8443` | `8443` | HTTPS Gateway |

Prometheus 和 Grafana 由 Console 代理访问，不需要额外暴露 `9090` 和
`3000` 端口。这样也可以避免将内部监控接口直接暴露到服务器外部。

查看启动日志：

```bash
docker logs -f --tail 200 higress-ai
```

确认容器状态：

```bash
docker ps --filter name=higress-ai
```

`latest-o11y` 是浮动标签。完成验证并确定版本后，正式部署建议将镜像固定为
具体 digest，避免后续重新拉取时镜像内容发生变化。

### 2.2 设置验证变量

设置后续命令使用的变量：

```bash
export HIGRESS_CONTAINER=higress-ai
export HIGRESS_HOST=127.0.0.1
export HIGRESS_HTTP_PORT=8080
```

如果实际容器名、服务器地址或端口不同，请替换上述变量。

## 3. 检查 O11Y 组件

检查容器是否配置了 O11Y：

```bash
docker inspect "$HIGRESS_CONTAINER" \
  --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^O11Y=on$'
```

检查 Prometheus 和 Grafana：

```bash
docker exec "$HIGRESS_CONTAINER" sh -lc '
curl -fsS http://127.0.0.1:9090/prometheus/-/ready
curl -fsS http://127.0.0.1:3000/grafana/api/health
'
```

预期结果：

- Prometheus 返回 ready。
- Grafana 返回包含 `database: ok` 的 JSON。

只有镜像名带 `-o11y` 仍不够，启动容器时建议明确传入：

```bash
-e O11Y=on
```

如果检查失败，应先修复 O11Y 启动配置，再继续下面的步骤。

## 4. 启动临时 Usage Sink

AI Statistics 需要请求经过一个真实后端，并在响应阶段完成指标记录。
验证阶段可以在 Higress 容器内启动一个无依赖的临时 HTTP 服务。

在宿主机创建 `/tmp/edge-usage-sink.py`：

```python
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        if self.path != "/internal/edge-usage":
            self.send_error(404)
            return

        content_length = int(self.headers.get("content-length", "0"))
        self.rfile.read(content_length)

        body = b"{}"
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


ThreadingHTTPServer(("0.0.0.0", 18080), Handler).serve_forever()
```

复制到容器并启动：

```bash
docker cp /tmp/edge-usage-sink.py \
  "$HIGRESS_CONTAINER":/data/edge-usage-sink.py

docker exec -d "$HIGRESS_CONTAINER" \
  python3 /data/edge-usage-sink.py
```

检查服务：

```bash
docker exec "$HIGRESS_CONTAINER" \
  curl -i http://127.0.0.1:18080/health
```

预期返回 `204 No Content`。

注意：

- 健康检查可以返回 `204`。
- Usage POST 必须返回 `200` 和 JSON 响应体 `{}`。
- 如果 Usage POST 返回 `204`，AI Statistics 不会完成响应阶段的 Token 指标记录。
- 使用 `docker exec -d` 启动的进程不会随容器自动重启，仅适合 MVP 验证。

## 5. 创建服务和路由

登录 Higress Console：

```text
http://HIGRESS_HOST:8001
```

默认账号密码通常为：

```text
admin / admin
```

### 5.1 创建静态服务

在“服务来源”中创建服务：

| 配置项 | 值 |
| --- | --- |
| 服务名 | `edge-usage` |
| 服务地址 | `127.0.0.1:18080` |
| 服务端口 | `80`，Console 固定的逻辑端口 |
| 协议 | HTTP |

这里有两层端口：

- 服务地址中的 `18080` 是 Sink 实际监听的端点端口。
- “服务端口”`80` 是 Higress 为固定地址服务发布的逻辑端口。

路由选择逻辑端口 `80` 后，Higress 会根据固定地址配置，把流量实际发送到
`127.0.0.1:18080`。这不要求宿主机或 Sink 监听 `80`，也不会额外开放
宿主机的 `80` 端口。

### 5.2 创建路由

在“路由配置”中创建路由：

| 配置项 | 值 |
| --- | --- |
| 路由名 | `edge-usage` |
| Path | `/internal/edge-usage` |
| Path 匹配类型 | Prefix |
| 后端服务 | `edge-usage` |
| 后端端口 | `80`，即固定地址服务的逻辑端口 |

路由名必须是 `edge-usage`。插件匹配规则和后续 Prometheus 查询都使用这个名字。

## 6. 配置 Key Auth

当前 `latest-o11y` Console 已提供“消费者管理”和路由“认证配置”。应优先
使用这两个入口，不要在“插件配置”或路由“策略”中重复手工维护 Key Auth。

### 6.1 在消费者管理中创建调用方

进入“消费者管理”，分别创建两个消费者：

| 配置项 | edge-0 | edge-1 |
| --- | --- | --- |
| 消费者名称 | `edge-0` | `edge-1` |
| 认证方式 | Key Auth | Key Auth |
| 凭证来源 | HTTP Header | HTTP Header |
| Header 名称 | `x-api-key` | `x-api-key` |
| 访问凭证 | 独立随机密钥 | 另一个独立随机密钥 |

不同消费者的访问凭证不能相同。可以使用下列命令生成凭证：

```bash
openssl rand -hex 32
```

消费者管理会自动同步全局 Key Auth 插件中的 Consumer 和凭证配置，不需要
再进入“插件配置 -> Key 认证”重复添加 `consumers`。

### 6.2 在路由编辑表单中开启认证

进入“路由配置”，对 `edge-usage` 执行“编辑”，不要点击“策略”：

1. 找到“认证配置”或“开启消费者认证”。
2. 打开认证开关。
3. 认证方式选择 Key Auth。当前版本中该选项可能是只读的。
4. “允许的消费者”同时选择 `edge-0`、`edge-1`。
5. 保存路由。

Console 会自动生成等价的路由级 Key Auth 配置：

```yaml
allow:
  - edge-0
  - edge-1
```

底层全局配置大致等价于：

```json
{
  "global_auth": false,
  "consumers": [
    {
      "name": "edge-0",
      "credential": "REPLACE_EDGE_0_KEY"
    },
    {
      "name": "edge-1",
      "credential": "REPLACE_EDGE_1_KEY"
    }
  ],
  "keys": ["x-api-key"],
  "in_header": true,
  "in_query": false
}
```

底层路由配置大致等价于：

```json
{
  "allow": ["edge-0", "edge-1"]
}
```

不要在“路由配置 -> edge-usage -> 策略 -> Key 认证”中再次填写
`global_auth`、`consumers` 等实例级字段。部分 Console 版本会在该入口错误
复用实例级表单；消费者管理和路由编辑表单才是当前版本的高层管理入口。

如果之前已经在“插件配置”中手工开启过 Key Auth，应将该手工实例关闭，避免
它与消费者管理自动维护的 `key-auth.internal` 重复。关闭后，旧的
WasmPlugin 文件可能仍然存在，但 `defaultConfigDisable: true` 表示它已禁用；
不要关闭或删除 `key-auth.internal`。

如果所用旧版 Console 没有“消费者管理”或路由“认证配置”，再使用项目提供
的 `higress_usage_plugins.yaml` 通过 WasmPlugin 配置 Key Auth。

认证成功后，Key Auth 会生成可信请求头 `x-mse-consumer`。AI Statistics
使用这个请求头作为 `ai_consumer` 指标维度，而不是信任请求体中的
`edge_id`。

生产环境必须替换示例 API Key，并确保不同边侧使用不同凭证。

## 7. 配置 AI Statistics

通过 Console 安装或启用当前镜像提供的 AI Statistics，并只在
`edge-usage` 路由上生效。本次验证的 `latest-o11y` 镜像内置版本为
`ai-statistics:2.0.1`；项目中的 WasmPlugin 示例使用独立 OCI URL，版本以
示例文件为准，不依赖容器内 `8002` 端口的插件缓存。

网页表单中的 `Key` 和 `Value` 含义不同：

- `Key` 是 AI Statistics 输出到日志或指标上下文中的属性名称。
- `Value` 是从 Usage Reporter 请求体中读取数据的 JSON 字段路径。

应严格按照下表配置：

| Key | Value Source | Value |
| --- | --- | --- |
| `consumer` | Request Header | `x-mse-consumer` |
| `model` | Request Body | `model` |
| `input_token` | Request Body | `usage.prompt_tokens` |
| `output_token` | Request Body | `usage.completion_tokens` |
| `total_token` | Request Body | `usage.total_tokens` |
| `edge_inference_duration_ms` | Request Body | `duration_ms` |
| `event_id` | Request Body | `event_id` |
| `inference_status_code` | Request Body | `status_code` |

特别注意，下面两组不能把 Key 原样复制到 Value：

```text
Key=edge_inference_duration_ms  Value=duration_ms
Key=inference_status_code       Value=status_code
```

如果错误配置为 `Value=edge_inference_duration_ms` 或
`Value=inference_status_code`，Token 指标仍然能够产生，但真实推理耗时和
状态码属性会为空。

路由级配置：

```json
{
  "attributes": [
    {
      "key": "consumer",
      "value_source": "request_header",
      "value": "x-mse-consumer",
      "apply_to_log": true
    },
    {
      "key": "model",
      "value_source": "request_body",
      "value": "model",
      "apply_to_log": true
    },
    {
      "key": "input_token",
      "value_source": "request_body",
      "value": "usage.prompt_tokens",
      "apply_to_log": true
    },
    {
      "key": "output_token",
      "value_source": "request_body",
      "value": "usage.completion_tokens",
      "apply_to_log": true
    },
    {
      "key": "total_token",
      "value_source": "request_body",
      "value": "usage.total_tokens",
      "apply_to_log": true
    },
    {
      "key": "edge_inference_duration_ms",
      "value_source": "request_body",
      "value": "duration_ms",
      "apply_to_log": true
    },
    {
      "key": "event_id",
      "value_source": "request_body",
      "value": "event_id",
      "apply_to_log": true
    },
    {
      "key": "inference_status_code",
      "value_source": "request_body",
      "value": "status_code",
      "apply_to_log": true
    }
  ]
}
```

不要设置：

```json
{
  "disable_openai_usage": true
}
```

在当前验证版本中，该配置会阻止自定义 Token 指标生成。

项目内已经提供完整 WasmPlugin 模板：

```text
vllm-ascend/examples/edge_cloud/higress_usage_plugins.yaml
```

应用模板前必须替换其中的两个 `CHANGE_ME_*` 凭证。

## 8. 检查 Wasm 文件和插件加载状态

Console 中显示“已启用”只代表配置已经保存，不能证明 Envoy 成功加载了
Wasm。Key Auth 出现“任意 Key 或不带 Key 都能访问”时，应按本节检查实际
下发的 URL、插件缓存和 Envoy 加载状态。

### 8.1 检查实际引用版本和容器缓存

消费者管理和路由认证自动生成的 Key Auth 配置位于：

```text
/data/wasmplugins/key-auth.internal.yaml
```

检查它实际引用的 Wasm URL，以及镜像中实际存在的版本：

```bash
docker exec "$HIGRESS_CONTAINER" sh -lc '
grep -E "wasm-plugin-version:|url:" \
  /data/wasmplugins/key-auth.internal.yaml

find /usr/share/nginx/html/plugins/key-auth \
  -maxdepth 2 -name plugin.wasm -print
'
```

继续直接测试插件 URL：

```bash
docker exec "$HIGRESS_CONTAINER" sh -lc '
curl -sS -o /dev/null -w "key-auth-1.0.0=%{http_code}\n" \
  http://127.0.0.1:8002/plugins/key-auth/1.0.0/plugin.wasm

curl -sS -o /dev/null -w "key-auth-2.0.0=%{http_code}\n" \
  http://127.0.0.1:8002/plugins/key-auth/2.0.0/plugin.wasm

curl -sS -o /dev/null -w "ai-statistics-2.0.1=%{http_code}\n" \
  http://127.0.0.1:8002/plugins/ai-statistics/2.0.1/plugin.wasm
'
```

Envoy 引用的 URL 必须返回 `200`。如果配置引用 `1.0.0`，但只有 `2.0.0`
返回 `200`，就发生了插件版本错配。

### 8.2 检查 xDS 和 Envoy 加载状态

查看 Higress Controller 向 Envoy 下发扩展配置时的错误：

```bash
docker exec "$HIGRESS_CONTAINER" sh -lc '
curl -sS http://127.0.0.1:15014/debug/syncz
'
```

如果输出的 `lastError` 包含以下内容，说明插件配置存在，但 Wasm 文件加载
失败：

```text
cannot fetch Wasm module ... status code 404
```

再检查 Key Auth 的 Envoy 运行时计数：

```bash
docker exec "$HIGRESS_CONTAINER" sh -lc '
curl -sS \
  "http://127.0.0.1:15000/stats?filter=extension_config_discovery.http_filter.extensions.istio.io/wasmplugin/higress-system.key-auth.internal"
'
```

成功加载后应满足：

- `version_text` 非空。
- `last_update_success` 为 `1`。
- `update_success` 大于 `0`。

`update_failure` 是累计历史计数。修复后只要最后一次更新成功，旧的失败计数
仍然存在是正常现象。

### 8.3 本次 latest-o11y 的 Key Auth 版本错配

本次实际验证的 `latest-o11y` 镜像存在以下组合：

- 容器中的 Higress Console 构建版本为 `2.2.4`，它内嵌的 Key Auth
  插件元数据版本仍为 `1.0.0`。
- Higress Console 生成的 `key-auth.internal` 引用
  `/plugins/key-auth/1.0.0/plugin.wasm`。
- 镜像中的 Nginx 插件缓存只提供
  `/plugins/key-auth/2.0.0/plugin.wasm`。
- `1.0.0` URL 返回 `404`，`2.0.0` URL 返回 `200`。
- Key Auth 的失败策略为 `FAIL_OPEN`，因此 Wasm 加载失败时请求被放行。

这就是无 Key、错误 Key 和正确 Key 都返回 `200` 的直接原因。

此前在“插件配置”中手工开启 Key Auth 会额外创建一个重复插件实例，应当
关闭，但它只是干扰项，不是版本错配的根因。即使关闭手工实例，自动生成的
`key-auth.internal` 仍然引用不存在的 `1.0.0` 文件。因此，就本次容器中的
证据而言，核心问题是 `latest-o11y` 镜像中的 Console 元数据和插件缓存版本
没有对齐，不是 API Key 或路由配置填错导致的。

`latest-o11y` 是浮动标签，其他时间拉取到的镜像不一定存在相同问题，应以
上述 URL 和 Envoy 状态检查结果为准。

### 8.4 本地 MVP 兼容链接

本地验证可以让 Console 继续引用 `1.0.0` URL，但将该路径链接到镜像实际
提供的 `2.0.0` Wasm：

```bash
docker exec "$HIGRESS_CONTAINER" sh -lc '
mkdir -p /usr/share/nginx/html/plugins/key-auth/1.0.0
ln -sf ../2.0.0/plugin.wasm \
  /usr/share/nginx/html/plugins/key-auth/1.0.0/plugin.wasm
'
```

确认兼容 URL 已恢复：

```bash
docker exec "$HIGRESS_CONTAINER" curl -I \
  http://127.0.0.1:8002/plugins/key-auth/1.0.0/plugin.wasm
```

预期返回 `200`。然后在 Console 中编辑 `edge-usage` 路由：

1. 关闭请求认证并保存。
2. 再次编辑路由，开启请求认证。
3. 重新选择 `edge-0`、`edge-1` 并保存。

这会触发 `key-auth.internal` 重新下发。由于临时 Usage Sink 是通过
`docker exec -d` 手工启动的，建议使用路由开关触发重新下发，不要为此重启
容器。

该链接在同一个容器执行 `docker restart` 后仍然存在，但删除并重新创建容器
后会丢失。容器重建后需要重新创建链接；如果重启了容器，还需要重新启动
临时 Usage Sink。正式部署应使用组件版本匹配的固定镜像或修正镜像构建，
不要长期依赖容器内兼容链接。

### 8.5 认证回归检查

修复后必须至少检查三种请求：

| 请求 | 预期结果 |
| --- | --- |
| 不带 `x-api-key` | `401 Unauthorized` |
| 使用任意错误 Key | `401 Unauthorized` |
| 使用 `edge-0` 或 `edge-1` 的合法 Key | `200 {}` |

可以先用最小请求快速检查：

```bash
curl -o /dev/null -w 'no-key=%{http_code}\n' \
  -X POST "http://${HIGRESS_HOST}:${HIGRESS_HTTP_PORT}/internal/edge-usage" \
  -H 'content-type: application/json' -d '{}'

curl -o /dev/null -w 'wrong-key=%{http_code}\n' \
  -X POST "http://${HIGRESS_HOST}:${HIGRESS_HTTP_PORT}/internal/edge-usage" \
  -H 'content-type: application/json' \
  -H 'x-api-key: definitely-wrong' -d '{}'

curl -o /dev/null -w 'edge-0=%{http_code}\n' \
  -X POST "http://${HIGRESS_HOST}:${HIGRESS_HTTP_PORT}/internal/edge-usage" \
  -H 'content-type: application/json' \
  -H 'x-api-key: REPLACE_EDGE_0_KEY' -d '{}'
```

预期输出：

```text
no-key=401
wrong-key=401
edge-0=200
```

完整 Usage 请求继续使用第 9 节的示例。

### 8.6 检查 AI Statistics 加载

AI Statistics 也必须同时满足 Wasm URL 返回 `200` 且 Envoy 更新成功。当前
镜像可以检查内置的 `2.0.1`：

```bash
docker exec "$HIGRESS_CONTAINER" sh -lc '
curl -fsS \
  http://127.0.0.1:8002/plugins/ai-statistics/2.0.1/plugin.wasm \
  >/dev/null && echo "ai-statistics wasm: OK"

curl -sS http://127.0.0.1:15000/config_dump \
  | grep -q ai-statistics \
  && echo "ai-statistics ECDS: OK"
'
```

## 9. 发送模拟 Usage 请求

### 9.1 验证未认证请求

```bash
curl -i \
  "http://${HIGRESS_HOST}:${HIGRESS_HTTP_PORT}/internal/edge-usage" \
  -H 'content-type: application/json' \
  -d '{
    "event_id": "unauthorized-test",
    "edge_id": "edge-0",
    "model": "qwen3.5-27b",
    "usage": {
      "prompt_tokens": 101,
      "completion_tokens": 37,
      "total_tokens": 138
    },
    "duration_ms": 1234,
    "status_code": 200
  }'
```

预期返回 `401 Unauthorized`。

### 9.2 发送 edge-0 用量

```bash
curl -i \
  "http://${HIGRESS_HOST}:${HIGRESS_HTTP_PORT}/internal/edge-usage" \
  -H 'content-type: application/json' \
  -H 'x-api-key: REPLACE_EDGE_0_KEY' \
  -d '{
    "event_id": "edge-0-test-001",
    "edge_id": "edge-0",
    "model": "qwen3.5-27b",
    "usage": {
      "prompt_tokens": 101,
      "completion_tokens": 37,
      "total_tokens": 138
    },
    "duration_ms": 1234,
    "status_code": 200
  }'
```

### 9.3 发送 edge-1 用量

```bash
curl -i \
  "http://${HIGRESS_HOST}:${HIGRESS_HTTP_PORT}/internal/edge-usage" \
  -H 'content-type: application/json' \
  -H 'x-api-key: REPLACE_EDGE_1_KEY' \
  -d '{
    "event_id": "edge-1-test-001",
    "edge_id": "edge-1",
    "model": "qwen3.5-27b",
    "usage": {
      "prompt_tokens": 203,
      "completion_tokens": 41,
      "total_tokens": 244
    },
    "duration_ms": 1512,
    "status_code": 200
  }'
```

两个合法请求都应返回：

```http
HTTP/1.1 200 OK
content-type: application/json

{}
```

Token 字段必须是 JSON Number，不能使用字符串：

```json
{
  "prompt_tokens": 101
}
```

不要写成：

```json
{
  "prompt_tokens": "101"
}
```

建议等待 20 秒后再发送第二轮请求，使数据至少跨越两个 Prometheus
采集点。默认采集间隔约为 15 秒，Token/s 面板需要多个采集点才能计算速率。

### 9.4 持续并发模拟 edge-0 和 edge-1

项目提供了一个无第三方依赖的并发注入脚本：

```text
vllm-ascend/examples/edge_cloud/simulate_edge_usage.py
```

默认行为：

- edge-0 和 edge-1 各使用一个独立线程并发上报。
- 每个边独立地每隔 0.5 到 3 秒发送一条请求。
- 每条请求的输入 Token 在 1,000 到 1,000,000 之间。
- 每条请求的输出 Token 在 10 到 10,000 之间。
- `total_tokens` 等于输入与输出 Token 之和。
- 正式注入前先为两个边发送一条 0 Token 初始化事件，并等待 20 秒，避免
  Prometheus 将第一条真实 Usage 当作 Counter 的初始基线。
- 默认持续运行，按 `Ctrl-C` 停止。

直接运行，脚本会以隐藏输入方式依次询问两个 API Key：

```bash
python3 vllm-ascend/examples/edge_cloud/simulate_edge_usage.py \
  --url "http://${HIGRESS_HOST}:${HIGRESS_HTTP_PORT}/internal/edge-usage" \
  --model qwen3.5-27b
```

运行 5 分钟后自动停止：

```bash
python3 vllm-ascend/examples/edge_cloud/simulate_edge_usage.py \
  --url "http://${HIGRESS_HOST}:${HIGRESS_HTTP_PORT}/internal/edge-usage" \
  --model qwen3.5-27b \
  --duration-seconds 300
```

自动化环境也可以直接传入 Key，但需要注意命令行参数可能出现在进程列表和
Shell 历史中：

```bash
python3 vllm-ascend/examples/edge_cloud/simulate_edge_usage.py \
  --edge-0-key REPLACE_EDGE_0_KEY \
  --edge-1-key REPLACE_EDGE_1_KEY \
  --duration-seconds 300
```

不发送 HTTP 请求，只检查随机数据和并发节奏：

```bash
python3 vllm-ascend/examples/edge_cloud/simulate_edge_usage.py \
  --dry-run \
  --bootstrap-wait-seconds 0 \
  --duration-seconds 5 \
  --seed 42
```

如果目标 Consumer/Model 的 Counter 已经建立，可以使用
`--bootstrap-wait-seconds 0` 跳过初始化。每一行输出都是 JSON，程序退出时
还会打印两个边分别累计的请求数、失败数和 Token 数。

## 10. 验证 Prometheus 指标

查询累计输入 Token：

```bash
docker exec "$HIGRESS_CONTAINER" curl -sG \
  http://127.0.0.1:9090/prometheus/api/v1/query \
  --data-urlencode \
  'query=sum by (ai_consumer) (route_upstream_model_consumer_metric_input_token{ai_route="edge-usage"})'
```

查询累计输出 Token：

```bash
docker exec "$HIGRESS_CONTAINER" curl -sG \
  http://127.0.0.1:9090/prometheus/api/v1/query \
  --data-urlencode \
  'query=sum by (ai_consumer) (route_upstream_model_consumer_metric_output_token{ai_route="edge-usage"})'
```

查询累计总 Token：

```bash
docker exec "$HIGRESS_CONTAINER" curl -sG \
  http://127.0.0.1:9090/prometheus/api/v1/query \
  --data-urlencode \
  'query=sum by (ai_consumer) (route_upstream_model_consumer_metric_total_token{ai_route="edge-usage"})'
```

结果中应出现：

```text
ai_consumer="edge-0"
ai_consumer="edge-1"
```

如果 Gateway 原始指标存在，但 Prometheus 查询没有结果，通常是尚未到达
下一次采集时间。等待 15 到 30 秒后重试。

## 11. 验证 Grafana Dashboard

1. 登录 Higress Console。
2. 打开监控面板。
3. 进入 `Higress AI Gateway Dashboard`。
4. 时间范围选择 `Last 5 minutes`。
5. 检查 `Consumer Usage` 中是否出现 `edge-0` 和 `edge-1`。
6. 检查 Input Token、Output Token 和上方 Token/s 面板。

`Consumer Usage` 中可能显示小数，这是因为内置面板使用 Prometheus
`increase(counter[$__range])`。Prometheus 会对采集时间和查询窗口边界做
外推，因此整数 Counter 的区间增量也可能显示为小数。

小数的单位仍然是 Token，不是 Token/s。需要精确累计整数时，应使用第
10 节中的直接 Counter 查询。Token/s 只对应明确标注为 `Per Second` 的面板。

## 12. 接入边侧 vLLM Usage Reporter

模拟请求验证通过后，在每个边侧 vLLM 实例配置：

```bash
export VLLM_ASCEND_USAGE_REPORT_URL=\
"https://HIGRESS_HOST/internal/edge-usage"
export VLLM_ASCEND_USAGE_REPORT_API_KEY="EDGE_SPECIFIC_SECRET"
export VLLM_ASCEND_USAGE_REPORT_EDGE_ID="edge-0"
export VLLM_ASCEND_USAGE_REPORT_MODEL="qwen3.5-27b"
```

启动命令增加：

```bash
--middleware \
vllm_ascend.entrypoints.openai.usage_reporter.UsageReporterMiddleware
```

不同边侧必须使用不同的 API Key 和 `EDGE_ID`。Higress 应部署在云侧，边侧
只把用量元数据发往 Higress，不发送原始推理请求。

## 13. 验收标准

部署验证完成必须同时满足：

- 无 API Key 的请求返回 `401`。
- edge-0、edge-1 的合法 API Key 请求返回 `200 {}`。
- 两个 Wasm 文件都能成功加载。
- Envoy 配置中包含两个插件。
- Prometheus 中出现 `ai_consumer="edge-0"` 和 `edge-1`。
- Grafana `Consumer Usage` 面板出现两个 Consumer。
- 抓取上报请求后，确认请求体不包含提示词和生成内容。

## 14. MVP 限制与生产化注意事项

- all-in-one O11Y 镜像中的 Prometheus 默认只保留约 6 小时数据，适合部署
  验证，不适合长期计费。
- 临时 Usage Sink 不做持久化和 `event_id` 去重。
- 网络超时后的重试存在重复计数风险，当前方案不是严格的账单系统。
- 正式环境应把 Sink 独立部署并配置自动重启、持久化和幂等去重。
- 边侧到 Higress 应使用 HTTPS 或可信专网。
- Gateway 的 Usage 路由应限制来源地址，并使用每个边唯一的强随机 API Key。
- Prometheus、Grafana、Higress 内置 API Server 不应直接暴露到公网。

## 15. 常见问题定位

### 请求返回 404

- 检查路由名是否为 `edge-usage`。
- 检查 Path 是否为 `/internal/edge-usage`。
- 检查 Gateway 是否映射到宿主机 `8080`。

### 请求返回 503

- 检查 Sink 是否监听 `18080`。
- 检查静态服务地址是否为 `127.0.0.1:18080`。
- 检查路由选择的服务端口是否为固定逻辑端口 `80`。
- 在 Higress 容器中直接请求 Sink 的健康检查。

### 合法 API Key 仍然返回 401

- 检查请求头名称是否为 `x-api-key`。
- 检查 Consumer credential 是否与请求值完全一致。
- 检查路由级 `allow` 是否包含对应 Consumer。

### 任意 API Key 或不带 Key 都返回 200

- 确认没有在“插件配置”中重复开启手工 Key Auth。
- 检查 `key-auth.internal.yaml` 实际引用的 Wasm URL。
- 检查该 URL 是否返回 `200`，不能只看 Console 的“已启用”状态。
- 检查 Controller `syncz` 中是否出现 `cannot fetch Wasm module`。
- 检查 Envoy Key Auth 的 `last_update_success` 和 `version_text`。
- 如果遇到本次 `1.0.0` 与 `2.0.0` 错配，按第 8.4 节创建兼容链接并
  重新下发路由认证配置。

### 请求成功但没有 Token 指标

- 检查 Sink 是否返回 `200 {}`，而不是 `204`。
- 检查 Token 是否使用 JSON Number。
- 检查 `ai-statistics` 是否匹配 `edge-usage` 路由。
- 检查是否误设 `disable_openai_usage: true`。
- 检查两个 Wasm URL 是否返回 200。
- 检查 Envoy ECDS 配置中是否已经加载插件。
- 等待 15 到 30 秒后重新查询 Prometheus。

### Dashboard 有数据但显示小数

这是 `increase()` 的边界外推行为。单位仍然是 Token。精确累计值请直接查询
Prometheus Counter，不要通过四舍五入把区间估算值当作计费数据。
