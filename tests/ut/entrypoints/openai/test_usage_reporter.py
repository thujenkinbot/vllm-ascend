import asyncio
from collections.abc import MutableMapping
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm_ascend.entrypoints.openai.usage_reporter import (
    UsageReportConfig,
    UsageReporter,
    UsageReporterMiddleware,
    UsageReportEvent,
)


def _config() -> UsageReportConfig:
    return UsageReportConfig(
        url="https://higress.example/internal/edge-usage",
        api_key="edge-secret",
        edge_id="edge-0",
        model="qwen3.5-27b",
        queue_size=2,
        request_timeout_seconds=0.1,
        max_retries=0,
        shutdown_timeout_seconds=0.1,
    )


def _event() -> UsageReportEvent:
    return UsageReportEvent(
        event_id="usage-event-1",
        edge_id="edge-0",
        model="qwen3.5-27b",
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
        duration_ms=123,
        status_code=200,
    )


def test_config_from_env_derives_edge_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "VLLM_ASCEND_USAGE_REPORT_URL",
        "https://higress.example/internal/edge-usage",
    )
    monkeypatch.setenv("VLLM_ASCEND_USAGE_REPORT_API_KEY", "secret")
    monkeypatch.setenv("VLLM_ASCEND_USAGE_REPORT_MODEL", "qwen3.5-27b")
    monkeypatch.delenv("VLLM_ASCEND_USAGE_REPORT_EDGE_ID", raising=False)
    monkeypatch.setenv("VLLM_ASCEND_EDGE_CLOUD_EDGE_IDX", "1")

    config = UsageReportConfig.from_env()

    assert config.edge_id == "edge-1"
    assert "secret" not in repr(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("url", "file:///tmp/usage", "http or https"),
        ("api_key", "", "API key"),
        ("edge_id", "edge-0\nforwarded", "header"),
        ("queue_size", 0, "queue size"),
    ],
)
def test_config_rejects_invalid_values(field: str, value: object, message: str) -> None:
    values = _config().__dict__ | {field: value}

    with pytest.raises(ValueError, match=message):
        UsageReportConfig(**values)


def test_event_serialization_contains_accounting_metadata_only() -> None:
    headers = _event().to_headers(api_key="edge-secret")

    assert headers == {
        "x-api-key": "edge-secret",
        "x-ai-edge-id": "edge-0",
        "x-ai-event-id": "usage-event-1",
    }
    assert _event().to_payload() == {
        "event_id": "usage-event-1",
        "edge_id": "edge-0",
        "model": "qwen3.5-27b",
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        },
        "duration_ms": 123,
        "status_code": 200,
    }


@pytest.mark.asyncio
async def test_reporter_delivers_queued_event() -> None:
    delivered: list[UsageReportEvent] = []

    async def send_event(event: UsageReportEvent) -> None:
        delivered.append(event)

    reporter = UsageReporter(_config(), send_event=send_event)
    try:
        assert reporter.enqueue(_event())
        await reporter.flush()
    finally:
        await reporter.close()

    assert delivered == [_event()]


@pytest.mark.asyncio
async def test_http_report_has_json_metadata_without_inference_content() -> None:
    captured: dict[str, Any] = {}

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            pass

        async def read(self) -> bytes:
            return b""

    class Session:
        def post(self, url: str, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return Response()

        async def close(self) -> None:
            pass

    reporter = UsageReporter(_config())
    reporter._session = cast(Any, Session())
    try:
        assert reporter.enqueue(_event())
        await reporter.flush()
    finally:
        await reporter.close()

    assert captured["url"] == "https://higress.example/internal/edge-usage"
    assert captured["json"] == _event().to_payload()
    assert not captured["allow_redirects"]
    assert captured["headers"]["x-ai-edge-id"] == "edge-0"
    assert captured["headers"]["x-api-key"] == "edge-secret"
    assert "prompt" not in captured["json"]
    assert "messages" not in captured["json"]


@pytest.mark.asyncio
async def test_reporter_retries_without_blocking_enqueue() -> None:
    attempts = 0

    async def send_event(event: UsageReportEvent) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary failure")

    config = UsageReportConfig(**(_config().__dict__ | {"max_retries": 1, "retry_backoff_seconds": 0}))
    reporter = UsageReporter(config, send_event=send_event)
    try:
        assert reporter.enqueue(_event())
        await reporter.flush()
    finally:
        await reporter.close()

    assert attempts == 2


@pytest.mark.asyncio
async def test_reporter_drops_event_when_queue_is_full() -> None:
    gate = asyncio.Event()

    async def send_event(event: UsageReportEvent) -> None:
        await gate.wait()

    config = UsageReportConfig(**(_config().__dict__ | {"queue_size": 1}))
    reporter = UsageReporter(config, send_event=send_event)
    try:
        assert reporter.enqueue(_event())
        await asyncio.sleep(0)
        assert reporter.enqueue(_event())
        assert not reporter.enqueue(_event())
        gate.set()
        await reporter.flush()
    finally:
        await reporter.close()


@pytest.mark.asyncio
async def test_middleware_reports_usage_after_response_completion() -> None:
    delivered: list[UsageReportEvent] = []

    async def send_event(event: UsageReportEvent) -> None:
        delivered.append(event)

    async def app(scope, receive, send) -> None:
        scope["state"]["request_metadata"] = SimpleNamespace(
            request_id="chatcmpl-request-1",
            final_usage_info=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
            ),
        )
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"chunk", "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    times = iter((10.0, 10.123))
    reporter = UsageReporter(_config(), send_event=send_event)
    middleware = UsageReporterMiddleware(
        app,
        reporter=reporter,
        clock=lambda: next(times),
        event_id_factory=lambda: "usage-event-1",
    )
    sent: list[MutableMapping[str, Any]] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "state": {},
    }
    try:
        await middleware(scope, receive, send)
        await reporter.flush()
    finally:
        await reporter.close()

    assert len(sent) == 3
    assert delivered == [_event()]


@pytest.mark.asyncio
async def test_middleware_ignores_requests_without_final_usage() -> None:
    delivered: list[UsageReportEvent] = []

    async def send_event(event: UsageReportEvent) -> None:
        delivered.append(event)

    async def app(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    reporter = UsageReporter(_config(), send_event=send_event)
    middleware = UsageReporterMiddleware(app, reporter=reporter)

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: MutableMapping[str, Any]) -> None:
        pass

    try:
        await middleware(
            {"type": "http", "method": "GET", "path": "/health", "state": {}},
            receive,
            send,
        )
        await reporter.flush()
    finally:
        await reporter.close()

    assert delivered == []
