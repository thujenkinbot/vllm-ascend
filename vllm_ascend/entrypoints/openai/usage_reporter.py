"""Privacy-preserving edge usage reporting for the shared-cloud MVP.

The middleware observes only vLLM's final request metadata. It never reads the
inference request or response body. Reports contain a small JSON accounting
event and are sent from a bounded background queue so Higress availability
cannot delay or fail an inference response.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, MutableMapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import aiohttp

from vllm_ascend import envs

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 1024
DEFAULT_REQUEST_TIMEOUT_SECONDS = 2.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 0.25
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 3.0
RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 425, 429})

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]
SendEvent = Callable[["UsageReportEvent"], Awaitable[None]]


def _new_event_id() -> str:
    return uuid4().hex


def _validate_header_value(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must not be empty")
    if any(char in value for char in ("\r", "\n", "\x00")):
        raise ValueError(f"{name} must be a valid HTTP header value")


@dataclass(frozen=True)
class UsageReportConfig:
    """Configuration for one edge reporter.

    Runtime tuning fields are constructor options for tests and future
    integrations. The MVP intentionally exposes only identity and destination
    through environment variables.
    """

    url: str
    api_key: str = field(repr=False)
    edge_id: str
    model: str
    queue_size: int = DEFAULT_QUEUE_SIZE
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        parsed_url = urlparse(self.url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise ValueError("usage report URL must use http or https")
        _validate_header_value("usage report API key", self.api_key)
        _validate_header_value("usage report edge ID header", self.edge_id)
        _validate_header_value("usage report model header", self.model)
        if self.queue_size <= 0:
            raise ValueError("usage report queue size must be greater than zero")
        if self.request_timeout_seconds <= 0:
            raise ValueError("usage report request timeout must be greater than zero")
        if self.max_retries < 0:
            raise ValueError("usage report max retries must not be negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("usage report retry backoff must not be negative")
        if self.shutdown_timeout_seconds <= 0:
            raise ValueError("usage report shutdown timeout must be greater than zero")

    @classmethod
    def from_env(cls) -> "UsageReportConfig":
        edge_id = envs.VLLM_ASCEND_USAGE_REPORT_EDGE_ID
        if not edge_id:
            edge_id = f"edge-{envs.VLLM_ASCEND_EDGE_CLOUD_EDGE_IDX}"
        return cls(
            url=envs.VLLM_ASCEND_USAGE_REPORT_URL,
            api_key=envs.VLLM_ASCEND_USAGE_REPORT_API_KEY,
            edge_id=edge_id,
            model=envs.VLLM_ASCEND_USAGE_REPORT_MODEL,
        )


@dataclass(frozen=True)
class UsageReportEvent:
    """Content-free accounting event emitted after a completed response."""

    event_id: str
    edge_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: int
    status_code: int

    def __post_init__(self) -> None:
        _validate_header_value("usage report event ID header", self.event_id)
        _validate_header_value("usage report edge ID header", self.edge_id)
        _validate_header_value("usage report model header", self.model)
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "duration_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"usage report {name} must not be negative")
        if not 100 <= self.status_code <= 599:
            raise ValueError("usage report status code must be a valid HTTP status")

    def to_headers(self, *, api_key: str) -> dict[str, str]:
        """Serialize authentication and correlation metadata headers."""
        _validate_header_value("usage report API key", api_key)
        return {
            "x-api-key": api_key,
            "x-ai-edge-id": self.edge_id,
            "x-ai-event-id": self.event_id,
        }

    def to_payload(self) -> dict[str, object]:
        """Serialize content-free accounting metadata for AI Statistics.

        Token counts must be JSON numbers. Higress AI Statistics v2.2.3 does
        not convert numeric strings extracted from HTTP headers into counters.
        """
        return {
            "event_id": self.event_id,
            "edge_id": self.edge_id,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "duration_ms": self.duration_ms,
            "status_code": self.status_code,
        }


class _PermanentReportError(RuntimeError):
    pass


class UsageReporter:
    """Deliver usage events from a bounded, best-effort background queue."""

    def __init__(
        self,
        config: UsageReportConfig,
        *,
        send_event: SendEvent | None = None,
    ) -> None:
        self.config = config
        self._queue: asyncio.Queue[UsageReportEvent | None] = asyncio.Queue(maxsize=config.queue_size)
        self._send_event = send_event or self._send_http
        self._worker_task: asyncio.Task[None] | None = None
        self._session: aiohttp.ClientSession | None = None
        self._closed = False

    def start(self) -> None:
        if self._closed or self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._run(), name="vllm-ascend-usage-reporter")

    def enqueue(self, event: UsageReportEvent) -> bool:
        """Queue an event without waiting for any network operation."""
        if self._closed:
            return False
        self.start()
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "Usage report queue is full; dropping event_id=%s edge_id=%s",
                event.event_id,
                event.edge_id,
            )
            return False
        return True

    async def flush(self) -> None:
        """Wait until all events currently in the queue have been handled."""
        await self._queue.join()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        worker_task = self._worker_task
        if worker_task is not None:
            try:
                await asyncio.wait_for(
                    self._queue.join(),
                    timeout=self.config.shutdown_timeout_seconds,
                )
            except TimeoutError:
                logger.warning(
                    "Timed out draining usage report queue during shutdown; discarding %d event(s)",
                    self._queue.qsize(),
                )
                worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await worker_task
                self._discard_queued_events()
            else:
                self._queue.put_nowait(None)
                await worker_task
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _discard_queued_events(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                self._queue.task_done()

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                if event is None:
                    return
                await self._deliver(event)
            finally:
                self._queue.task_done()

    async def _deliver(self, event: UsageReportEvent) -> None:
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                await self._send_event(event)
                return
            except asyncio.CancelledError:
                raise
            except _PermanentReportError as error:
                logger.warning(
                    "Usage report rejected; dropping event_id=%s edge_id=%s: %s",
                    event.event_id,
                    event.edge_id,
                    error,
                )
                return
            except Exception as error:
                if attempt + 1 == attempts:
                    logger.warning(
                        "Usage report failed after %d attempt(s); dropping event_id=%s edge_id=%s: %s",
                        attempts,
                        event.event_id,
                        event.edge_id,
                        error,
                    )
                    return
                backoff = self.config.retry_backoff_seconds * (2**attempt)
                if backoff:
                    await asyncio.sleep(backoff)

    async def _send_http(self, event: UsageReportEvent) -> None:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

        async with self._session.post(
            self.config.url,
            headers=event.to_headers(api_key=self.config.api_key),
            json=event.to_payload(),
            allow_redirects=False,
        ) as response:
            if 200 <= response.status < 300:
                return
            message = f"Higress returned HTTP {response.status}"
            if response.status < 500 and response.status not in RETRYABLE_HTTP_STATUS_CODES:
                raise _PermanentReportError(message)
            raise RuntimeError(message)


class UsageReporterMiddleware:
    """ASGI middleware that reports vLLM final usage after the last body chunk."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        reporter: UsageReporter | None = None,
        clock: Callable[[], float] = time.monotonic,
        event_id_factory: Callable[[], str] = _new_event_id,
    ) -> None:
        self.app = app
        self.reporter = reporter or UsageReporter(UsageReportConfig.from_env())
        self._clock = clock
        self._event_id_factory = event_id_factory
        logger.info(
            "Edge usage reporter enabled: edge_id=%s model=%s",
            self.reporter.config.edge_id,
            self.reporter.config.model,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await self._handle_lifespan(scope, receive, send)
            return
        if scope_type != "http":
            await self.app(scope, receive, send)
            return

        started_at = self._clock()
        status_code = 500
        response_finished = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_finished, status_code
            message_type = message.get("type")
            if message_type == "http.response.start":
                status_code = int(message.get("status", 500))

            await send(message)

            if message_type == "http.response.body" and not message.get("more_body", False) and not response_finished:
                response_finished = True
                self._enqueue_final_usage(
                    scope,
                    status_code=status_code,
                    duration_ms=max(0, round((self._clock() - started_at) * 1000)),
                )

        await self.app(scope, receive, send_wrapper)

    def _enqueue_final_usage(
        self,
        scope: Scope,
        *,
        status_code: int,
        duration_ms: int,
    ) -> None:
        try:
            state = scope.get("state") or {}
            request_metadata = state.get("request_metadata")
            if request_metadata is None:
                return
            usage = getattr(request_metadata, "final_usage_info", None)
            if usage is None:
                return
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            event = UsageReportEvent(
                event_id=self._event_id_factory(),
                edge_id=self.reporter.config.edge_id,
                model=self.reporter.config.model,
                prompt_tokens=int(usage.prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(usage.total_tokens),
                duration_ms=duration_ms,
                status_code=status_code,
            )
            self.reporter.enqueue(event)
        except Exception:
            # Reporting must never turn a successful inference into an error.
            logger.exception("Failed to enqueue completed vLLM request usage")

    async def _handle_lifespan(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        async def receive_wrapper() -> Message:
            message = await receive()
            if message.get("type") == "lifespan.startup":
                self.reporter.start()
            elif message.get("type") == "lifespan.shutdown":
                await self.reporter.close()
            return message

        await self.app(scope, receive_wrapper, send)
