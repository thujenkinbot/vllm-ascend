#!/usr/bin/env python3
"""Continuously inject synthetic edge usage events through Higress."""

from __future__ import annotations

import argparse
import getpass
import json
import random
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_URL = "http://127.0.0.1:8080/internal/edge-usage"
DEFAULT_MODEL = "qwen3.5-27b"
DEFAULT_MIN_PROMPT_TOKENS = 1_000
DEFAULT_MAX_PROMPT_TOKENS = 1_000_000
DEFAULT_MIN_COMPLETION_TOKENS = 10
DEFAULT_MAX_COMPLETION_TOKENS = 10_000
DEFAULT_MIN_DELAY_SECONDS = 0.5
DEFAULT_MAX_DELAY_SECONDS = 3.0
DEFAULT_BOOTSTRAP_WAIT_SECONDS = 20.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 10.0
MIN_INFERENCE_DURATION_MS = 50
MAX_INFERENCE_DURATION_MS = 10_000


@dataclass(frozen=True)
class EdgeConfig:
    edge_id: str
    api_key: str
    seed_offset: int


@dataclass
class WorkerSummary:
    requests: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def build_usage_payload(
    *,
    edge_id: str,
    model: str,
    rng: random.Random,
    bootstrap: bool = False,
) -> dict[str, Any]:
    """Build one usage report using the same schema as Usage Reporter."""
    if bootstrap:
        prompt_tokens = 0
        completion_tokens = 0
        duration_ms = 0
    else:
        prompt_tokens = rng.randint(
            DEFAULT_MIN_PROMPT_TOKENS,
            DEFAULT_MAX_PROMPT_TOKENS,
        )
        completion_tokens = rng.randint(
            DEFAULT_MIN_COMPLETION_TOKENS,
            DEFAULT_MAX_COMPLETION_TOKENS,
        )
        duration_ms = rng.randint(
            MIN_INFERENCE_DURATION_MS,
            MAX_INFERENCE_DURATION_MS,
        )

    return {
        "event_id": f"usage-sim-{edge_id}-{uuid.uuid4()}",
        "edge_id": edge_id,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "duration_ms": duration_ms,
        "status_code": 200,
    }


def post_usage(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> int:
    """Post one usage report and return the HTTP status code."""
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        status_code = response.status
        response.read()

    if status_code != 200:
        raise RuntimeError(f"usage endpoint returned HTTP {status_code}")
    return status_code


def print_event(
    *,
    edge_id: str,
    kind: str,
    payload: dict[str, Any],
    http_status: int | str,
) -> None:
    usage = payload["usage"]
    print(
        json.dumps(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "edge_id": edge_id,
                "kind": kind,
                "event_id": payload["event_id"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "http_status": http_status,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def inject_once(
    *,
    edge: EdgeConfig,
    model: str,
    url: str,
    rng: random.Random,
    timeout_seconds: float,
    dry_run: bool,
    bootstrap: bool = False,
) -> dict[str, Any]:
    payload = build_usage_payload(
        edge_id=edge.edge_id,
        model=model,
        rng=rng,
        bootstrap=bootstrap,
    )
    if dry_run:
        status: int | str = "DRY_RUN"
    else:
        status = post_usage(
            url=url,
            api_key=edge.api_key,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
    print_event(
        edge_id=edge.edge_id,
        kind="bootstrap" if bootstrap else "usage",
        payload=payload,
        http_status=status,
    )
    return payload


def run_worker(
    *,
    edge: EdgeConfig,
    args: argparse.Namespace,
    stop_event: threading.Event,
    deadline: float | None,
    summary: WorkerSummary,
) -> None:
    seed = None if args.seed is None else args.seed + edge.seed_offset
    rng = random.Random(seed)

    while not stop_event.is_set():
        if deadline is not None and time.monotonic() >= deadline:
            break

        try:
            payload = inject_once(
                edge=edge,
                model=args.model,
                url=args.url,
                rng=rng,
                timeout_seconds=args.request_timeout_seconds,
                dry_run=args.dry_run,
            )
            usage = payload["usage"]
            summary.requests += 1
            summary.prompt_tokens += usage["prompt_tokens"]
            summary.completion_tokens += usage["completion_tokens"]
        except (HTTPError, URLError, TimeoutError, RuntimeError, OSError) as exc:
            summary.failures += 1
            print(
                json.dumps(
                    {
                        "time": datetime.now(timezone.utc).isoformat(),
                        "edge_id": edge.edge_id,
                        "kind": "error",
                        "error": str(exc),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

        delay_seconds = rng.uniform(
            args.min_delay_seconds,
            args.max_delay_seconds,
        )
        if deadline is not None:
            delay_seconds = min(
                delay_seconds,
                max(0.0, deadline - time.monotonic()),
            )
        stop_event.wait(delay_seconds)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run concurrent edge-0 and edge-1 usage injectors. Each edge "
            "independently waits a random interval between requests."
        )
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--edge-0-key")
    parser.add_argument("--edge-1-key")
    parser.add_argument(
        "--min-delay-seconds",
        type=positive_float,
        default=DEFAULT_MIN_DELAY_SECONDS,
    )
    parser.add_argument(
        "--max-delay-seconds",
        type=positive_float,
        default=DEFAULT_MAX_DELAY_SECONDS,
    )
    parser.add_argument(
        "--duration-seconds",
        type=non_negative_float,
        default=0.0,
        help="Stop after this many seconds; 0 means run until Ctrl-C.",
    )
    parser.add_argument(
        "--bootstrap-wait-seconds",
        type=non_negative_float,
        default=DEFAULT_BOOTSTRAP_WAIT_SECONDS,
        help=("Send zero-token bootstrap events, then wait this long before load starts; 0 disables bootstrapping."),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=positive_float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and print events without sending HTTP requests.",
    )
    args = parser.parse_args()

    if args.min_delay_seconds > args.max_delay_seconds:
        parser.error("--min-delay-seconds must not exceed --max-delay-seconds")

    if args.dry_run:
        args.edge_0_key = args.edge_0_key or "dry-run"
        args.edge_1_key = args.edge_1_key or "dry-run"
    else:
        args.edge_0_key = args.edge_0_key or getpass.getpass("edge-0 API key: ")
        args.edge_1_key = args.edge_1_key or getpass.getpass("edge-1 API key: ")
        if not args.edge_0_key or not args.edge_1_key:
            parser.error("both edge API keys must be non-empty")

    return args


def main() -> int:
    args = parse_args()
    edges = (
        EdgeConfig("edge-0", args.edge_0_key, 0),
        EdgeConfig("edge-1", args.edge_1_key, 1),
    )
    summaries = {edge.edge_id: WorkerSummary() for edge in edges}
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if args.bootstrap_wait_seconds > 0:
        with ThreadPoolExecutor(
            max_workers=len(edges),
            thread_name_prefix="usage-bootstrap",
        ) as executor:
            futures = [
                executor.submit(
                    inject_once,
                    edge=edge,
                    model=args.model,
                    url=args.url,
                    rng=random.Random(args.seed),
                    timeout_seconds=args.request_timeout_seconds,
                    dry_run=args.dry_run,
                    bootstrap=True,
                )
                for edge in edges
            ]
            try:
                for future in futures:
                    future.result()
            except (HTTPError, URLError, TimeoutError, RuntimeError, OSError) as exc:
                print(f"Bootstrap failed: {exc}", flush=True)
                return 1

        print(
            f"Waiting {args.bootstrap_wait_seconds:g}s for Prometheus to scrape zero-token baselines...",
            flush=True,
        )
        if stop_event.wait(args.bootstrap_wait_seconds):
            return 0

    deadline = None
    if args.duration_seconds > 0:
        deadline = time.monotonic() + args.duration_seconds

    workers = []
    for edge in edges:
        thread = threading.Thread(
            target=run_worker,
            kwargs={
                "edge": edge,
                "args": args,
                "stop_event": stop_event,
                "deadline": deadline,
                "summary": summaries[edge.edge_id],
            },
            name=f"{edge.edge_id}-injector",
        )
        thread.start()
        workers.append(thread)

    try:
        while any(thread.is_alive() for thread in workers):
            for thread in workers:
                thread.join(timeout=0.2)
    finally:
        stop_event.set()
        for thread in workers:
            thread.join()

    print("Summary:", flush=True)
    for edge in edges:
        summary = summaries[edge.edge_id]
        print(
            json.dumps(
                {
                    "edge_id": edge.edge_id,
                    "requests": summary.requests,
                    "failures": summary.failures,
                    "prompt_tokens": summary.prompt_tokens,
                    "completion_tokens": summary.completion_tokens,
                    "total_tokens": (summary.prompt_tokens + summary.completion_tokens),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
