# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Small, request-owned D inputs; never retain a complete UP receive view."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LwdTailInput:
    connection: Any
    binding_id: str
    attempt_id: str
    start: int
    rows: int
    payload: Any
    ready_event: Any


class LwdTailBuffer:
    """Keep only received tail rows, with explicit identity and copy ownership."""

    def __init__(self, retention_s: float = 2400.0) -> None:
        self._lock = threading.Lock()
        self._inputs: dict[tuple[str, str, str], LwdTailInput] = {}
        self._retired: OrderedDict[tuple[str, str, str], float] = OrderedDict()
        self._retired_all: OrderedDict[str, float] = OrderedDict()
        self._retention_s = retention_s

    def _reap(self) -> None:
        now = time.monotonic()
        for table in (self._retired, self._retired_all):
            while table:
                if table[next(iter(table))] > now:
                    break
                table.popitem(last=False)

    def put(self, req_id: str, value: LwdTailInput) -> bool:
        if not value.binding_id or not value.attempt_id or value.rows < 1 or value.start < 0:
            raise ValueError("PD tail input is missing its identity or row geometry")
        with self._lock:
            self._reap()
            key = (req_id, value.binding_id, value.attempt_id)
            if key in self._retired or req_id in self._retired_all:
                return False
            if key in self._inputs:
                raise ValueError(f"PD tail input already buffered for {req_id}")
            self._inputs[key] = value
        return True

    def take(self, req_id: str, binding_id: str, attempt_id: str, start: int, rows: int) -> LwdTailInput:
        with self._lock:
            key = (req_id, binding_id, attempt_id)
            value = self._inputs.get(key)
            if value is None:
                raise RuntimeError(f"PD tail input is not ready or its identity does not match for {req_id}")
            if (value.binding_id, value.attempt_id, value.start, value.rows) != (binding_id, attempt_id, start, rows):
                raise RuntimeError(f"PD tail input identity or window mismatch for {req_id}")
            return self._inputs.pop(key)

    def retire(self, req_id: str, binding_id: str = "", attempt_id: str = "") -> None:
        with self._lock:
            self._reap()
            expiry = time.monotonic() + self._retention_s
            if binding_id and attempt_id:
                key = (req_id, binding_id, attempt_id)
                self._inputs.pop(key, None)
                self._retired[key] = expiry
                self._retired.move_to_end(key)
            else:
                for key in list(self._inputs):
                    if key[0] == req_id:
                        self._inputs.pop(key)
                self._retired_all[req_id] = expiry
                self._retired_all.move_to_end(req_id)

    def abandon_connection(self, connection: Any) -> None:
        with self._lock:
            for key, value in list(self._inputs.items()):
                if value.connection == connection:
                    self._inputs.pop(key)
                    self._retired[key] = time.monotonic() + self._retention_s
                    self._retired.move_to_end(key)
