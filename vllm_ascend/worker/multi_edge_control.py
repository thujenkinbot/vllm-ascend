"""Trusted control plane for the v0.20 multi-edge-cloud MVP.

The transport deliberately stays small: edge executors submit one synchronous
RPC at a time and the cloud executes RPCs serially. Payloads use cloudpickle so
the endpoint must only be reachable on a trusted inference network.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cloudpickle
import zmq


class DeferCloudRPC(Exception):
    """Ask the arbiter to retry an RPC after another edge makes progress."""


@dataclass
class CloudRPCRequest:
    edge_id: int
    method: str | Callable
    timeout: float | None
    args: tuple
    kwargs: dict[str, Any]


class MultiEdgeControlClient:
    def __init__(self, *, edge_id: int, cloud_addr: str, port: int) -> None:
        if not cloud_addr:
            raise ValueError(
                "VLLM_ASCEND_EDGE_CLOUD_CLOUD_ADDR is required on every edge when num_edges is greater than 1"
            )
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.IDENTITY, f"edge-{edge_id}".encode())
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{cloud_addr}:{port}")
        self.edge_id = edge_id

    def send(self, request: CloudRPCRequest) -> None:
        self._socket.send(cloudpickle.dumps(request))

    def receive(self, timeout: float | None = None) -> Any:
        if timeout is not None:
            poller = zmq.Poller()
            poller.register(self._socket, zmq.POLLIN)
            if not poller.poll(timeout=max(0, int(timeout * 1000))):
                raise TimeoutError("Timed out waiting for the shared cloud executor")
        ok, result = cloudpickle.loads(self._socket.recv())
        if not ok:
            raise RuntimeError(f"Cloud multi-edge RPC failed: {result}")
        return result

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


class MultiEdgeCloudArbiter:
    """Serial, round-robin dispatcher for independent edge control streams."""

    def __init__(self, *, num_edges: int, port: int) -> None:
        if num_edges < 2:
            raise ValueError("MultiEdgeCloudArbiter requires at least two edges")
        self.num_edges = num_edges
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.LINGER, 0)
        if port == 0:
            self.port = self._socket.bind_to_random_port("tcp://*")
        else:
            if not 1 <= port <= 65535:
                raise ValueError("control port must be in [1, 65535]")
            self.port = port
            self._socket.bind(f"tcp://*:{port}")
        self._pending = [deque() for _ in range(num_edges)]
        self._next_edge = 0

    def _drain(self) -> None:
        while True:
            try:
                identity, payload = self._socket.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            request = cloudpickle.loads(payload)
            if not isinstance(request, CloudRPCRequest):
                self._socket.send_multipart([identity, cloudpickle.dumps((False, "invalid RPC payload"))])
                continue
            if not 0 <= request.edge_id < self.num_edges:
                self._socket.send_multipart(
                    [
                        identity,
                        cloudpickle.dumps((False, f"edge_id={request.edge_id} is out of range")),
                    ]
                )
                continue
            if identity != f"edge-{request.edge_id}".encode():
                self._socket.send_multipart([identity, cloudpickle.dumps((False, "edge identity mismatch"))])
                continue
            self._pending[request.edge_id].append((identity, request))

    def _pop_next(self):
        for offset in range(self.num_edges):
            edge_id = (self._next_edge + offset) % self.num_edges
            if self._pending[edge_id]:
                self._next_edge = (edge_id + 1) % self.num_edges
                return self._pending[edge_id].popleft()
        return None

    def run(
        self,
        execute: Callable[[CloudRPCRequest], Any],
        should_stop: Callable[[], bool],
    ) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not should_stop():
            poller.poll(timeout=100)
            self._drain()
            item = self._pop_next()
            if item is None:
                continue
            identity, request = item
            try:
                response = (True, execute(request))
            except DeferCloudRPC:
                self._pending[request.edge_id].append((identity, request))
                continue
            except Exception as error:
                response = (False, str(error))
            self._socket.send_multipart([identity, cloudpickle.dumps(response)])

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()
