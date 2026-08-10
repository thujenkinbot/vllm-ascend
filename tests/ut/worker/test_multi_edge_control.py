from threading import Event, Thread

from vllm_ascend.worker.multi_edge_control import (
    CloudRPCRequest,
    MultiEdgeCloudArbiter,
    MultiEdgeControlClient,
)


def test_cloud_arbiter_pops_edges_round_robin() -> None:
    arbiter = MultiEdgeCloudArbiter(num_edges=2, port=0)
    try:
        arbiter._pending[0].extend([("edge-0", "0a"), ("edge-0", "0b")])
        arbiter._pending[1].extend([("edge-1", "1a"), ("edge-1", "1b")])

        assert arbiter._pop_next() == ("edge-0", "0a")
        assert arbiter._pop_next() == ("edge-1", "1a")
        assert arbiter._pop_next() == ("edge-0", "0b")
        assert arbiter._pop_next() == ("edge-1", "1b")
    finally:
        arbiter.close()


def test_cloud_arbiter_routes_responses_to_each_edge() -> None:
    arbiter = MultiEdgeCloudArbiter(num_edges=2, port=0)
    stop = Event()
    thread = Thread(
        target=arbiter.run,
        args=(lambda request: request.edge_id, stop.is_set),
        daemon=True,
    )
    edge_0 = MultiEdgeControlClient(edge_id=0, cloud_addr="127.0.0.1", port=arbiter.port)
    edge_1 = MultiEdgeControlClient(edge_id=1, cloud_addr="127.0.0.1", port=arbiter.port)
    thread.start()
    try:
        edge_0.send(CloudRPCRequest(0, "execute_model", 1, (), {}))
        edge_1.send(CloudRPCRequest(1, "execute_model", 1, (), {}))

        assert edge_0.receive(timeout=1) == 0
        assert edge_1.receive(timeout=1) == 1
    finally:
        stop.set()
        thread.join(timeout=1)
        edge_0.close()
        edge_1.close()
        arbiter.close()
