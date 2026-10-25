"""Injected failures with recovery: upstream restart, cold start, disconnect storm."""

import time
import httpx
import pytest
import asyncio
from pathlib import Path
from llmserve.servers import free_port, serve
from llmserve.gateway import Settings, create_app
from llmserve.mock_upstream import MockState, create_mock

BODY = {"messages": [{"role": "user", "content": "tok " * 30}], "max_tokens": 4}


def test_upstream_restart_is_survived(tmp_path: Path) -> None:
    port = free_port()
    upstream = f"http://127.0.0.1:{port}"
    st = MockState(time_scale=0.001)
    with serve(
        create_app(Settings(upstream=upstream, log_path=str(tmp_path / "g.jsonl"), max_queue=2, connect_timeout_s=2))
    ) as gw:
        with serve(create_mock(st), port=port):
            assert httpx.post(f"{gw}/v1/chat/completions", json=BODY, timeout=10).status_code == 200
        # upstream gone (killed): the gateway answers 502 and keeps its slot accounting straight
        down = httpx.post(f"{gw}/v1/chat/completions", json=BODY, timeout=10)
        assert down.status_code == 502 and down.json()["error"]["type"] == "upstream_unavailable"
        assert httpx.get(f"{gw}/ready").status_code == 503
        with serve(create_mock(st), port=port):  # restarted on the same port
            assert httpx.get(f"{gw}/ready").status_code == 200
            assert httpx.post(f"{gw}/v1/chat/completions", json=BODY, timeout=10).status_code == 200
        m = httpx.get(f"{gw}/metrics").json()
    assert m["admission"]["inflight"] == 0 and m["status_counts"] == {"200": 2, "502": 1}


@pytest.mark.parametrize("stream", [False, True])
def test_cold_start_503_carries_retry_after(gateway_factory, mock_state: MockState, stream: bool) -> None:  # type: ignore[no-untyped-def]
    mock_state.faults.loading_until = time.monotonic() + 60
    with gateway_factory(max_queue=2) as gw:
        r = httpx.post(f"{gw}/v1/chat/completions", json={**BODY, "stream": stream})
        assert r.status_code == 503 and r.headers["Retry-After"] == "10"
        mock_state.faults.loading_until = 0
        assert httpx.post(f"{gw}/v1/chat/completions", json={**BODY, "stream": stream}).status_code == 200
        assert httpx.get(f"{gw}/metrics").json()["admission"]["inflight"] == 0


@pytest.mark.anyio
async def test_disconnect_storm_leaks_no_slots(gateway_factory, mock_state: MockState) -> None:  # type: ignore[no-untyped-def]
    mock_state.time_scale = 0.005
    with gateway_factory(max_queue=12) as gw:
        async with httpx.AsyncClient(timeout=60) as c:

            async def client(i: int) -> int:
                body = {**BODY, "max_tokens": 200, "stream": True}
                async with c.stream("POST", f"{gw}/v1/chat/completions", json=body) as r:
                    n = 0
                    async for _ in r.aiter_lines():
                        n += 1
                        if i % 2 and n > 3:
                            return -1  # half the clients hang up mid-stream
                    return r.status_code

            results = await asyncio.gather(*(client(i) for i in range(10)))
            t0 = time.monotonic()
            while True:
                m = (await c.get(f"{gw}/metrics")).json()["admission"]
                if m["inflight"] == 0 and m["queued"] == 0:
                    break
                assert time.monotonic() - t0 < 2, f"slots leaked: {m}"
                await asyncio.sleep(0.05)
    assert results.count(200) == 5 and results.count(-1) == 5
    assert not any(s.busy for s in mock_state.slots)


def test_failures_do_not_shrink_retry_after(gateway_factory, mock_state: MockState) -> None:  # type: ignore[no-untyped-def]
    # Regression: fast 502/504 failures were counted as service times, so Retry-After fell to 1 s
    mock_state.faults.extra_latency_s = 2000  # a normal request: 2 s real
    with gateway_factory(max_queue=1) as gw:
        assert httpx.post(f"{gw}/v1/chat/completions", json=BODY, timeout=30).status_code == 200
        mock_state.faults.error_status = 500
        for _ in range(5):
            httpx.post(f"{gw}/v1/chat/completions", json=BODY)
        m = httpx.get(f"{gw}/metrics").json()["admission"]
    assert m["median_service_s"] >= 1.9


def test_mock_cancels_a_non_streaming_request_when_the_client_leaves(mock_url: str, mock_state: MockState) -> None:
    mock_state.faults.extra_latency_s = 60000  # would take a minute
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(f"{mock_url}/v1/chat/completions", json=BODY, timeout=0.3)
    t0 = time.monotonic()
    while mock_state.cancelled == 0:
        assert time.monotonic() - t0 < 2, "mock kept working for a client that left"
        time.sleep(0.02)
    assert not any(s.busy for s in mock_state.slots)
