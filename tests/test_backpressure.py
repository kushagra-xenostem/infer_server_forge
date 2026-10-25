"""End-to-end admission control: real gateway + mock upstream servers, concurrent clients."""

import json
import time
import httpx
import pytest
import asyncio
from pathlib import Path
from llmserve.mock_upstream import MockState

pytestmark = pytest.mark.anyio
BODY = {"messages": [{"role": "user", "content": "tok " * 50}], "max_tokens": 4, "stream": True}


def log_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def metrics(c: httpx.AsyncClient, gw: str) -> dict[str, object]:
    return (await c.get(f"{gw}/metrics")).json()  # type: ignore[no-any-return]


async def test_queue_never_exceeds_its_bound(gateway_factory, mock_state: MockState) -> None:  # type: ignore[no-untyped-def]
    mock_state.faults.extra_latency_s = 300  # each request holds the slot ~0.3 s of real time
    with gateway_factory(max_queue=2) as gw:
        async with httpx.AsyncClient(timeout=30) as c:
            rs = await asyncio.gather(*(c.post(f"{gw}/v1/chat/completions", json=BODY) for _ in range(6)))
            m = await metrics(c, gw)
    codes = sorted(r.status_code for r in rs)
    assert codes == [200, 200, 200, 429, 429, 429]  # 1 in flight + 2 queued; the rest rejected
    for r in rs:
        if r.status_code == 429:
            assert int(r.headers["Retry-After"]) >= 1
            assert r.json()["error"]["type"] == "queue_full"
    adm = m["admission"]
    assert adm["rejected"] == 3 and adm["admitted"] == 3 and adm["inflight"] == 0  # type: ignore[index]
    assert mock_state.served == 3


async def test_queued_requests_record_their_wait(gateway_factory, mock_state: MockState, log_path: Path) -> None:  # type: ignore[no-untyped-def]
    mock_state.faults.extra_latency_s = 200
    with gateway_factory(max_queue=4) as gw:
        async with httpx.AsyncClient(timeout=30) as c:
            await asyncio.gather(*(c.post(f"{gw}/v1/chat/completions", json=BODY) for _ in range(3)))
    waits = sorted(float(r["queue_wait_s"]) for r in log_lines(log_path))  # type: ignore[arg-type]
    assert waits[0] == 0.0 and waits[2] >= 0.3  # the third waited for two others


async def test_disconnected_stream_releases_the_slot(gateway_factory, mock_state: MockState, log_path: Path) -> None:  # type: ignore[no-untyped-def]
    with gateway_factory(max_queue=2) as gw:
        async with httpx.AsyncClient(timeout=30) as c:
            long = {**BODY, "max_tokens": 100000}
            mock_state.ctx_size = 10**6
            mock_state.time_scale = 0.01  # ~0.6 ms per token: the stream would run for a minute
            async with c.stream("POST", f"{gw}/v1/chat/completions", json=long) as r:
                async for _ in r.aiter_bytes():
                    break  # read one chunk, then hang up
            t0 = time.monotonic()
            while (await metrics(c, gw))["admission"]["inflight"] != 0:  # type: ignore[index]
                assert time.monotonic() - t0 < 2.0, "slot not released within 2 s"  # ~0.1 s alone; slack for load
                await asyncio.sleep(0.02)
            while mock_state.cancelled != 1:  # the upstream stopped generating, not just the gateway slot
                assert time.monotonic() - t0 < 2.0, "upstream generation not cancelled within 2 s"
                await asyncio.sleep(0.02)
            mock_state.time_scale = 0.001
            ok = await c.post(f"{gw}/v1/chat/completions", json=BODY)  # the upstream slot is free too
            assert ok.status_code == 200
    first = log_lines(log_path)[0]
    assert first["error"] == "client_disconnected"


async def test_client_leaving_the_queue_frees_its_place(gateway_factory, mock_state: MockState, log_path: Path) -> None:  # type: ignore[no-untyped-def]
    mock_state.faults.extra_latency_s = 1500
    with gateway_factory(max_queue=2) as gw:
        async with httpx.AsyncClient(timeout=30) as c:
            holder = asyncio.create_task(c.post(f"{gw}/v1/chat/completions", json=BODY))
            await asyncio.sleep(0.2)
            async with httpx.AsyncClient(timeout=0.4) as impatient:
                with pytest.raises(httpx.ReadTimeout):
                    await impatient.post(f"{gw}/v1/chat/completions", json=BODY)
            t0 = time.monotonic()
            while (await metrics(c, gw))["admission"]["queued"] != 0:  # type: ignore[index]
                assert time.monotonic() - t0 < 1.0, "queued client not removed within 1 s"
                await asyncio.sleep(0.02)
            assert (await holder).status_code == 200
    assert any(r["status"] == 499 for r in log_lines(log_path))
    assert mock_state.served == 1


@pytest.mark.parametrize("stream", [True, False])
async def test_deadline_cancels_a_hung_upstream(gateway_factory, mock_state: MockState, stream: bool) -> None:  # type: ignore[no-untyped-def]
    mock_state.faults.hang = True
    with gateway_factory(max_queue=2, deadline_s=0.5) as gw:
        async with httpx.AsyncClient(timeout=10) as c:
            t0 = time.monotonic()
            r = await c.post(f"{gw}/v1/chat/completions", json={**BODY, "stream": stream})
            took = time.monotonic() - t0
            if stream:
                assert r.status_code == 200
                last = [ln for ln in r.text.splitlines() if ln.startswith("data: ")][-1]
                assert json.loads(last[6:])["error"]["type"] == "deadline_exceeded"
            else:
                assert r.status_code == 504
            assert 0.4 < took < 2.0
            assert (await metrics(c, gw))["admission"]["inflight"] == 0  # type: ignore[index]
            mock_state.faults.hang = False
            assert (await c.post(f"{gw}/v1/chat/completions", json=BODY)).status_code == 200


async def test_queue_timeout_is_a_503_with_retry_after(gateway_factory, mock_state: MockState) -> None:  # type: ignore[no-untyped-def]
    mock_state.faults.extra_latency_s = 1000
    with gateway_factory(max_queue=2, queue_timeout_s=0.3) as gw:
        async with httpx.AsyncClient(timeout=30) as c:
            first, second = await asyncio.gather(
                c.post(f"{gw}/v1/chat/completions", json=BODY), c.post(f"{gw}/v1/chat/completions", json=BODY)
            )
    assert sorted([first.status_code, second.status_code]) == [200, 503]
    late = first if first.status_code == 503 else second
    assert late.json()["error"]["type"] == "queue_timeout" and "Retry-After" in late.headers


async def test_ready_and_metrics(gateway_factory, mock_state: MockState) -> None:  # type: ignore[no-untyped-def]
    with gateway_factory(max_queue=2) as gw:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(f"{gw}/v1/chat/completions", json=BODY)
            assert (await c.get(f"{gw}/ready")).json()["ready"] is True
            mock_state.faults.loading_until = time.monotonic() + 60
            r = await c.get(f"{gw}/ready")
            assert r.status_code == 503 and r.json()["upstream_health"] == 503
            m = await metrics(c, gw)
    assert m["status_counts"] == {"200": 1}
    assert m["ttft_s"]["n"] == 1  # type: ignore[index]


async def test_admission_off_keeps_the_old_behaviour(gateway_factory, mock_state: MockState) -> None:  # type: ignore[no-untyped-def]
    mock_state.faults.extra_latency_s = 100
    with gateway_factory() as gw:
        async with httpx.AsyncClient(timeout=30) as c:
            rs = await asyncio.gather(*(c.post(f"{gw}/v1/chat/completions", json=BODY) for _ in range(5)))
            assert (await metrics(c, gw))["admission"] is None
    assert all(r.status_code == 200 for r in rs)  # no rejections: everything piles up on the upstream slot


async def test_disconnect_detection_survives_the_auth_middleware(gateway_factory, mock_state: MockState) -> None:  # type: ignore[no-untyped-def]
    mock_state.faults.extra_latency_s = 1500
    auth = {"Authorization": "Bearer k"}
    with gateway_factory(max_queue=2, api_keys=frozenset({"k"})) as gw:
        async with httpx.AsyncClient(timeout=30, headers=auth) as c:
            holder = asyncio.create_task(c.post(f"{gw}/v1/chat/completions", json=BODY))
            await asyncio.sleep(0.2)
            async with httpx.AsyncClient(timeout=0.4, headers=auth) as impatient:
                with pytest.raises(httpx.ReadTimeout):
                    await impatient.post(f"{gw}/v1/chat/completions", json=BODY)
            t0 = time.monotonic()
            while (await metrics(c, gw))["admission"]["queued"] != 0:  # type: ignore[index]
                assert time.monotonic() - t0 < 1.0, "queued client not removed within 1 s"
                await asyncio.sleep(0.02)
            assert (await holder).status_code == 200
    assert mock_state.served == 1
