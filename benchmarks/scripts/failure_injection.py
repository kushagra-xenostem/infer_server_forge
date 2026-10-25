#!/usr/bin/env python3
"""Failure injection with measured detection and recovery times (docs/16).

  python benchmarks/scripts/failure_injection.py live   black-box faults against the real endpoint (.env)
  python benchmarks/scripts/failure_injection.py sim    faults against this repo's gateway + the mock upstream

Each fault writes one JSON line to benchmarks/results/failures/<mode>.jsonl: what was injected, what the
client saw, how long until the failure was visible (detect_s) and until a normal request succeeded again
(recover_s). Live faults are limited to what a client can do; server-side kills are in `sim` (and listed
as NOT RUN on hardware in docs/16).
"""

import sys
import json
import time
import httpx
import asyncio
from typing import Any
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from llmserve.cli import read_env  # noqa: E402

OUT = ROOT / "benchmarks/results/failures"
SMALL = {"messages": [{"role": "user", "content": "What is 17*23? Reply with only the number."}], "max_tokens": 8}


def record(mode: str, rec: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": mode, **rec}
    with (OUT / f"{mode}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


async def ttft(c: httpx.AsyncClient, body: dict[str, Any]) -> tuple[int, float | None, float]:
    """(status, time to first content chunk, total) for a streamed request."""
    t0 = time.perf_counter()
    first = None
    async with c.stream("POST", "/chat/completions", json={**body, "stream": True}) as r:
        if r.status_code != 200:
            await r.aread()
            return r.status_code, None, time.perf_counter() - t0
        async for line in r.aiter_lines():
            if first is None and '"content"' in line:
                first = time.perf_counter() - t0
    return 200, first, time.perf_counter() - t0


async def live() -> None:
    env = read_env(ROOT / ".env")
    base, key, model = env["LLM_BASE_URL"].rstrip("/"), env["LLM_API_KEY"], env.get("LLM_MODEL", "default")
    hdr = {"Authorization": f"Bearer {key}"}
    small = {**SMALL, "model": model}
    async with httpx.AsyncClient(base_url=base, headers=hdr, timeout=600) as c:
        # warm-up + baseline
        await c.post("/chat/completions", json=small)
        st, base_ttft, base_total = await ttft(c, small)
        record("live", {"fault": "baseline", "status": st, "ttft_s": base_ttft, "total_s": base_total})

        # 1) bad API key
        t0 = time.perf_counter()
        r = await c.post("/chat/completions", json=small, headers={"Authorization": "Bearer wrong"})
        record(
            "live",
            {
                "fault": "bad_api_key",
                "status": r.status_code,
                "detect_s": time.perf_counter() - t0,
                "body": r.text[:160],
            },
        )

        # 2) malformed JSON body
        t0 = time.perf_counter()
        r = await c.post("/chat/completions", content=b"{not json", headers={"content-type": "application/json"})
        record(
            "live",
            {
                "fault": "malformed_body",
                "status": r.status_code,
                "detect_s": time.perf_counter() - t0,
                "body": r.text[:160],
            },
        )

        # 3) oversized prompt (> 64k context): must fail fast, not hang or take the slot for minutes
        big = {
            "model": model,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "the and " * 40000}],
        }  # ~80k tokens
        t0 = time.perf_counter()
        r = await c.post("/chat/completions", json=big)
        detect = time.perf_counter() - t0
        st2, t2, _ = await ttft(c, small)
        record(
            "live",
            {
                "fault": "oversized_prompt",
                "status": r.status_code,
                "detect_s": detect,
                "body": r.text[:200],
                "next_request_status": st2,
                "next_ttft_s": t2,
            },
        )

        # 4) client disconnect after the first token of a long generation, then a new request at once:
        #    if the server keeps generating for the gone client, the new request waits behind it
        long = {
            "model": model,
            "max_tokens": 1500,
            "messages": [{"role": "user", "content": "Count upward from 1, one number per line. Never stop."}],
        }
        t0 = time.perf_counter()
        async with c.stream("POST", "/chat/completions", json={**long, "stream": True}) as r:
            async for line in r.aiter_lines():
                if '"content"' in line:
                    break  # hang up
        hung_up = time.perf_counter() - t0
        st3, t3, total3 = await ttft(c, small)
        record(
            "live",
            {
                "fault": "client_disconnect_mid_stream",
                "hung_up_after_s": hung_up,
                "next_request_status": st3,
                "next_ttft_s": t3,
                "baseline_ttft_s": base_ttft,
                "note": "next_ttft ~ baseline: slot freed on disconnect; ~1500/16 s: kept generating",
            },
        )

        # 5) two requests at once (the deployed gateway's admission behaviour)
        t0 = time.perf_counter()
        rs = await asyncio.gather(*(c.post("/chat/completions", json=small) for _ in range(4)))
        record(
            "live",
            {
                "fault": "burst_of_4",
                "statuses": [x.status_code for x in rs],
                "retry_after": [x.headers.get("retry-after") for x in rs],
                "total_s": time.perf_counter() - t0,
            },
        )


def sim() -> None:
    """Server-side faults on this repo's gateway + the mock (real-time speeds, small prompts).

    Every fault gets a fresh mock upstream on the same port (as after a restart), so one fault's leftovers
    (e.g. a hung slot) cannot leak into the next measurement.
    """
    import tempfile
    from contextlib import contextmanager

    from llmserve.gateway import Settings, create_app
    from llmserve.mock_upstream import MockState, create_mock
    from llmserve.servers import free_port, serve

    small = {"messages": [{"role": "user", "content": "the " * 50}], "max_tokens": 16}
    long = {**small, "max_tokens": 200}
    port = free_port()
    cfg = Settings(
        upstream=f"http://127.0.0.1:{port}",
        log_path=tempfile.mktemp(suffix=".jsonl"),
        max_queue=2,
        deadline_s=8.0,
        connect_timeout_s=2.0,
        log_preview=False,
    )

    @contextmanager
    def upstream() -> Any:
        st = MockState(time_scale=1.0)
        with serve(create_mock(st), port=port):
            yield st

    async def first_success(c: httpx.AsyncClient, honour_retry_after: bool, limit_s: float = 60) -> tuple[float, int]:
        t0 = time.perf_counter()
        attempts = 0
        while time.perf_counter() - t0 < limit_s:
            attempts += 1
            r = await c.post("/chat/completions", json=small)
            if r.status_code == 200:
                return time.perf_counter() - t0, attempts
            await asyncio.sleep(float(r.headers.get("retry-after", 1)) if honour_retry_after else 0.2)
        return float("nan"), attempts

    with serve(create_app(cfg)) as gw:
        client = lambda: httpx.AsyncClient(base_url=f"{gw}/v1", timeout=60)  # noqa: E731

        # 1) upstream dies mid-stream (connection reset after 5 tokens)
        with upstream() as st:

            async def drop() -> None:
                async with client() as c:
                    st.faults.drop_after_tokens = 5
                    t0 = time.perf_counter()
                    last = ""
                    async with c.stream("POST", "/chat/completions", json={**small, "stream": True}) as r:
                        async for line in r.aiter_lines():
                            if line.startswith("data: "):
                                last = line
                    detect = time.perf_counter() - t0
                    st.faults.drop_after_tokens = None
                    rec, _ = await first_success(c, True)
                    record(
                        "sim",
                        {
                            "fault": "upstream_drop_mid_stream",
                            "client_saw": json.loads(last[6:]).get("error"),
                            "detect_s": round(detect, 3),
                            "recover_s": round(rec, 3),
                        },
                    )

            asyncio.run(drop())

        # 2) hung upstream: the gateway deadline answers the client and frees the gateway slot; the hung
        #    server keeps its own slot, so only a restart recovers it (systemd/watchdog in production)
        with upstream() as st:

            async def hang() -> None:
                async with client() as c:
                    st.faults.hang = True
                    t0 = time.perf_counter()
                    r = await c.post("/chat/completions", json=small)
                    detect = time.perf_counter() - t0
                    m = (await c.get(f"{gw}/metrics")).json()["admission"]
                    record(
                        "sim",
                        {
                            "fault": "upstream_hang",
                            "status": r.status_code,
                            "deadline_s": cfg.deadline_s,
                            "detect_s": round(detect, 3),
                            "gateway_inflight_after": m["inflight"],
                            "upstream_slot_still_busy": any(s.busy for s in st.slots),
                            "recovery": "restart of the upstream (next fault starts on a fresh one)",
                        },
                    )

            asyncio.run(hang())

        # 3) cold start: the model "loads" for 25 s; a client honouring Retry-After vs a 0.2 s retry loop
        for honour in (True, False):
            with upstream() as st:

                async def cold(honour: bool = honour) -> None:
                    async with client() as c:
                        st.faults.loading_until = time.monotonic() + 25
                        took, attempts = await first_success(c, honour)
                        record(
                            "sim",
                            {
                                "fault": "cold_start_25s",
                                "client": "honours Retry-After" if honour else "retries every 0.2 s",
                                "first_success_s": round(took, 2),
                                "requests_sent": attempts,
                            },
                        )

                asyncio.run(cold())

        # 4) queue overflow: a burst of 10 long requests against 1 slot + queue <= 2
        with upstream() as st:

            async def burst() -> None:
                async with client() as c:
                    await c.post("/chat/completions", json=long)  # one completed request feeds the estimate
                    t0 = time.perf_counter()
                    rs = await asyncio.gather(*(c.post("/chat/completions", json=long) for _ in range(10)))
                    rejected = [x for x in rs if x.status_code == 429]
                    reject_ms = [x.elapsed.total_seconds() * 1000 for x in rejected]
                    record(
                        "sim",
                        {
                            "fault": "burst_10_vs_queue_2",
                            "statuses": sorted(x.status_code for x in rs),
                            "retry_after": sorted({x.headers["retry-after"] for x in rejected}),
                            "reject_ms_max": round(max(reject_ms), 1) if reject_ms else None,
                            "total_s": round(time.perf_counter() - t0, 2),
                        },
                    )

            asyncio.run(burst())

        # 5) disconnect storm: 3 streams hang up after 3 lines; time until gateway and upstream slots are free
        with upstream() as st:

            async def storm() -> None:
                async with client() as c:

                    async def hang_up() -> None:
                        async with c.stream("POST", "/chat/completions", json={**long, "stream": True}) as r:
                            n = 0
                            async for _ in r.aiter_lines():
                                n += 1
                                if n > 3:
                                    return

                    await asyncio.gather(*(hang_up() for _ in range(3)), return_exceptions=True)
                    t0 = time.perf_counter()
                    while True:
                        m = (await c.get(f"{gw}/metrics")).json()["admission"]
                        if m["inflight"] == 0 and m["queued"] == 0 and not any(s.busy for s in st.slots):
                            break
                        await asyncio.sleep(0.02)
                    record(
                        "sim",
                        {
                            "fault": "disconnect_storm",
                            "slots_free_after_s": round(time.perf_counter() - t0, 3),
                            "upstream_streams_cancelled": st.cancelled,
                        },
                    )

            asyncio.run(storm())

        # 6) upstream process down, then restarted on the same port
        async def down() -> None:
            async with client() as c:
                t0 = time.perf_counter()
                r = await c.post("/chat/completions", json=small)
                record(
                    "sim",
                    {
                        "fault": "upstream_down",
                        "status": r.status_code,
                        "detect_s": round(time.perf_counter() - t0, 3),
                        "ready_status": (await c.get(f"{gw}/ready")).status_code,
                    },
                )

        asyncio.run(down())
        with upstream():

            async def restarted() -> None:
                async with client() as c:
                    rec, attempts = await first_success(c, True)
                    record(
                        "sim", {"fault": "upstream_restarted", "first_success_s": round(rec, 3), "attempts": attempts}
                    )

            asyncio.run(restarted())


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"
    if mode == "live":
        asyncio.run(live())
    elif mode == "sim":
        sim()
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()


