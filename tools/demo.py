#!/usr/bin/env python3
"""A ~30 s demo that needs no GPU: this repo's gateway in front of the calibrated mock llama-server.

  1. cache thrash: one agent session vs two on a single slot
  2. backpressure: a burst against 1 slot + queue <= 2 -> 429 with Retry-After
  3. failure: the upstream crashes mid-stream -> a typed error event, then recovery
  4. the gateway's own /metrics

Run: make demo   (or: uv run python tools/demo.py)
"""

import time
import httpx
import asyncio
import logging
import tempfile
from dataclasses import replace
from llmserve.servers import serve
from llmserve.loadgen import Workload
from llmserve.sim import SimConfig, run_sim
from llmserve.gateway import Settings, create_app
from llmserve.mock_upstream import MockState, create_mock

SCALE = 0.1  # the mock runs 10x faster than the real server; times below are scaled back


def say(msg: str) -> None:
    print(f"\n=== {msg}")


def thrash() -> None:
    say("1. Cache thrash: agent sessions sharing one KV cache (simulated seconds)")
    w = Workload(
        kind="agent", turns=3, first_context=(4000, 4000), new_tokens=(150,), gen_tokens=(60,), think_s=(1.0, 3.0)
    )
    for n in (1, 2):
        summary, results = run_sim(w, SimConfig(sessions=n, time_scale=SCALE, max_queue=8))
        later = [r for r in results if r.turn > 0 and r.status == 200]
        cached = sum(r.prompt_cached or 0 for r in later) / max(
            1, sum((r.prompt_new or 0) + (r.prompt_cached or 0) for r in later)
        )
        print(f"   {n} session(s): cached share of prompt tokens {cached:4.0%}   TTFT p50 {summary['ttft_s']['p50']} s")


async def burst_and_crash(gw: str, st: MockState) -> None:
    async with httpx.AsyncClient(base_url=f"{gw}/v1", timeout=60) as c:
        say("2. Backpressure: 6 requests at once against 1 slot + queue <= 2")
        body = {"messages": [{"role": "user", "content": "the " * 200}], "max_tokens": 120}
        await c.post("/chat/completions", json=body)  # one finished request seeds the Retry-After estimate
        rs = await asyncio.gather(*(c.post("/chat/completions", json=body) for _ in range(6)))
        for r in rs:
            extra = f"  Retry-After: {r.headers['retry-after']} s" if r.status_code == 429 else ""
            print(f"   HTTP {r.status_code}{extra}")

        say("3. Failure: the upstream drops the stream after 5 tokens")
        st.faults.drop_after_tokens = 5
        last = ""
        async with c.stream("POST", "/chat/completions", json={**body, "stream": True}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    last = line
        print(f"   last event the client saw: {last[:110]}")
        st.faults.drop_after_tokens = None
        t0 = time.perf_counter()
        ok = await c.post("/chat/completions", json=body)
        print(f"   next request: HTTP {ok.status_code} after {time.perf_counter() - t0:.2f} s (recovered)")

        say("4. Gateway /metrics")
        m = (await c.get(f"{gw}/metrics")).json()
        print(f"   admission: {m['admission']}")
        print(f"   status counts: {m['status_counts']}")


def main() -> None:
    thrash()
    st = MockState(time_scale=SCALE)
    settings = Settings(upstream="", log_path=tempfile.mktemp(), max_queue=2, log_preview=False)
    with serve(create_mock(st)) as up, serve(create_app(replace(settings, upstream=up))) as gw:
        # the mock's injected crash makes uvicorn log a traceback; it is the point of step 3, so keep it quiet
        logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
        asyncio.run(burst_and_crash(gw, st))
    print("\nFull experiments: docs/15 (concurrency) and docs/16 (failure injection).")


if __name__ == "__main__":
    main()
