import json
import time
import httpx
import pytest
import random
import asyncio
from llmserve.servers import serve
from llmserve.gateway import make_policy
from llmserve.mock_upstream import MockState, create_mock
from llmserve.mock_upstream import common_prefix as mock_common_prefix
from llmserve.admission import FIFO, Admission, CacheAffinity, Waiter, common_prefix, render, same_session


def naive_prefix(a: str, b: str) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


@pytest.mark.parametrize("fn", [common_prefix, mock_common_prefix])
def test_common_prefix_matches_the_naive_version(fn) -> None:  # type: ignore[no-untyped-def]
    rng = random.Random(1)
    for _ in range(500):
        base = "".join(rng.choice("ab") for _ in range(rng.randint(0, 40)))
        a = base + "".join(rng.choice("ab") for _ in range(rng.randint(0, 5)))
        b = base + "".join(rng.choice("ab") for _ in range(rng.randint(0, 5)))
        assert fn(a, b) == naive_prefix(a, b)


def conv(session: int, turns: int) -> dict[str, object]:
    msgs = [{"role": "system", "content": f"session {session} " + "notes " * 50}]
    for t in range(turns):
        msgs += [{"role": "user", "content": f"q{t}"}, {"role": "assistant", "content": f"a{t}"}]
    return {"messages": msgs}


def test_same_session_recognises_a_continued_conversation() -> None:
    resident = render(conv(1, 2))
    assert same_session(render(conv(1, 3)), resident)
    assert not same_session(render(conv(2, 3)), resident)
    assert not same_session(render(conv(1, 3)), "")


def waiter(prompt: str, age_s: float = 0.0) -> Waiter:
    return Waiter({"render": prompt}, time.monotonic() - age_s, None)  # type: ignore[arg-type]


def test_affinity_prefers_the_resident_session_and_fifo_does_not() -> None:
    resident = render(conv(2, 1))
    ws = [waiter(render(conv(1, 2))), waiter(render(conv(2, 2))), waiter(render(conv(3, 2)))]
    assert CacheAffinity().pick(ws, resident) == 1
    assert FIFO().pick(ws, resident) == 0


def test_affinity_starvation_guards() -> None:
    resident = render(conv(2, 1))
    old = [waiter(render(conv(1, 2)), age_s=500), waiter(render(conv(2, 2)))]
    assert CacheAffinity(max_wait_s=120).pick(old, resident) == 0  # too old: it goes first
    pol = CacheAffinity(max_skips=2)
    ws = [waiter(render(conv(1, 2))), waiter(render(conv(2, 2)))]
    assert pol.pick(ws, resident) == 1
    assert pol.pick(ws, resident) == 1
    assert pol.pick(ws, resident) == 0  # skipped twice: its turn now


def test_make_policy() -> None:
    assert make_policy("fifo").name == "fifo" and make_policy("affinity").name == "affinity"
    with pytest.raises(ValueError, match="unknown queue policy"):
        make_policy("lifo")


@pytest.mark.anyio
async def test_hold_keeps_the_slot_for_the_same_session() -> None:
    a = Admission(max_queue=4, hold_s=0.5)
    s1 = render(conv(1, 1))
    await a.acquire({"render": s1})
    a.release(1.0)
    assert a.holding and a.inflight == 1
    other = asyncio.create_task(a.acquire({"render": render(conv(2, 1))}))
    await asyncio.sleep(0.05)
    assert not other.done()  # another session queues while the slot is held
    assert await a.acquire({"render": render(conv(1, 2))}) == 0.0  # the held-for session walks in
    assert a.hold_hits == 1 and not a.holding
    a.release(1.0)  # session 1 done again: hold again; nobody from session 1 comes back
    await asyncio.wait_for(other, 2)  # after hold_s the queued session gets the slot
    assert a.hold_expired == 1 and a.inflight == 1


@pytest.mark.anyio
async def test_hold_hands_over_at_once_when_the_session_is_already_queued() -> None:
    a = Admission(max_queue=4, hold_s=10)
    await a.acquire({"render": render(conv(1, 1))})
    queued_other = asyncio.create_task(a.acquire({"render": render(conv(2, 1))}))
    queued_same = asyncio.create_task(a.acquire({"render": render(conv(1, 2))}))
    await asyncio.sleep(0.05)
    a.release(1.0)
    await asyncio.wait_for(queued_same, 1)  # no waiting for the hold: it was already here
    assert not queued_other.done() and not a.holding
    queued_other.cancel()


@pytest.mark.anyio
async def test_hold_off_is_plain_handoff() -> None:
    a = Admission(max_queue=4)
    await a.acquire({"render": "x"})
    a.release(1.0)
    assert not a.holding and a.inflight == 0


def chat(url: str, body: dict[str, object], max_tokens: int = 4) -> dict[str, object]:
    r = httpx.post(f"{url}/v1/chat/completions", json={**body, "max_tokens": max_tokens}, timeout=30)
    assert r.status_code == 200, r.text
    return r.json()["timings"]  # type: ignore[no-any-return]


def test_two_slots_keep_two_caches() -> None:
    st = MockState(time_scale=0.001, n_slots=2)
    with serve(create_mock(st)) as url:
        chat(url, conv(1, 1))
        chat(url, conv(2, 1))
        t1 = chat(url, conv(1, 2))
        t2 = chat(url, conv(2, 2))
    assert t1["cache_n"] > 50 and t2["cache_n"] > 50  # each session found its own slot's cache


def test_one_slot_thrashes_with_the_same_traffic() -> None:
    st = MockState(time_scale=0.001, n_slots=1)
    with serve(create_mock(st)) as url:
        chat(url, conv(1, 1))
        chat(url, conv(2, 1))
        t1 = chat(url, conv(1, 2))
    assert t1["cache_n"] < 10  # session 2 evicted session 1's cache


def test_slots_split_the_context() -> None:
    st = MockState(time_scale=0.001, n_slots=2, ctx_size=1000)
    body = {"messages": [{"role": "user", "content": "tok " * 600}], "max_tokens": 4}
    with serve(create_mock(st)) as url:
        r = httpx.post(f"{url}/v1/chat/completions", json=body)
    assert r.status_code == 400 and "500" in r.json()["error"]["message"]


@pytest.mark.parametrize("beta", [0.0, 0.6, 1.0])
def test_batched_decode_cost_follows_beta(beta: float) -> None:
    """Two streams decoding together: each token costs (1 + beta) times a lone one. Checked on the mock's
    simulated clock (reported `timings`), which does not depend on how busy the test machine is."""
    st = MockState(time_scale=0.01, n_slots=2, batch_beta=beta)

    async def both(url: str) -> list[dict[str, float]]:
        async with httpx.AsyncClient(timeout=60) as c:

            async def one(tag: str) -> dict[str, float]:
                b = {"messages": [{"role": "user", "content": tag + " tok" * 10}], "max_tokens": 60, "stream": True}
                timings: dict[str, float] = {}
                async with c.stream("POST", f"{url}/v1/chat/completions", json=b) as r:
                    async for line in r.aiter_lines():
                        if '"timings"' in line:
                            timings = json.loads(line[6:])["timings"]
                return timings

            return list(await asyncio.gather(one("aaa"), one("bbb")))

    with serve(create_mock(st)) as url:
        t = asyncio.run(both(url))
    n = t[0]["prompt_n"] + t[0]["cache_n"]
    alone = st.cost.decode_s(n, 60) * 1000
    other_prefill = st.cost.prefill_s(t[1]["prompt_n"], 0) * 1000
    first, second = sorted(timing["predicted_ms"] for timing in t)
    # the later request decodes batched for all 60 tokens
    assert first == pytest.approx(alone * (1 + beta), rel=0.04)
    # the earlier one also waits out the other's prefill chunk in one of its steps (prefill interference)
    assert second == pytest.approx(alone * (1 + beta) + other_prefill, rel=0.04)


def test_idle_time_is_not_caught_up_later() -> None:
    """After an idle gap the engine must restart its timeline; otherwise the next request would 'catch up'
    on the idle time and finish almost instantly."""
    st = MockState(time_scale=0.05)
    body = {"messages": [{"role": "user", "content": "tok " * 20}], "max_tokens": 40}
    with serve(create_mock(st)) as url:
        httpx.post(f"{url}/v1/chat/completions", json=body, timeout=30)
        time.sleep(0.5)  # idle: worth 10 simulated seconds
        t0 = time.perf_counter()
        r = httpx.post(f"{url}/v1/chat/completions", json=body, timeout=30)
        took = time.perf_counter() - t0
    modeled = (r.json()["timings"]["prompt_ms"] + r.json()["timings"]["predicted_ms"]) / 1000 * 0.05
    assert took >= 0.8 * modeled


@pytest.mark.anyio
async def test_hold_fairness_guard_hands_over_when_someone_waited_too_long() -> None:
    a = Admission(max_queue=4, hold_s=10, hold_max_wait_s=0.05)
    await a.acquire({"render": render(conv(1, 1))})
    starving = asyncio.create_task(a.acquire({"render": render(conv(2, 1))}))
    await asyncio.sleep(0.1)  # longer than hold_max_wait_s
    a.release(1.0)
    await asyncio.wait_for(starving, 1)  # handed over at once instead of holding for session 1
    assert not a.holding and a.hold_skipped == 1
