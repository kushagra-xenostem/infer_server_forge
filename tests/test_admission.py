import pytest
import random
import asyncio
from llmserve.admission import Admission, ClientGone, QueueFull, QueueTimeout

pytestmark = pytest.mark.anyio


async def test_admits_immediately_when_a_slot_is_free() -> None:
    a = Admission(max_inflight=1, max_queue=2)
    assert await a.acquire() == 0.0
    assert a.inflight == 1 and not a.waiters


async def test_full_queue_rejects_at_once_with_retry_after() -> None:
    a = Admission(max_inflight=1, max_queue=2, prior_service_s=10)
    await a.acquire()
    waiting = [asyncio.create_task(a.acquire()) for _ in range(2)]
    await asyncio.sleep(0)
    assert len(a.waiters) == 2
    with pytest.raises(QueueFull) as exc:
        await a.acquire()
    assert exc.value.retry_after_s == 30  # (2 queued + 1 in flight) * 10 s / 1 slot
    assert a.rejected == 1
    for _ in range(3):
        a.release()
    await asyncio.gather(*waiting)
    assert a.inflight == 0


async def test_fifo_order_and_slot_handoff() -> None:
    a = Admission(max_inflight=1, max_queue=5)
    await a.acquire()
    order: list[int] = []

    async def worker(i: int) -> None:
        await a.acquire()
        order.append(i)

    tasks = []
    for i in range(4):
        tasks.append(asyncio.create_task(worker(i)))
        await asyncio.sleep(0)  # enqueue in a known order
    for _ in range(4):
        a.release()
        await asyncio.sleep(0.01)
        assert a.inflight == 1  # the slot is handed over, never freed and re-grabbed
    await asyncio.gather(*tasks)
    assert order == [0, 1, 2, 3]


async def test_queue_timeout_removes_the_waiter() -> None:
    a = Admission(max_inflight=1, max_queue=2, queue_timeout_s=0.05)
    await a.acquire()
    with pytest.raises(QueueTimeout):
        await a.acquire(poll_s=0.01)
    assert not a.waiters and a.timed_out == 1 and a.inflight == 1


async def test_client_gone_while_queued() -> None:
    a = Admission(max_inflight=1, max_queue=2)
    await a.acquire()

    async def gone() -> bool:
        return True

    with pytest.raises(ClientGone):
        await a.acquire(is_disconnected=gone, poll_s=0.01)
    assert not a.waiters


async def test_cancelled_waiter_is_removed() -> None:
    a = Admission(max_inflight=1, max_queue=2)
    await a.acquire()
    t = asyncio.create_task(a.acquire())
    await asyncio.sleep(0.01)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert not a.waiters
    a.release()
    assert a.inflight == 0


async def test_slot_handed_to_a_waiter_that_is_cancelled_is_passed_on() -> None:
    a = Admission(max_inflight=1, max_queue=3)
    await a.acquire()
    first = asyncio.create_task(a.acquire())
    second = asyncio.create_task(a.acquire())
    await asyncio.sleep(0.01)
    a.release()  # hands the slot to `first`...
    first.cancel()  # ...which is cancelled before it runs
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(second, 1)  # the slot reached `second` instead of leaking
    assert a.inflight == 1
    a.release()
    assert a.inflight == 0


async def test_retry_after_uses_the_median_service_time() -> None:
    a = Admission(max_inflight=2, max_queue=4, prior_service_s=99)
    for s in (1.0, 2.0, 30.0):
        await a.acquire()
        a.release(s)
    assert a.median_service_s() == 2.0
    await a.acquire()
    await a.acquire()
    assert a.retry_after_s() == 2  # 2 in flight * 2 s / 2 slots


async def test_inflight_never_exceeds_the_limit_under_random_load() -> None:
    rng = random.Random(0)
    a = Admission(max_inflight=2, max_queue=6, queue_timeout_s=5)
    peak = 0
    outcomes = {"ok": 0, "full": 0}

    async def client() -> None:
        nonlocal peak
        await asyncio.sleep(rng.random() * 0.05)
        try:
            await a.acquire(poll_s=0.005)
        except QueueFull:
            outcomes["full"] += 1
            return
        peak = max(peak, a.inflight)
        await asyncio.sleep(rng.random() * 0.01)
        a.release(0.01)
        outcomes["ok"] += 1

    await asyncio.gather(*(client() for _ in range(200)))
    assert peak <= 2
    assert a.inflight == 0 and not a.waiters
    assert outcomes["ok"] + outcomes["full"] == 200 and outcomes["ok"] == a.admitted
