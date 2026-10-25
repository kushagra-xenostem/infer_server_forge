"""Admission control for a server with a fixed number of slots.

Requests beyond `max_inflight` wait in a bounded queue. When the queue is full, the request is rejected at
once with an estimated `Retry-After`, instead of piling up behind a single slot until every client times out.

Which waiter gets a freed slot is a policy:
- `FIFO`: arrival order.
- `CacheAffinity`: the waiter whose prompt shares the longest prefix with the context resident in the KV
  cache (the last admitted request), with a starvation guard.
- Anticipatory hold (`hold_s > 0`, works with either policy): when a request finishes, keep the slot for up
  to `hold_s` for the *same* session's next turn, which then reuses its cache instead of paying a cold
  prefill. The idea is anticipatory disk scheduling: idling briefly can beat switching. A fairness guard
  (`hold_max_wait_s`) skips the hold while anyone has waited too long; without it other sessions starve.
"""

import math
import time
import asyncio
from collections import deque
from typing import Any, Protocol
from dataclasses import dataclass, field
from collections.abc import Awaitable, Callable


class QueueFull(Exception):
    def __init__(self, retry_after_s: int) -> None:
        super().__init__(f"queue full, retry after {retry_after_s} s")
        self.retry_after_s = retry_after_s


class QueueTimeout(Exception):
    def __init__(self, waited_s: float, retry_after_s: int) -> None:
        super().__init__(f"waited {waited_s:.1f} s in the queue without getting a slot")
        self.retry_after_s = retry_after_s


class ClientGone(Exception):
    pass


def render(body: dict[str, Any]) -> str:
    """Cheap stand-in for the chat template: role + content of every message, in order."""
    out = []
    for m in body.get("messages", []):
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        out.append(f"{m.get('role')}:{c}\n")
    return "".join(out)


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n  # binary search on the first mismatch: slicing compares in C
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def same_session(prompt: str, resident: str, share: float = 0.9) -> bool:
    """Does `prompt` continue the resident conversation (it re-sends almost all of it as a prefix)?"""
    return bool(resident) and common_prefix(prompt, resident) >= share * len(resident)


@dataclass
class Waiter:
    info: dict[str, Any]  # what a policy may look at: "render" = the rendered prompt
    enqueued: float
    future: asyncio.Future[None]


class Policy(Protocol):
    name: str

    def pick(self, waiters: list[Waiter], resident: str) -> int: ...


class FIFO:
    name = "fifo"

    def pick(self, waiters: list[Waiter], resident: str) -> int:
        return 0


@dataclass
class CacheAffinity:
    max_wait_s: float = 120.0  # a waiter older than this goes first, whatever its prefix
    max_skips: int = 3  # ... and so does one that has been passed over this many times
    name: str = "affinity"
    skips: dict[int, int] = field(default_factory=dict)

    def pick(self, waiters: list[Waiter], resident: str) -> int:
        now = time.monotonic()
        chosen = None
        for i, w in enumerate(waiters):  # starvation guard, oldest first
            if now - w.enqueued > self.max_wait_s or self.skips.get(id(w), 0) >= self.max_skips:
                chosen = i
                break
        if chosen is None:
            best = -1
            for i, w in enumerate(waiters):  # ties keep arrival order
                n = common_prefix(w.info.get("render", ""), resident)
                if n > best:
                    chosen, best = i, n
        assert chosen is not None
        for w in waiters[:chosen]:
            self.skips[id(w)] = self.skips.get(id(w), 0) + 1
        self.skips.pop(id(waiters[chosen]), None)
        return chosen


@dataclass
class Admission:
    max_inflight: int = 1
    max_queue: int = 4
    queue_timeout_s: float = 300.0
    policy: Policy = field(default_factory=FIFO)
    hold_s: float = 0.0  # anticipatory hold; 0 = off
    # Fairness guard for the hold: no hold while any waiter has already waited longer than this. Without it,
    # a session with short think times keeps winning the slot back and others starve (measured: E-SCHED).
    hold_max_wait_s: float = 60.0
    prior_service_s: float = 10.0  # Retry-After estimate before any request has finished
    window: int = 50
    inflight: int = 0
    waiters: list[Waiter] = field(default_factory=list)
    service_times: deque[float] = field(default_factory=deque)
    resident: str = ""  # rendered prompt of the last admitted request (what the KV cache holds)
    holding: bool = False
    _hold_timer: asyncio.TimerHandle | None = None
    rejected: int = 0
    timed_out: int = 0
    admitted: int = 0
    hold_hits: int = 0  # the held-for session came back in time
    hold_expired: int = 0  # it did not: the slot sat idle for hold_s for nothing
    hold_skipped: int = 0  # no hold because of the fairness guard

    def median_service_s(self) -> float:
        if not self.service_times:
            return self.prior_service_s
        s = sorted(self.service_times)
        return s[len(s) // 2]

    def retry_after_s(self) -> int:
        """Rough time until a newly queued request would start: work ahead of it / slots."""
        ahead = len(self.waiters) + self.inflight
        return max(1, math.ceil(ahead * self.median_service_s() / self.max_inflight))

    async def acquire(
        self,
        info: dict[str, Any] | None = None,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
        poll_s: float = 0.25,
    ) -> float:
        """Wait for a slot; returns the seconds spent queued. Raises QueueFull, QueueTimeout or ClientGone."""
        info = info or {}
        prompt = info.get("render", "")
        if self.holding and same_session(prompt, self.resident):
            self._stop_hold()  # the slot was being kept for exactly this session
            self.hold_hits += 1
            self._admit(prompt)
            return 0.0
        if self.inflight < self.max_inflight and not self.waiters and not self.holding:
            self.inflight += 1
            self._admit(prompt)
            return 0.0
        if len(self.waiters) >= self.max_queue:
            self.rejected += 1
            raise QueueFull(self.retry_after_s())
        t0 = time.monotonic()
        w = Waiter(info, t0, asyncio.get_running_loop().create_future())
        self.waiters.append(w)
        try:
            while True:
                left = self.queue_timeout_s - (time.monotonic() - t0)
                if left <= 0:
                    self.timed_out += 1
                    raise QueueTimeout(time.monotonic() - t0, self.retry_after_s())
                # asyncio.wait, not wait_for: on 3.11 wait_for swallows a cancellation that races the handoff
                await asyncio.wait({w.future}, timeout=min(poll_s, left))
                if w.future.done():
                    break
                if is_disconnected is not None and await is_disconnected():
                    raise ClientGone()
        except BaseException:
            if w.future.done() and not w.future.cancelled():
                self._release_slot()  # the slot was handed to us just as we gave up: pass it on
            elif w in self.waiters:
                self.waiters.remove(w)
            raise
        self._admit(prompt)
        return time.monotonic() - t0

    def _admit(self, prompt: str) -> None:
        self.admitted += 1
        self.resident = prompt

    def release(self, service_s: float | None = None) -> None:
        if service_s is not None:
            self.service_times.append(service_s)
            while len(self.service_times) > self.window:
                self.service_times.popleft()
        oldest = min((w.enqueued for w in self.waiters), default=None)
        starving = oldest is not None and time.monotonic() - oldest > self.hold_max_wait_s
        if starving and self.hold_s > 0:
            self.hold_skipped += 1  # fairness guard: someone has waited too long, hand over now
        if self.hold_s > 0 and self.resident and not self.holding and not starving:
            for i, w in enumerate(self.waiters):  # the same session is already queued: no need to wait
                if same_session(w.info.get("render", ""), self.resident) and not w.future.done():
                    self.waiters.pop(i)
                    w.future.set_result(None)
                    return
            self.holding = True
            self._hold_timer = asyncio.get_running_loop().call_later(self.hold_s, self._end_hold)
            return
        self._release_slot()

    def _stop_hold(self) -> None:
        self.holding = False
        if self._hold_timer is not None:
            self._hold_timer.cancel()
            self._hold_timer = None

    def _end_hold(self) -> None:
        self._stop_hold()
        self.hold_expired += 1
        self._release_slot()

    def _release_slot(self) -> None:
        # hand the slot straight to the next waiter (inflight stays the same), or free it
        while self.waiters:
            w = self.waiters.pop(self.policy.pick(self.waiters, self.resident))
            if not w.future.done():
                w.future.set_result(None)
                return
        self.inflight -= 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "inflight": self.inflight,
            "queued": len(self.waiters),
            "max_inflight": self.max_inflight,
            "max_queue": self.max_queue,
            "policy": self.policy.name,
            "hold_s": self.hold_s,
            "holding": self.holding,
            "admitted": self.admitted,
            "rejected": self.rejected,
            "timed_out": self.timed_out,
            "hold_hits": self.hold_hits,
            "hold_expired": self.hold_expired,
            "hold_skipped": self.hold_skipped,
            "hold_max_wait_s": self.hold_max_wait_s,
            "median_service_s": round(self.median_service_s(), 2),
        }
