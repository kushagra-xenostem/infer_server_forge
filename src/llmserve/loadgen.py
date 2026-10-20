"""Load generator: concurrent simulated client sessions against an OpenAI-compatible endpoint.

Two workloads:
- `agent`: Cline-like sessions. Each turn re-sends the whole conversation plus a new user message, so the
  prefix cache decides whether a turn costs ~1 s or a full cold prefill. New tokens per turn and output
  lengths are drawn from the real Cline log; think time between turns is drawn from a range.
- `chat`: independent short prompts with short answers.

Every session starts with a unique tag, so sessions never share a cached prefix. Prompts ask the model to
keep counting, so a real model generates until `max_tokens`, like the mock does.

`time_scale` (<1 for the accelerated mock) shrinks think time; durations in the results are divided by it
so they read in simulated seconds. A Retry-After header is real seconds (HTTP allows only whole seconds),
so it is slept as is and recorded divided by `time_scale`.
"""

import time
import json
import httpx
import random
import asyncio
from typing import Any
from pathlib import Path
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

# 3-letter words + a space = 4 characters: one token for the mock (4 chars/token) and ~one for a real tokenizer
VOCAB = (
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old",
    "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use",
)  # fmt: skip
COUNT_PROMPT = "\nContinue counting upward from 1, one number per line. Never stop and never add anything else."


@dataclass(frozen=True)
class Workload:
    kind: str = "agent"  # agent | chat
    turns: int = 6
    first_context: tuple[int, int] = (3000, 12000)  # filler tokens in the first prompt (agent)
    new_tokens: Sequence[int] = (190,)  # sampled per turn (agent: appended; chat: whole prompt)
    gen_tokens: Sequence[int] = (121,)  # sampled per request
    max_gen: int = 1500
    max_context: int = 60000  # a session restarts its conversation before exceeding this
    think_s: tuple[float, float] = (2.0, 20.0)

    @classmethod
    def from_log(cls, path: str | Path, **overrides: Any) -> "Workload":
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        new = tuple(int(r["prompt_n"]) for r in rows if r.get("prompt_n"))
        gen = tuple(int(r["gen_n"]) for r in rows if r.get("gen_n"))
        return cls(new_tokens=new, gen_tokens=gen, **overrides)


@dataclass
class Result:
    session: int
    turn: int
    start_s: float
    status: int
    ttft_s: float | None = None
    e2e_s: float | None = None
    gen_tokens: int | None = None
    prompt_new: int | None = None
    prompt_cached: int | None = None
    prefill_s: float | None = None
    gen_tps: float | None = None
    retry_after_s: float | None = None
    attempts: int = 1
    error: str | None = None


def filler(rng: random.Random, n_tokens: int) -> str:
    return " ".join(rng.choice(VOCAB) for _ in range(max(1, n_tokens)))


@dataclass
class Target:
    base_url: str  # .../v1
    api_key: str | None = None
    model: str = "default"
    timeout_s: float = 900.0


@dataclass
class Runner:
    target: Target
    workload: Workload
    sessions: int
    seed: int = 0
    time_scale: float = 1.0
    retry_on_429: int = 3  # a well-behaved client honours Retry-After this many times, then gives up
    results: list[Result] = field(default_factory=list)
    t0: float = 0.0
    wall_s: float = 0.0

    async def run(self) -> list[Result]:
        headers = {"Authorization": f"Bearer {self.target.api_key}"} if self.target.api_key else {}
        limits = httpx.Limits(max_connections=self.sessions + 4, max_keepalive_connections=self.sessions + 4)
        async with httpx.AsyncClient(
            base_url=self.target.base_url.rstrip("/"),
            headers=headers,
            limits=limits,
            timeout=httpx.Timeout(self.target.timeout_s, connect=15),
        ) as client:
            self.t0 = time.perf_counter()
            await asyncio.gather(*(self._session(client, i) for i in range(self.sessions)))
            self.wall_s = (time.perf_counter() - self.t0) / self.time_scale
        return self.results

    async def _session(self, client: httpx.AsyncClient, sid: int) -> None:
        rng = random.Random(f"{self.seed}-{sid}")
        w = self.workload
        await asyncio.sleep(rng.uniform(0, w.think_s[0]) * self.time_scale)  # staggered start
        messages = self._fresh(rng, sid)
        context = 0
        for turn in range(w.turns):
            gen = min(w.max_gen, rng.choice(w.gen_tokens))
            if w.kind == "chat":
                messages = [
                    {
                        "role": "user",
                        "content": f"[session {sid} turn {turn}] "
                        + filler(rng, rng.choice(w.new_tokens))
                        + COUNT_PROMPT,
                    }
                ]
            else:
                new = min(rng.choice(w.new_tokens), w.max_context // 2)  # one huge turn must still fit
                if context + new + gen > w.max_context:
                    messages, context = self._fresh(rng, sid), 0
                messages.append({"role": "user", "content": filler(rng, new) + COUNT_PROMPT})
            res, reply = await self._request(client, sid, turn, messages, gen)
            self.results.append(res)
            if w.kind == "agent":
                messages.append({"role": "assistant", "content": reply})
                context = (res.prompt_new or 0) + (res.prompt_cached or 0) + (res.gen_tokens or 0)
            await asyncio.sleep(rng.uniform(*w.think_s) * self.time_scale)

    def _fresh(self, rng: random.Random, sid: int) -> list[dict[str, Any]]:
        if self.workload.kind == "chat":
            return []
        lo, hi = self.workload.first_context
        system = f"[session {sid} seed {self.seed}] You are a counting assistant. Reference notes: " + filler(
            rng, rng.randint(lo, hi)
        )
        return [{"role": "system", "content": system}]

    async def _request(
        self, client: httpx.AsyncClient, sid: int, turn: int, messages: list[dict[str, Any]], gen: int
    ) -> tuple[Result, str]:
        body = {
            "model": self.target.model,
            "messages": messages,
            "max_tokens": gen,
            "stream": True,
            "temperature": 0,
            "stream_options": {"include_usage": True},
        }
        attempts = 0
        while True:
            attempts += 1
            start = time.perf_counter()
            res = Result(
                session=sid, turn=turn, start_s=(start - self.t0) / self.time_scale, status=0, attempts=attempts
            )
            reply: list[str] = []
            try:
                async with client.stream("POST", "/chat/completions", json=body) as r:
                    res.status = r.status_code
                    if r.status_code != 200:
                        res.error = (await r.aread()).decode(errors="replace")[:200]
                        if "retry-after" in r.headers:
                            res.retry_after_s = float(r.headers["retry-after"]) / self.time_scale
                    else:
                        await self._consume(r, res, reply, start)
            except httpx.HTTPError as e:
                res.error = f"{type(e).__name__}: {e}"[:200]
            res.e2e_s = (time.perf_counter() - start) / self.time_scale
            if res.status == 429 and res.retry_after_s is not None and attempts <= self.retry_on_429:
                self.results.append(res)  # keep the rejected attempt in the record
                await asyncio.sleep(res.retry_after_s * self.time_scale)  # back to real seconds
                continue
            return res, "".join(reply)

    async def _consume(self, r: httpx.Response, res: Result, reply: list[str], start: float) -> None:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                ev = json.loads(line[6:])
            except ValueError:
                continue
            if ev.get("error"):
                res.error = str(ev["error"])[:200]
            for c in ev.get("choices") or []:
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    if res.ttft_s is None:
                        res.ttft_s = (time.perf_counter() - start) / self.time_scale
                    reply.append(piece)
            if ev.get("usage"):
                res.gen_tokens = ev["usage"].get("completion_tokens")
            t = ev.get("timings")
            if t:
                res.prompt_new, res.prompt_cached = t.get("prompt_n"), t.get("cache_n")
                res.prefill_s = round(t.get("prompt_ms", 0) / 1000, 3)
                res.gen_tps = round(t.get("predicted_per_second", 0), 2)
                res.gen_tokens = res.gen_tokens or t.get("predicted_n")


def percentile(xs: Sequence[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def summarize(results: Sequence[Result], wall_s: float) -> dict[str, Any]:
    """Per (session, turn) only the last attempt counts; rejected attempts are counted separately."""
    last: dict[tuple[int, int], Result] = {}
    first_start: dict[tuple[int, int], float] = {}
    for r in results:
        key = (r.session, r.turn)
        first_start[key] = min(first_start.get(key, r.start_s), r.start_s)
        if key not in last or r.attempts > last[key].attempts:
            last[key] = r
    final = list(last.values())
    # what a retrying client lives through: first attempt sent -> final answer done
    turn = [
        r.start_s + r.e2e_s - first_start[(r.session, r.turn)]
        for r in final
        if r.status == 200 and r.error is None and r.e2e_s is not None
    ]
    ok = [r for r in final if r.status == 200 and r.error is None]
    gave_up = [r for r in final if r.status == 429]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    e2e = [r.e2e_s for r in ok if r.e2e_s is not None]
    tokens = sum(r.gen_tokens or 0 for r in ok)

    def pct(xs: Sequence[float]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for k, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
            v = percentile(xs, q)
            out[k] = None if v is None else round(v, 2)
        return out

    return {
        "requests": len(final),
        "ok": len(ok),
        "rejected_429": sum(r.status == 429 for r in results),
        "gave_up": len(gave_up),
        "errors": len(final) - len(ok) - len(gave_up),
        "ttft_s": pct(ttft),
        "e2e_s": pct(e2e),
        "turn_s": pct(turn),
        "wall_s": round(wall_s, 1),
        "req_per_min": round(60 * len(ok) / wall_s, 2) if wall_s else None,
        "gen_tok_per_s": round(tokens / wall_s, 2) if wall_s else None,
    }


def write_jsonl(path: str | Path, results: Sequence[Result], meta: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({**meta, **asdict(r)}) + "\n")
