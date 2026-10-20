"""A fake llama-server: OpenAI-compatible, N slots with prefix caches, calibrated timing, injectable faults.

It stands in for the real server wherever that cannot be touched (restarts, kills, `-np 2`) and makes
gateway tests and load experiments reproducible. Timing follows `CostModel`; `time_scale` shrinks every
sleep (tests use ~0.001) while the reported `timings` stay in simulated, unscaled milliseconds.

Slots (`n_slots`, like llama-server's `-np`): the context is split evenly across slots, each slot keeps its
own KV cache, and a request takes the free slot whose cache shares the longest prefix with it (or, below
50% similarity, the least recently used one), like llama.cpp's `--slot-prompt-similarity`. All slots share
one GPU through one engine loop, like llama-server's continuous batching: each step processes at most one
prompt chunk (`prefill_chunk` tokens) plus one token for every decoding slot, and costs
chunk time + `token_s * (1 + batch_beta * (k - 1))` for k decoding slots. One loop on one absolute timeline
also keeps sub-tick steps accurate. `batch_beta` is an assumption, not a measurement: measuring it needs a
server restart with `-np 2` (see EXECUTION_PLAN.md); experiments report a range of values.

Tokens are approximated as 4 characters, so the load generator builds prompts from 4-character words.
Faults are set at construction or at runtime with `POST /mock/faults` (JSON body of `Faults` fields).
"""

import time
import json
import asyncio
from typing import Any
from fastapi import FastAPI, Request
from llmserve.costmodel import CostModel
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field, fields
from fastapi.responses import JSONResponse, Response, StreamingResponse

CHARS_PER_TOKEN = 4
WORD = "tok "  # one simulated token
END = "<|end|>"


def n_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _content(c: Any) -> str:
    if isinstance(c, list):  # multimodal-style content parts
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c or "")


def prompt_text(body: dict[str, Any]) -> str:
    """A minimal chat template, so a generated reply sits in the cache exactly as the next prompt renders it
    (as with llama.cpp's real templates): the reply's tokens are reused on the next turn."""
    msgs = body.get("messages", [])
    return "".join(f"<|{m.get('role')}|>{_content(m.get('content'))}{END}" for m in msgs) + "<|assistant|>"


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


@dataclass
class Faults:
    error_status: int | None = None  # answer every request with this HTTP status
    hang: bool = False  # take the slot and never answer
    malformed_chunk: bool = False  # emit one broken SSE line mid-stream
    drop_after_tokens: int | None = None  # cut the connection after n streamed tokens
    loading_until: float = 0.0  # time.monotonic() before which the model is "loading" (503)
    extra_latency_s: float = 0.0  # added before prefill (unscaled)


@dataclass
class Slot:
    id: int
    cache_text: str = ""  # the prompt + output currently in this slot's KV cache
    busy: bool = False
    last_used: float = 0.0


@dataclass
class MockState:
    cost: CostModel = field(default_factory=CostModel)
    faults: Faults = field(default_factory=Faults)
    time_scale: float = 1.0
    ctx_size: int = 65536  # total, split across slots
    n_slots: int = 1
    batch_beta: float = 0.6  # decode slowdown per extra concurrently decoding slot (assumption)
    prefill_chunk: int = 2048  # llama-server -b/-ub
    default_gen_tokens: int = 64
    served: int = 0
    cancelled: int = 0  # streams stopped because the client went away
    last_body: dict[str, Any] = field(default_factory=dict)  # for tests: what the gateway forwarded
    slots: list[Slot] = field(init=False)

    def __post_init__(self) -> None:
        self.slots = [Slot(i) for i in range(self.n_slots)]

    @property
    def cache_text(self) -> str:  # single-slot shorthand used by tests
        return self.slots[0].cache_text

    @cache_text.setter
    def cache_text(self, value: str) -> None:
        self.slots[0].cache_text = value

    @property
    def slot_ctx(self) -> int:
        return self.ctx_size // self.n_slots


class Pacer:
    """Sleeps on an absolute timeline. asyncio fires any timer due within one clock tick (~15.6 ms on Windows)
    at once, so a plain per-token sleep of a few ms took no time at all; here each sleep targets
    `start + simulated time so far`, so short sleeps add up to the right total."""

    def __init__(self, time_scale: float) -> None:
        self.time_scale = time_scale
        self.target: float | None = None

    def resync(self) -> None:
        """After waiting for something else (the shared GPU), restart the timeline from now."""
        now = asyncio.get_running_loop().time()
        self.target = now if self.target is None else max(self.target, now)

    async def sleep(self, sim_s: float) -> None:
        loop = asyncio.get_running_loop()
        if self.target is None:
            self.target = loop.time()
        self.target += max(0.0, sim_s) * self.time_scale
        delay = self.target - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)


@dataclass
class Job:
    """One request inside the engine."""

    prompt_n: int
    gen_n: int
    chunks: list[float]  # simulated seconds per prompt chunk, still to run
    out: asyncio.Queue[tuple[str, Any]] = field(default_factory=asyncio.Queue)
    produced: int = 0
    cancelled: bool = False
    t_admit: float = 0.0  # engine simulated clock
    t_prefilled: float | None = None


class Engine:
    """The batched step loop shared by all slots (llama-server's `update_slots`)."""

    def __init__(self, st: MockState) -> None:
        self.st = st
        self.jobs: list[Job] = []
        self.wake = asyncio.Event()
        self.clock = 0.0  # simulated seconds of GPU work so far
        self.task: asyncio.Task[None] | None = None

    def submit(self, job: Job) -> None:
        job.t_admit = self.clock
        self.jobs.append(job)
        self.wake.set()
        if self.task is None or self.task.done():
            self.task = asyncio.get_running_loop().create_task(self.run())

    async def run(self) -> None:
        pace = Pacer(self.st.time_scale)
        while True:
            self.jobs = [j for j in self.jobs if not j.cancelled]
            if not self.jobs:
                self.wake.clear()
                await self.wake.wait()
                pace.resync()  # idle time is not work to catch up on
                continue
            prefilling = next((j for j in self.jobs if j.chunks), None)
            decoding = [j for j in self.jobs if not j.chunks and j.produced < j.gen_n]
            step = prefilling.chunks[0] if prefilling else 0.0
            k = len(decoding)
            if k:
                mean_token = sum(self.st.cost.token_s(j.prompt_n + j.produced) for j in decoding) / k
                step += mean_token * (1 + self.st.batch_beta * (k - 1))
            await pace.sleep(step)
            self.clock += step
            if prefilling is not None:
                prefilling.chunks.pop(0)
                if not prefilling.chunks:
                    prefilling.t_prefilled = self.clock
                    prefilling.out.put_nowait(("prefilled", None))
            for j in decoding:
                j.produced += 1
                j.out.put_nowait(("token", WORD))
                if j.produced >= j.gen_n:
                    j.out.put_nowait(("done", self.clock))
            self.jobs = [j for j in self.jobs if j.chunks or j.produced < j.gen_n]


def create_mock(state: MockState | None = None) -> FastAPI:
    st = state or MockState()
    free_slots = asyncio.Semaphore(st.n_slots)
    engines: list[Engine] = []  # created inside the server's event loop on first use
    app = FastAPI(title="mock llama-server")
    app.state.mock = st

    def engine() -> Engine:
        if not engines:
            engines.append(Engine(st))
        return engines[0]

    def loading() -> bool:
        return time.monotonic() < st.faults.loading_until

    async def take_slot(text: str) -> Slot:
        await free_slots.acquire()
        free = [s for s in st.slots if not s.busy]
        best = max(free, key=lambda s: common_prefix(text, s.cache_text))
        if common_prefix(text, best.cache_text) < 0.5 * len(text):
            best = min(free, key=lambda s: s.last_used)  # nothing similar: least recently used
        best.busy = True
        return best

    def give_slot(s: Slot) -> None:
        s.busy = False
        s.last_used = time.monotonic()
        free_slots.release()

    @app.get("/health")
    async def health() -> Response:
        if loading():
            return JSONResponse({"error": {"code": 503, "message": "Loading model"}}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "mock-model", "object": "model"}]}

    @app.get("/mock/state")
    async def mock_state() -> dict[str, Any]:
        return {
            "served": st.served,
            "cancelled": st.cancelled,
            "n_slots": st.n_slots,
            "busy": [s.busy for s in st.slots],
        }

    @app.post("/mock/faults")
    async def set_faults(req: Request) -> dict[str, Any]:
        body = await req.json()
        names = {f.name for f in fields(Faults)}
        unknown = set(body) - names
        if unknown:
            return {"error": f"unknown fault fields: {sorted(unknown)}"}
        for k, v in body.items():
            setattr(st.faults, k, v)
        return asdict(st.faults)

    @app.post("/v1/chat/completions")
    async def chat(req: Request) -> Response:
        body = await req.json()
        st.last_body = body
        if loading():
            return JSONResponse({"error": {"code": 503, "message": "Loading model"}}, status_code=503)
        if st.faults.error_status:
            return JSONResponse(
                {"error": {"code": st.faults.error_status, "message": "injected"}}, status_code=st.faults.error_status
            )
        text = prompt_text(body)
        prompt_n = n_tokens(text)
        gen_n = int(body.get("max_tokens") or st.default_gen_tokens)
        if prompt_n + gen_n > st.slot_ctx:
            msg = f"the request exceeds the available context size ({prompt_n + gen_n} > {st.slot_ctx})"
            return JSONResponse(
                {"error": {"code": 400, "message": msg, "type": "exceed_context_size_error"}}, status_code=400
            )
        if not body.get("stream"):
            content: list[str] = []
            final: dict[str, Any] = {}

            async def collect() -> None:
                async for kind, payload in _serve(text, prompt_n, gen_n):
                    if kind == "token":
                        content.append(payload)
                    elif kind == "final":
                        final.update(payload)

            # like llama-server, stop working for a non-streaming client that hung up
            task = asyncio.ensure_future(collect())
            while not task.done():
                await asyncio.wait({task}, timeout=0.25)
                if not task.done() and await req.is_disconnected():
                    task.cancel()
                    st.cancelled += 1
                    return Response(status_code=499)
            task.result()
            return JSONResponse(_completion("".join(content), final["timings"], gen_n))
        return StreamingResponse(_stream(text, prompt_n, gen_n), media_type="text/event-stream")

    async def _stream(text: str, prompt_n: int, gen_n: int) -> AsyncIterator[bytes]:
        try:
            async for kind, payload in _serve(text, prompt_n, gen_n):
                if kind == "token":
                    yield _sse({"choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}]})
                elif kind == "raw":
                    yield payload
                else:
                    yield _sse(payload)
                    yield b"data: [DONE]\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            st.cancelled += 1  # the client (the gateway) closed the connection: like llama-server, stop
            raise

    async def _serve(text: str, prompt_n: int, gen_n: int) -> AsyncIterator[tuple[str, Any]]:
        """Take a slot, prefill, decode; yields ("token", str) / ("raw", bytes) / ("final", dict)."""
        slot = await take_slot(text)
        job: Job | None = None
        try:
            if st.faults.hang:
                await asyncio.Event().wait()  # never set: holds the slot until the client goes away
            cached = min(prompt_n, common_prefix(text, slot.cache_text) // CHARS_PER_TOKEN)
            new = prompt_n - cached
            pre = st.faults.extra_latency_s + st.cost.prefill_s(new, cached)
            n_chunks = max(1, -(-new // st.prefill_chunk))
            job = Job(prompt_n=prompt_n, gen_n=gen_n, chunks=[pre / n_chunks] * n_chunks)
            engine().submit(job)
            i = 0
            while True:
                kind, payload = await job.out.get()
                if kind == "prefilled":
                    slot.cache_text = text  # the prompt is in the KV cache now, even if decoding is cut short
                elif kind == "token":
                    if st.faults.drop_after_tokens is not None and i >= st.faults.drop_after_tokens:
                        raise ConnectionResetError("injected: connection dropped mid-stream")
                    if st.faults.malformed_chunk and i == gen_n // 2:
                        yield "raw", b'data: {"choices": [{"delta": {"content": \n\n'
                    yield "token", payload
                    slot.cache_text += payload
                    i += 1
                elif kind == "done":
                    break
            slot.cache_text += END
            st.served += 1
            assert job.t_prefilled is not None
            prefill_s = job.t_prefilled - job.t_admit  # includes waiting behind other slots' work
            decode_s = payload - job.t_prefilled
            yield (
                "final",
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                    "usage": _usage(prompt_n, gen_n),
                    "timings": _timings(prompt_n, cached, prefill_s, gen_n, decode_s),
                },
            )
        finally:
            if job is not None:
                job.cancelled = True  # no-op when finished; stops the engine working for a gone client
            give_slot(slot)

    return app


def _sse(obj: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def _usage(prompt_n: int, gen_n: int) -> dict[str, int]:
    return {"prompt_tokens": prompt_n, "completion_tokens": gen_n, "total_tokens": prompt_n + gen_n}


def _timings(prompt_n: int, cached: int, pre_s: float, gen_n: int, dec_s: float) -> dict[str, Any]:
    new = prompt_n - cached
    return {
        "prompt_n": new,
        "cache_n": cached,
        "prompt_ms": round(pre_s * 1000, 1),
        "prompt_per_second": new / pre_s if pre_s else 0.0,
        "predicted_n": gen_n,
        "predicted_ms": round(dec_s * 1000, 1),
        "predicted_per_second": gen_n / dec_s if dec_s else 0.0,
    }


def _completion(content: str, timings: dict[str, Any], gen_n: int) -> dict[str, Any]:
    prompt_n = timings["prompt_n"] + timings["cache_n"]
    return {
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "length"}],
        "usage": _usage(prompt_n, gen_n),
        "timings": timings,
    }


app = create_mock()  # `uvicorn llmserve.mock_upstream:app`
