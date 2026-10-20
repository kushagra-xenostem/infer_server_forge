"""Gateway between OpenAI-compatible clients (Cline, agents) and llama-server.

Client -> :8081 (this) -> :8080 llama-server. Streams responses through unchanged and writes one JSON line
per chat request: prompt/cached tokens, prefill time, TTFT, generation speed, tool use, finish reason,
errors.

Sampling: the client's sampling params are stripped, so the server's defaults (Qwen-recommended, set in
prod_server.sh) apply. Set GW_KEEP_SAMPLING=1 to pass them through instead.

Admission (GW_MAX_QUEUE set): at most GW_MAX_INFLIGHT requests reach llama-server, up to GW_MAX_QUEUE wait,
and the rest get 429 + Retry-After at once. GW_DEADLINE_S bounds queue wait + upstream time per request.

Auth: when GW_API_KEYS (comma-separated) is set, every route except /health needs `Authorization: Bearer <key>`.

Run: uvicorn llmserve.gateway:app --host <tailscale-ip> --port 8081
"""

import os
import json
import time
import httpx
import asyncio
from typing import Any
from pathlib import Path
from fastapi import FastAPI, Request
from collections import Counter, deque
from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from starlette.types import ASGIApp, Receive, Scope, Send
from fastapi.responses import JSONResponse, Response, StreamingResponse
from llmserve.admission import FIFO, Admission, CacheAffinity, ClientGone, Policy, QueueFull, QueueTimeout, render

SAMPLING = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repeat_penalty")
OPEN_PATHS = {"/health"}


@dataclass(frozen=True)
class Settings:
    upstream: str = "http://127.0.0.1:8080"
    log_path: str = "/data/projects/llm-serving/logs/gateway.jsonl"
    keep_sampling: bool = False
    api_keys: frozenset[str] = field(default_factory=frozenset)
    log_preview: bool = True  # last 160 chars of the user message; turn off before publishing logs
    connect_timeout_s: float = 10.0
    max_queue: int | None = None  # None: no admission control (the old behaviour)
    max_inflight: int = 1  # must match llama-server's -np
    queue_timeout_s: float = 300.0
    deadline_s: float | None = None  # total budget per request: queue wait + upstream
    queue_policy: str = "fifo"  # fifo | affinity
    hold_s: float = 0.0  # anticipatory hold for the session that just finished; 0 = off
    hold_max_wait_s: float = 60.0  # fairness guard for the hold

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        keys = frozenset(k.strip() for k in env.get("GW_API_KEYS", "").split(",") if k.strip())
        return cls(
            upstream=env.get("GW_UPSTREAM", cls.upstream),
            log_path=env.get("GW_LOG", cls.log_path),
            keep_sampling=bool(env.get("GW_KEEP_SAMPLING")),
            api_keys=keys,
            log_preview=env.get("GW_LOG_PREVIEW", "1") != "0",
            max_queue=int(env["GW_MAX_QUEUE"]) if env.get("GW_MAX_QUEUE") else None,
            max_inflight=int(env.get("GW_MAX_INFLIGHT", "1")),
            queue_timeout_s=float(env.get("GW_QUEUE_TIMEOUT_S", "300")),
            deadline_s=float(env["GW_DEADLINE_S"]) if env.get("GW_DEADLINE_S") else None,
            queue_policy=env.get("GW_QUEUE_POLICY", "fifo"),
            hold_s=float(env.get("GW_HOLD_S", "0")),
            hold_max_wait_s=float(env.get("GW_HOLD_MAX_WAIT_S", "60")),
        )


LOADING_RETRY_AFTER_S = 10  # a model (re)load takes ~1 min on the real server; poll a few times meanwhile


def retry_headers(status: int, upstream_headers: httpx.Headers) -> dict[str, str]:
    """llama-server answers 503 while the model loads, without Retry-After; tell clients when to come back."""
    if status in (429, 503):
        return {"Retry-After": upstream_headers.get("retry-after", str(LOADING_RETRY_AFTER_S))}
    return {}


def make_policy(name: str) -> Policy:
    if name == "fifo":
        return FIFO()
    if name == "affinity":
        return CacheAffinity()
    raise ValueError(f"unknown queue policy {name!r} (fifo | affinity)")


def error_body(message: str, type_: str, code: int) -> dict[str, Any]:
    return {"error": {"message": message, "type": type_, "code": code}}


def describe(body: dict[str, Any], preview: bool) -> dict[str, Any]:
    msgs = body.get("messages", [])
    last_user: dict[str, Any] = next((m for m in reversed(msgs) if m.get("role") == "user"), {})
    content = last_user.get("content", "")
    if isinstance(content, list):  # multimodal-style content parts
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    rec: dict[str, Any] = {
        "n_messages": len(msgs),
        "request_chars": len(json.dumps(msgs)),
        "has_tools": bool(body.get("tools")),
    }
    if preview:
        rec["last_user_preview"] = str(content)[-160:].replace("\n", " ")
    return rec


def timings_fields(t: dict[str, Any] | None, usage: dict[str, Any] | None) -> dict[str, Any]:
    t = t or {}
    return {
        "prompt_n": t.get("prompt_n"),
        "cached_n": t.get("cache_n"),
        "prefill_s": round(t["prompt_ms"] / 1000, 2) if "prompt_ms" in t else None,
        "prefill_tps": round(t.get("prompt_per_second", 0), 1),
        "gen_n": t.get("predicted_n") or (usage or {}).get("completion_tokens"),
        "gen_tps": round(t.get("predicted_per_second", 0), 1),
    }


class SSEStats:
    """Incremental parser for llama-server's SSE stream; collects what the log line needs."""

    def __init__(self) -> None:
        self.buf = ""
        self.finish: str | None = None
        self.timings: dict[str, Any] | None = None
        self.usage: dict[str, Any] | None = None
        self.tool_calls = 0
        self.error: str | None = None
        self.bad_lines = 0

    def feed(self, chunk: bytes) -> None:
        self.buf += chunk.decode(errors="replace")
        *lines, self.buf = self.buf.split("\n")
        for line in lines:
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            try:
                ev = json.loads(line[6:])
            except ValueError:
                self.bad_lines += 1
                continue
            if ev.get("error"):
                self.error = str(ev["error"])[:300]
            for c in ev.get("choices", []):
                self.finish = c.get("finish_reason") or self.finish
                for tc in (c.get("delta") or {}).get("tool_calls") or []:
                    if tc.get("id"):
                        self.tool_calls += 1
            self.timings = ev.get("timings") or self.timings
            self.usage = ev.get("usage") or self.usage


class BearerAuth:
    """Pure ASGI middleware. Starlette's BaseHTTPMiddleware wraps `receive`, which hides client disconnects
    from the handlers; the queue needs to see them to drop clients that gave up."""

    def __init__(self, app: ASGIApp, keys: frozenset[str]) -> None:
        self.app = app
        self.keys = keys

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] not in OPEN_PATHS:
            header = dict(scope["headers"]).get(b"authorization", b"").decode(errors="replace")
            token = header[7:] if header.lower().startswith("bearer ") else ""
            if token not in self.keys:
                resp = JSONResponse(error_body("invalid or missing API key", "invalid_api_key", 401), status_code=401)
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    cfg = settings or Settings.from_env()
    Path(cfg.log_path).parent.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.client = httpx.AsyncClient(
            base_url=cfg.upstream, transport=transport, timeout=httpx.Timeout(None, connect=cfg.connect_timeout_s)
        )
        yield
        await app.state.client.aclose()

    app = FastAPI(title="llm gateway", lifespan=lifespan)
    app.state.settings = cfg

    def write_log(rec: dict[str, Any]) -> None:
        with open(cfg.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    if cfg.api_keys:
        app.add_middleware(BearerAuth, keys=cfg.api_keys)

    adm: Admission | None = None
    if cfg.max_queue is not None:
        adm = Admission(
            max_inflight=cfg.max_inflight,
            max_queue=cfg.max_queue,
            queue_timeout_s=cfg.queue_timeout_s,
            policy=make_policy(cfg.queue_policy),
            hold_s=cfg.hold_s,
            hold_max_wait_s=cfg.hold_max_wait_s,
        )
    app.state.admission = adm
    recent: deque[dict[str, Any]] = deque(maxlen=200)
    statuses: Counter[int] = Counter()

    def finish(rec: dict[str, Any]) -> None:
        statuses[rec.get("status") or 0] += 1
        recent.append(rec)
        write_log(rec)

    def busy(rec: dict[str, Any], status: int, type_: str, msg: str, retry_after_s: int) -> Response:
        rec.update(status=status, error=type_)
        finish(rec)
        return JSONResponse(
            error_body(msg, type_, status), status_code=status, headers={"Retry-After": str(retry_after_s)}
        )

    @app.post("/v1/chat/completions")
    async def chat(req: Request) -> Response:
        client: httpx.AsyncClient = req.app.state.client
        try:
            body = await req.json()
        except ValueError:
            return JSONResponse(
                error_body("request body is not valid JSON", "invalid_request_error", 400), status_code=400
            )
        if not cfg.keep_sampling:
            for k in SAMPLING:
                body.pop(k, None)
        rec: dict[str, Any] = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **describe(body, cfg.log_preview),
            "stream": bool(body.get("stream")),
        }
        t0 = time.perf_counter()
        deadline = None if cfg.deadline_s is None else time.monotonic() + cfg.deadline_s

        # Admission: wait for the slot in a bounded queue, or get a fast 429 with Retry-After.
        if adm is not None:
            try:
                rec["queue_wait_s"] = round(await adm.acquire({"render": render(body)}, req.is_disconnected), 2)
            except QueueFull as e:
                return busy(rec, 429, "queue_full", "server busy: queue full", e.retry_after_s)
            except QueueTimeout as e:
                return busy(rec, 503, "queue_timeout", "server busy: timed out waiting for a slot", e.retry_after_s)
            except ClientGone:
                rec.update(status=499, error="client_disconnected_in_queue", wall_s=round(time.perf_counter() - t0, 2))
                finish(rec)
                return Response(status_code=499)
        t_admit = time.perf_counter()
        released = False

        def release(ok: bool) -> None:
            # only completed requests feed the service-time estimate behind Retry-After; fast failures
            # (502/504) would drag it towards zero (measured: Retry-After 1 s with 3 requests ahead)
            nonlocal released
            if adm is not None and not released:
                released = True
                adm.release(time.perf_counter() - t_admit if ok else None)

        def left() -> float | None:
            return None if deadline is None else max(0.0, deadline - time.monotonic())

        def fail(status: int, type_: str, msg: str, detail: str) -> Response:
            release(ok=False)
            rec.update(status=status, wall_s=round(time.perf_counter() - t0, 2), error=f"{type_}: {detail}"[:300])
            finish(rec)
            return JSONResponse(error_body(msg, type_, status), status_code=status)

        if not body.get("stream"):
            try:
                r = await asyncio.wait_for(client.post("/v1/chat/completions", json=body), left())
            except TimeoutError:
                return fail(504, "deadline_exceeded", "request deadline exceeded", f"{cfg.deadline_s} s")
            except httpx.TransportError as e:
                return fail(502, "upstream_unavailable", "upstream unavailable", repr(e))
            release(ok=r.status_code == 200)
            rec.update(status=r.status_code, wall_s=round(time.perf_counter() - t0, 2))
            if r.status_code == 200:
                j = r.json()
                ch = j["choices"][0]
                rec.update(
                    finish=ch.get("finish_reason"),
                    tool_calls=len(ch["message"].get("tool_calls") or []),
                    **timings_fields(j.get("timings"), j.get("usage")),
                )
            else:
                rec["error"] = r.text[:300]
            finish(rec)
            return Response(
                r.content,
                status_code=r.status_code,
                media_type="application/json",
                headers=retry_headers(r.status_code, r.headers),
            )

        # Open the upstream stream before answering, so upstream errors keep their real status code.
        try:
            upstream = await asyncio.wait_for(
                client.send(client.build_request("POST", "/v1/chat/completions", json=body), stream=True), left()
            )
        except TimeoutError:
            return fail(504, "deadline_exceeded", "request deadline exceeded", f"{cfg.deadline_s} s")
        except httpx.TransportError as e:
            return fail(502, "upstream_unavailable", "upstream unavailable", repr(e))
        if upstream.status_code != 200:
            content = await upstream.aread()
            await upstream.aclose()
            release(ok=False)
            rec.update(
                status=upstream.status_code,
                wall_s=round(time.perf_counter() - t0, 2),
                error=content.decode(errors="replace")[:300],
            )
            finish(rec)
            return Response(
                content,
                status_code=upstream.status_code,
                media_type="application/json",
                headers=retry_headers(upstream.status_code, upstream.headers),
            )

        async def gen() -> AsyncIterator[bytes]:
            stats = SSEStats()
            first: float | None = None
            try:
                async with asyncio.timeout_at(None if deadline is None else _loop_time(deadline)):
                    async for chunk in upstream.aiter_bytes():
                        if first is None:
                            first = time.perf_counter() - t0
                        stats.feed(chunk)
                        yield chunk
            except TimeoutError:
                stats.error = f"deadline_exceeded after {cfg.deadline_s} s"
                yield _sse_error("request deadline exceeded", "deadline_exceeded", 504)
            except httpx.TransportError as e:
                stats.error = f"upstream dropped mid-stream: {e!r}"[:300]
                yield _sse_error("upstream dropped the stream", "upstream_error", 502)
            except (asyncio.CancelledError, GeneratorExit):
                stats.error = "client_disconnected"  # Starlette stops the stream when the client goes away
                raise
            finally:
                await upstream.aclose()  # closing the upstream connection makes llama-server stop generating
                release(ok=stats.error is None and stats.finish is not None)
                rec.update(
                    status=200,
                    ttft_s=round(first, 2) if first else None,
                    wall_s=round(time.perf_counter() - t0, 2),
                    finish=stats.finish,
                    tool_calls=stats.tool_calls,
                    **timings_fields(stats.timings, stats.usage),
                )
                if stats.error:
                    rec["error"] = stats.error
                if stats.bad_lines:
                    rec["bad_sse_lines"] = stats.bad_lines
                finish(rec)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        ok = [r for r in recent if r.get("status") == 200]
        return {
            "admission": adm.snapshot() if adm is not None else None,
            "status_counts": {str(k): v for k, v in sorted(statuses.items())},
            "window": len(recent),
            "ttft_s": _percentiles([r["ttft_s"] for r in ok if r.get("ttft_s") is not None]),
            "wall_s": _percentiles([r["wall_s"] for r in ok if r.get("wall_s") is not None]),
            "queue_wait_s": _percentiles([r["queue_wait_s"] for r in recent if "queue_wait_s" in r]),
        }

    @app.get("/ready")
    async def ready(req: Request) -> Response:
        client: httpx.AsyncClient = req.app.state.client
        try:
            up = (await client.get("/health", timeout=5)).status_code
        except httpx.TransportError:
            up = 0
        body = {"ready": up == 200, "upstream_health": up, "admission": adm.snapshot() if adm is not None else None}
        return JSONResponse(body, status_code=200 if up == 200 else 503)

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def passthrough(path: str, req: Request) -> Response:
        client: httpx.AsyncClient = req.app.state.client
        try:
            r = await client.request(
                req.method,
                f"/{path}",
                content=await req.body(),
                headers={"content-type": req.headers.get("content-type", "application/json")},
            )
        except httpx.TransportError:
            return JSONResponse(error_body("upstream unavailable", "upstream_unavailable", 502), status_code=502)
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))

    return app


def _sse_error(message: str, type_: str, code: int) -> bytes:
    return f"data: {json.dumps(error_body(message, type_, code))}\n\n".encode()


def _loop_time(monotonic_deadline: float) -> float:
    # asyncio.timeout_at wants event-loop time
    return asyncio.get_running_loop().time() + (monotonic_deadline - time.monotonic())


def _percentiles(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"n": 0, "p50": None, "p95": None, "p99": None}
    s = sorted(xs)

    def q(p: float) -> float:
        return s[min(len(s) - 1, int(p * len(s)))]

    return {"n": len(s), "p50": q(0.5), "p95": q(0.95), "p99": q(0.99)}


_APP: FastAPI | None = None


def __getattr__(name: str) -> Any:
    # `uvicorn llmserve.gateway:app` builds the app from environment variables on first access, once
    # (uvicorn reads the attribute more than once; building per access made a second admission queue)
    global _APP
    if name == "app":
        if _APP is None:
            _APP = create_app()
        return _APP
    raise AttributeError(name)
