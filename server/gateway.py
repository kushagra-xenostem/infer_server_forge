#!/usr/bin/env python3
"""Logging gateway between Cline (or any OpenAI client) and llama-server.

Client -> :8081 (this) -> :8080 llama-server. Streams responses through unchanged and writes one JSON line
per chat request to logs/gateway.jsonl: prompt/cached tokens, prefill time, TTFT, generation speed,
tool use, finish reason, errors.

Sampling: the client's sampling params are stripped, so the server's defaults (Qwen-recommended,
set in prod_server.sh) apply. Set GW_KEEP_SAMPLING=1 to pass them through instead.

Run: .venv/bin/uvicorn --app-dir scripts gateway:app --host 127.0.0.1 --port 8081
"""
import os
import json
import time
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAM = os.environ.get("GW_UPSTREAM", "http://127.0.0.1:8080")
LOG = os.environ.get("GW_LOG", "/data/projects/llm-serving/logs/gateway.jsonl")
SAMPLING = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repeat_penalty")

app = FastAPI()
client = httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(None, connect=10))


def write_log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def describe(body):
    msgs = body.get("messages", [])
    last_user = next((m for m in reversed(msgs) if m.get("role") == "user"), {})
    content = last_user.get("content", "")
    if isinstance(content, list):  # multimodal-style content parts
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return {"n_messages": len(msgs), "request_chars": len(json.dumps(msgs)), "has_tools": bool(body.get("tools")),
            "last_user_preview": content[-160:].replace("\n", " ")}


def timings_fields(t, usage):
    t = t or {}
    return {"prompt_n": t.get("prompt_n"), "cached_n": t.get("cache_n"),
            "prefill_s": round(t["prompt_ms"] / 1000, 2) if "prompt_ms" in t else None,
            "prefill_tps": round(t.get("prompt_per_second", 0), 1),
            "gen_n": t.get("predicted_n") or (usage or {}).get("completion_tokens"),
            "gen_tps": round(t.get("predicted_per_second", 0), 1)}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    if not os.environ.get("GW_KEEP_SAMPLING"):
        for k in SAMPLING:
            body.pop(k, None)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **describe(body), "stream": bool(body.get("stream"))}
    t0 = time.perf_counter()

    if not body.get("stream"):
        r = await client.post("/v1/chat/completions", json=body)
        rec.update(status=r.status_code, wall_s=round(time.perf_counter() - t0, 2))
        if r.status_code == 200:
            j = r.json()
            ch = j["choices"][0]
            rec.update(finish=ch.get("finish_reason"), tool_calls=len(ch["message"].get("tool_calls") or []),
                       **timings_fields(j.get("timings"), j.get("usage")))
        else:
            rec["error"] = r.text[:300]
        write_log(rec)
        return Response(r.content, status_code=r.status_code, media_type="application/json")

    async def gen():
        first, finish, timings, usage, tool_calls, buf = None, None, None, None, 0, ""
        status = None
        try:
            async with client.stream("POST", "/v1/chat/completions", json=body) as r:
                status = r.status_code
                async for chunk in r.aiter_bytes():
                    if first is None:
                        first = time.perf_counter() - t0
                    yield chunk
                    buf += chunk.decode(errors="replace")
                    *lines, buf = buf.split("\n")
                    for line in lines:
                        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                            continue
                        try:
                            ev = json.loads(line[6:])
                        except ValueError:
                            continue
                        if ev.get("error"):
                            rec["error"] = str(ev["error"])[:300]
                        for c in ev.get("choices", []):
                            finish = c.get("finish_reason") or finish
                            for tc in (c.get("delta") or {}).get("tool_calls") or []:
                                if tc.get("id"):
                                    tool_calls += 1
                        timings = ev.get("timings") or timings
                        usage = ev.get("usage") or usage
        finally:
            rec.update(status=status, ttft_s=round(first, 2) if first else None,
                       wall_s=round(time.perf_counter() - t0, 2), finish=finish, tool_calls=tool_calls,
                       **timings_fields(timings, usage))
            write_log(rec)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def passthrough(path: str, req: Request):
    r = await client.request(req.method, f"/{path}", content=await req.body(),
                             headers={"content-type": req.headers.get("content-type", "application/json")})
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
