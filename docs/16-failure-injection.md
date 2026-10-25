# 16 · Failure injection and recovery

A server that works for one request has to keep working when requests fail around it. This page injects the
failures a single-slot LLM server meets in practice and measures two things for each: how long until the
failure is **visible** to the client (detect), and how long until a normal request **succeeds** again
(recover).

Two test beds, because the real server can only be reached through its public API (no shell access):

| Test bed | What can be injected | Source of the numbers |
|---|---|---|
| **Live** (the real Qwen3.6-35B server, black box) | anything a client can do: bad requests, oversized prompts, hanging up, bursts | `benchmarks/results/failures/live.jsonl` |
| **Simulator** (this repo's gateway in front of the calibrated mock, real-time speed) | server-side faults: a crash mid-stream, a hang, a cold start, a dead process, a restart | `benchmarks/results/failures/sim.jsonl` |

Reproduce: `python benchmarks/scripts/failure_injection.py live` and `... sim`. The behaviour behind every
row is also pinned by a test in `tests/test_failures.py`, `tests/test_backpressure.py` or
`tests/test_gateway.py`.

## Live server (2026-10-18)

| Fault | Client saw | Detect | Recovery |
|---|---|---|---|
| Wrong API key | 401 `authentication_error` | 0.12 s | n/a |
| Body is not JSON | 400 `Body must be JSON.` | 0.08 s | n/a |
| 80k-token prompt (context is 64k) | 400 with the token count in the message | 0.48 s | next request normal (TTFT 0.28 s) |
| Client hangs up after the first token of a 1,500-token answer | (client gone) | n/a | next request TTFT **1.6 s**, not ~94 s: the server **stops generating** for a client that left |
| 4 small requests at once | 4 × 200, no 429 | n/a | all done in 1.9 s |

From the load tests (docs/15): under the agent workload the deployed gateway turned 8 of 24 attempts at
4 sessions into `429 Server busy` with a fixed `Retry-After: 20`, but 4 tiny requests in a burst all went
through. So its admission control depends on load, not on a plain concurrency count.

## Simulator: server-side faults (this repo's gateway, `deadline 8 s`, `queue ≤ 2`)

| Fault | Client saw | Detect | Recovery |
|---|---|---|---|
| Upstream crashes mid-stream (connection reset after 5 tokens) | an SSE error event `upstream_error` (502) as the last event | 1.6 s | next request OK **2.1 s** later |
| Upstream hangs (stops answering, keeps the connection) | 504 `deadline_exceeded` | **8.0 s** (the deadline) | the gateway's slot is free at once (`inflight 0`). The hung server **still holds its own slot**: only a restart helps, so the watchdog below exists |
| Cold start: model "loading" for 25 s, client **honours `Retry-After: 10`** | 503 + `Retry-After` | first answer | first success after 32.2 s with **4 requests** |
| Same, client retries every 0.2 s | 503 | first answer | first success after 27.2 s with **117 requests** |
| Burst of 10 long requests, 1 slot + queue ≤ 2 | 7 × 429 + `Retry-After: 7` (each rejected in ≤ 32 ms) | immediate | the 3 admitted hit the 8 s deadline: their 200-token answers need ~13 s. **A deadline shorter than the service time cuts real work** |
| 3 streaming clients hang up after 3 lines | (clients gone) | n/a | gateway and upstream slots free after **5 ms**; all 3 upstream generations cancelled |
| Upstream process down (connection refused) | 502 `upstream_unavailable`, `/ready` 503 | 2.0 s (the OS connect retry) | after a restart on the same port: first request OK in 2.1 s |

## What the injections changed in the code

Every bug below was found by these injections or by the tests written for them, and each has a regression
test (details in [09](09-failures-and-lessons.md), #16–#20):

1. **Retry-After fell to 1 s under failures.** The estimate is the work ahead × the median service time, and
   fast 502/504 failures were counted as service times. Only completed requests count now; with 3 requests
   ahead the burst above gets `Retry-After: 7` (it got `1` before the fix).
2. **A queued client that hung up kept its place** (Starlette's `BaseHTTPMiddleware` hid the disconnect).
3. **A cancelled waiter could still take the slot** (Python 3.11 `asyncio.wait_for`).
4. **Upstream errors on streams reached clients as HTTP 200** in the old gateway.
5. **503 "Loading model" had no `Retry-After`.** llama-server sends none; the gateway now adds 10 s, and the
   cold-start rows show what that buys: 29× fewer requests for a 5 s later first answer.

## Production pieces (written, not deployed)

These need root on the server, so they are in the repo for the owner to install and are **NOT RUN on
hardware**:

- [`server/systemd/llm-serving.service`](../server/systemd/llm-serving.service): llama-server + gateway under
  systemd with `Restart=on-failure` and a start-rate limit (3 per 10 min), so a crash loop cannot hammer the GPU.
- [`server/watchdog.sh`](../server/watchdog.sh) + `llm-watchdog.timer`: every 30 s a real `/health` request
  with a 10 s timeout; after 3 failures in a row, `systemctl restart`. This is the recovery path for the
  hung-upstream row above, which a deadline alone cannot fix.
- Not covered by any of this: a GPU driver deadlock (09 #9) leaves the process in uninterruptible sleep;
  only a reboot clears it. The VRAM guard in `prod_server.sh` exists to stop that from happening.

To validate on hardware (owner): install the units, then `sudo kill -STOP $(pgrep -f build/bin/llama-server)`
to simulate a hang and time the watchdog's restart from `logs/watchdog.log`.
