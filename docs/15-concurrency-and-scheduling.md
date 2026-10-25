# 15 · Concurrency, admission control and scheduling

The server is built for one Cline session (`-np 1`, one 64k KV cache). This page asks what happens when
several agents share it, and which serving choices help: a bounded queue with backpressure, smarter queue
order, holding the slot for a returning session, or splitting the context into two slots.

**Short answer.** With two or more agent sessions on one slot, every turn becomes a cold prefill, because
each session evicts the other's KV cache. Queue order cannot fix that. Two slots can: in simulation they cut
the median time to first token by ~10× at two sessions. A bounded queue does not add capacity, but it turns
unbounded waiting into fast, explicit `429`s.

![Two agent sessions on one slot](../plots/load_thrash-light.png)

## Method

The real server can only be reached through its public API, so two test beds are used, and the second is
checked against the first:

1. **Live, black box.** `llmserve load` drives the real endpoint with simulated agent sessions (1, 2 and 4
   at once; 4 turns each; ≤ 300 output tokens) and records llama.cpp's own `timings` per request.
2. **Simulator.** `llmserve sim` puts this repo's gateway in front of a mock llama-server that has a prefix
   cache per slot, a continuous-batching engine loop, and service times from a cost model fitted to real
   requests. Workload: Cline-like sessions (6 turns, 3k–12k-token start, per-turn new tokens and answer
   lengths drawn from the real 98-request Cline log, 2–20 s think time). It runs 4× faster than real time;
   all numbers are in simulated seconds.

### Is the simulator right? (per request, against the live server's own timings)

| Component | Model | n | Median error | Worst |
|---|---|---|---|---|
| Decode speed | fit to the Oct 6 Cline log | 28 | +1.5% | 12.1% |
| Cold prefill | fit to the Oct 6 Cline log | 16 | **+25.9%** | 42.1% |
| Cold prefill | refit on the live 1–2-session runs, tested on the 4-session run (`0.73 s + new/700 tok/s`) | 16 | **+2.9%** | 17.1% |
| Prefill with a cache hit | Oct 6 fit | 3 | −31.6% | 36.7% |

The server's prefill got ~20% faster since the Oct 6 log, so all experiments below use the live refit
(`--cost live`). The Oct 6 model is kept as a sensitivity check: with it every latency is higher, and every
conclusion below is the same (`benchmarks/results/load/REPORT.md`). Cache-hit prefills (n = 3) are the
weakest part of the model; they are short in absolute terms (1.5–3 s).

## E-LOAD: what a bounded queue buys

Hypothesis: without admission control, latency grows without bound as sessions are added; a bounded queue
keeps the latency of accepted requests stable and turns overload into fast rejections.

| Sessions | No admission: TTFT p50 / p95 | Queue ≤ 2: TTFT p50 / p95 | Queue ≤ 2: turns rejected for good | Throughput (tok/s), both |
|---|---|---|---|---|
| 1 | 5.3 / 18 s | 5.3 / 18 s | 0 | 3.0 |
| 2 | 33 / 53 s | 33 / 53 s | 0 | 3.4 |
| 4 | 110 / 182 s | 63 / 168 s | 0 of 24 (5 retried 429s) | 5.7 |
| 8 | 262 / 316 s | 66 / 151 s | **10 of 48** (67 retried 429s) | 6.5–6.9 |

Supported, with the cost made visible: at 8 sessions the accepted requests' p95 stays near 150 s instead
of 316 s, but 21% of turns are turned away after their client honours `Retry-After` three times. Throughput
is the same either way: the GPU is saturated, and a queue only decides *who* waits and *how they find out*.

![Admission control](../plots/load_admission-light.png)

Live, the deployed gateway already does a version of this: at 4 sessions it answered 8 of 24 attempts with
`429 Server busy` and a fixed `Retry-After: 20`.

## The cache thrash (live)

| Run | Turns after the first | Prompt tokens served from the KV cache | TTFT on those turns |
|---|---|---|---|
| 1 session | 3 | **91%** (~4.5–5k tokens per turn) | 2.4–3.5 s |
| 2 sessions | 6 | **0%** | 17–26 s |
| 4 sessions | 12 | **0%** | 31–53 s |

One slot has one KV cache. When session B's turn runs between two of session A's turns, A's next turn has
to prefill its whole conversation again. At ~700 tokens/s, a 10k-token conversation costs ~15 s before the
first token, every turn. The simulator reproduces this exactly (cached share 0.90 for one session, 0 for
two or more).

## E-SCHED: can queue order fix it?

Two policies were built and tested (one slot, queue ≤ 8, so nobody is rejected):

- **Cache affinity**: when the slot frees up, admit the waiter whose prompt shares the longest prefix with
  the resident cache.
- **Anticipatory hold**: when a request finishes, keep the slot idle for up to *h* seconds for the *same*
  session's next turn (the idea comes from anticipatory disk scheduling).

| One slot, 2 sessions | TTFT p50 | TTFT p95 | Cached share | Turns failed |
|---|---|---|---|---|
| FIFO | 33.5 s | 53 s | 0% | 0 |
| Cache affinity | 33.4 s | 53 s | 0% | 0 |
| Hold 15 s, no fairness guard | **2.6 s** | **159 s** | 90% | 0 (4 sessions: 1 of 24) |
| Hold 25 s, no fairness guard | 2.5 s | 169 s | 90% | 0 (4 sessions: **3 of 24** timed out after 300 s) |
| Hold 15 s + fairness guard (no hold once anyone waited > 60 s) | 5.4 s | 93 s | 70% | 0 |

- **Affinity changes nothing.** By the time the slot frees, the session whose context is cached is still
  "thinking" (writing its next turn), so it is never in the queue to be picked. Hypothesis rejected.
- **The hold works for the median and starves the tail.** A session with short think times keeps winning
  the slot back; the others wait up to 300 s and time out. The fairness guard bounds that, but over three
  seeds the guarded hold's median swings from 5 s to 59 s at two sessions and its p95 is worse than FIFO
  (111 vs 68 s). Rejected as a default.

## E-NP2: two slots

`-np 2` gives each session its own KV cache but halves the context per slot (2 × 32k) and makes the two
slots share the GPU. Batched decoding of two sequences costs more than one; how much (β) could not be
measured without restarting the real server, so it is swept: per-step cost = token time × (1 + β).

| 2 sessions | TTFT p50 | TTFT p95 | Cached share |
|---|---|---|---|
| 1 slot, 30k context cap (control: same cap as 2 slots) | 33.4 s | 53 s | 0% |
| 2 slots, β = 0.3 | 5.9 s | 29 s | 90% |
| 2 slots, β = 0.6 | 5.8 s | 29 s | 90% |
| 2 slots, β = 1.0 (no gain from batching at all) | 6.0 s | 29 s | 90% |

Two slots win at every β, including β = 1.0, where two sequences decode no faster together than one after
the other. The gain comes from **not repeating prefills**, not from batching. The one-slot control with the
same 30k cap shows that the smaller context is not what makes the difference.

### Robustness over three seeds (different think times, prompt and answer lengths)

| Config | Sessions | TTFT p50, mean [range] | TTFT p95 | Turn p95 | Cached share |
|---|---|---|---|---|---|
| 1 slot, FIFO | 2 | 32.4 [26.4–37.1] s | 68 s | 75 s | 0% |
| 1 slot, hold 15 s + guard | 2 | 23.3 [5.0–59.4] s | 111 s | 117 s | 61% |
| **2 slots (β = 0.6)** | 2 | **3.0 [1.4–5.8] s** | **27 s** | 58 s | 93% |
| 1 slot, FIFO | 4 | 102.5 [72.5–124.9] s | 180 s | 190 s | 0% |
| 1 slot, hold 15 s + guard | 4 | 103.5 [83.2–121.2] s | 196 s | 221 s | 19% |
| **2 slots (β = 0.6)** | 4 | **55.3 [49.5–63.4] s** | **109 s** | 186 s | 9% |

At four sessions, two slots still halve the median TTFT, but the turn-time tail is no better (186 s vs
190 s, with a wide 120–304 s range): four sessions on two caches thrash again.

![Scheduling vs slots](../plots/load_sched-light.png)

## Recommendation

| Use | Configuration |
|---|---|
| One agent (today's daily use) | keep `-np 1`, 64k: the full context, and its cache is never evicted |
| Two agents at once | `-np 2` (2 × 32k), clients capped at ~30k context: ~10× faster first token (simulated) |
| Shared by more people | add the bounded queue (`GW_MAX_QUEUE=2`, `GW_DEADLINE_S` above the p99 service time): latency stays bounded and overload becomes `429` + `Retry-After` |

**Not yet validated on hardware:** the two-slot result rests on the simulator and on an unmeasured β. The
check costs ~15 minutes: restart with `-np 2` (and `-c 65536`), then run
`llmserve load --sessions 2 --turns 4 --max-gen 300 --first-context 2000 6000 --think 2 10 --max-context 30000`
and compare with the one-slot row above.

## Limits

- One workload shape (Cline-like agent sessions). A chat workload with short prompts would shift the
  balance towards batching and away from caches.
- Small samples: 12–48 turns per configuration; the seed ranges above show how much that matters.
- The live runs share the server with its owner's own use; one cold first request (19 s for a 4.5k prompt,
  model warm-up) is excluded from the model fit and kept in the tables.

## Reproduce

```bash
make install
uv run llmserve load --sessions 1,2,4 --turns 4 --max-gen 300 --first-context 2000 6000 --think 2 10 --max-context 16000
bash benchmarks/scripts/run_load_experiments.sh all     # simulator, ~1 h
python benchmarks/scripts/load_report.py                 # -> benchmarks/results/load/REPORT.md
python tools/make_plots.py
```
