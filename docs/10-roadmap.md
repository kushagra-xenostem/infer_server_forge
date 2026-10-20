# 10 · Status against the project brief, and what's next

**Research question:** how do quantization, CPU/GPU offload, KV-cache policy, context length, batching and scheduling change quality, latency and throughput on constrained hardware?

## Metrics

| Metric | Status | Where |
|---|---|---|
| TTFT | ✅ per request (gateway) and per turn (long-context tests) | [06](06-serving-gateway.md), [05](05-long-context.md) |
| Inter-token latency / tokens per second | ✅ prefill and decode tok/s for every config | [04](04-memory-engineering.md) |
| p50 / p95 latency | ✅ single-user (real Cline session) · ⏳ under concurrent load | [06](06-serving-gateway.md) |
| Requests/sec | ⏳ needs the load test | — |
| Peak VRAM / RAM | ✅ every config (VRAM after load, server RSS, swap) | [04](04-memory-engineering.md) |
| OOM rate | ⚠ observed real OOMs/hangs and fixed them; no deliberate injection yet | [09](09-failures-and-lessons.md) |
| Context length | ✅ 8k / 32k / 48k / 64k measured, 128k attempted | [05](05-long-context.md) |
| Coding-task success | ✅ HumanEval+, MBPP+, 20 hidden-test agent tasks | [03](03-model-selection.md) |

## Experiments

| Experiment | Status |
|---|---|
| Baseline single-request inference | ✅ |
| Compare quantization levels (weights Q8_0/Q4_K_M/IQ4_XS; KV f16/q8_0/q4_0/mixed) | ✅ |
| Vary context length | ✅ |
| CPU/GPU offload strategies (layer split, `-nkvo`, MoE expert offload, op offload) | ✅ |
| Compare batching strategies | ⚠ micro-batch sizes measured (ub 512/1024/2048); parallel slots / continuous batching ⏳ |
| Bounded concurrency + backpressure | ⏳ gateway is the place to add it |
| Stress memory | ✅ (found the cache-ram, SVM and 128k failure modes) |
| Inject OOM / timeouts deliberately | ⏳ |
| Compare against a reference runtime | ⏳ Ollama (vLLM does not target gfx1031) |

## Deliverables

| Deliverable | Status |
|---|---|
| FastAPI inference endpoint | ✅ [`server/gateway.py`](../server/gateway.py) |
| Repeatable benchmark suite | ✅ [`benchmarks/`](../benchmarks/) |
| Memory/latency plots | ✅ [`plots/`](../plots/) (regenerate with `tools/make_plots.py`) |
| Architecture diagram | ✅ README + [06](06-serving-gateway.md) |
| Benchmark CLI | ⚠ scripts exist; a single `bench` entry point ⏳ |
| Failure-injection tests | ⏳ |
| pytest suite for the gateway | ⏳ |
| OpenTelemetry traces | ⏳ (JSONL metrics today) |

## Next milestones
1. **Load test:** 1/2/4 concurrent simulated Cline sessions through the gateway → p50/p95/p99 TTFT, requests/sec, memory; compare `-np 1` with queueing against `-np 2`.
2. **Backpressure:** bounded queue in the gateway with 429 + `Retry-After` beyond N waiting requests.
3. **Failure injection:** kill the server mid-stream, oversized prompts, client disconnects; measure recovery.
4. **systemd service** with restart-on-failure and health checks.
5. **Reference runtime:** the same model through Ollama on the same box.
6. **128k retry** with more experts on the CPU, or after a RAM upgrade.
7. **More models and model types:** GLM-4.7-Flash and newer MoE releases; thinking models, image generation, vision, speech, embeddings. Full plan in [12-future-scope](12-future-scope.md).
