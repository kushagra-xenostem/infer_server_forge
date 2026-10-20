# Benchmarks: how every number was produced

All scripts are the exact ones used, with network identifiers redacted. They assume the layout `/data/projects/llm-serving` (scripts, venv, logs), `/data/projects/llama.cpp`, and TheRock at `/data/rocm/therock-7.14/install`. Adjust `server/common.sh` for your paths.

## Harness principles
- **Verify gate per step:** preflight (no stale llama process, port free, VRAM idle) → load → known-answer sanity (`17×23 = 391`) → run → check sample counts → record with the exact server config → stop and confirm VRAM is released.
- **Sandboxed execution:** model-generated code runs in Docker with `--network none`, a read-only dependency mount, and memory/PID limits ([`Dockerfile.sandbox`](Dockerfile.sandbox), [`../server/sandbox.sh`](../server/sandbox.sh)).
- **Provenance:** every result row in [`results/summary.jsonl`](results/summary.jsonl) stores the model file hash prefix, the full llama-server command line and the llama.cpp commit.
- **Single-instance lock:** `flock` prevents two GPU pipelines from running at once.

## Suites

| Suite | Script | What it measures |
|---|---|---|
| HumanEval+ / MBPP+ | [`coding_suite.sh`](scripts/coding_suite.sh) + [`codegen_robust.py`](scripts/codegen_robust.py) | pass@1, greedy; generator verified identical to evalplus's own |
| Agent tasks (20) | [`agent_tasks.py`](scripts/agent_tasks.py), tasks in [`tasks/build_tasks.py`](tasks/build_tasks.py) | repo edits graded by hidden pytest tests; `validate` mode proves every task is solvable and initially failing |
| General reasoning | [`run_lmeval.sh`](scripts/run_lmeval.sh) | GSM8K, IFEval, MMLU-Pro via lm-eval against the server |
| Quantization fidelity | [`kld.sh`](scripts/kld.sh) | perplexity + KL divergence vs Q8_0 logits on wikitext-2 |
| KV-cache speed | `llama-bench -ctk/-ctv … -d 0,4096` | prefill/decode tok/s by cache type and depth |
| MoE offload probe | [`probe_30b.sh`](scripts/probe_30b.sh) | load time, VRAM, RSS, swap, prefill/decode speed per `--n-cpu-moe` |
| Long context | [`longctx.sh`](scripts/longctx.sh) + [`longctx.py`](scripts/longctx.py) | needle accuracy, prefill time, cache reuse, multi-turn session |
| Model bake-off | [`bakeoff.sh`](scripts/bakeoff.sh) | the coding suite for each candidate at its best config |

## Reproduce
```bash
python benchmarks/tasks/build_tasks.py                # build the 20 agent tasks
python benchmarks/scripts/agent_tasks.py validate     # expect 20/20 valid
CTX=65536 SERVE_ARGS="--n-cpu-moe 24 --no-host --no-repack -b 2048 -ub 2048 -rea off" \
  benchmarks/scripts/coding_suite.sh ~/models/gguf/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf q36_35b q8_0-q8_0
python tools/make_plots.py                            # regenerate charts from results/
```

## Results layout
| Path | Content |
|---|---|
| `results/summary.jsonl` | one row per (benchmark, model, KV config), with full server provenance |
| `results/REPORT.md` | generated tables |
| `results/evalplus/<run>/` | every generated solution + per-test pass/fail |
| `results/agent/<run>/responses.json` | every agent response, latency, token usage, pass/fail |
| `results/lmeval/` | lm-eval results + per-sample outputs |
| `results/kld/` | llama-perplexity logs |
| `results/longctx/` | needle and session results |
| `results/gateway/cline_session_metrics.jsonl` | 98 real Cline requests (metrics only; prompts removed) |
| `results/model_catalog.json` | 27 free models with sources; paid-model reference numbers |

Production model file: `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` from [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), 17,730,509,792 bytes, SHA-256 `649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3`.
