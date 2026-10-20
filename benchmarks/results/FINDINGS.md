# Qwen2.5-Coder-7B-Instruct on RX 6700: findings

Date: 2026-10-04/05 · llama.cpp b10660 (6c84c7d5d) · ROCm 7.14 (TheRock) · gfx1031.
Raw tables are in `REPORT.md`; every number traces back to `summary.jsonl`, which records the model file, server command line and commit.

## Verdict
The 7B is a strong **single-function coder** and a reliable **bug fixer**. It is a mediocre **instruction follower** and is weak on **broad knowledge**. Q4_K_M weights cost almost nothing. The q4_0 **K** cache breaks the model completely, and you were running it in production.

## 1. Coding accuracy matches the published BF16 model
| | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---|---|---|---|
| Published (BF16) | 88.4 | 84.1 | 83.5 | 71.7 |
| Q4_K_M, KV f16 | 89.0 | 84.1 | 83.6 | 73.3 |
| Q8_0, KV f16 | 90.2 | 85.4 | 84.4 | 73.5 |

Differences between quantizations are 1–3 problems, which is within noise.

## 2. Agent tasks: 14/20 (Q4_K_M), 13/20 (Q8_0)
- Bug fixes 7/7 and multi-file refactors 2/2: reliable.
- Fails the same 6 tasks under every working configuration, so these are capability limits, not quantization damage:
  - subtle spec details (bool-as-int, regex group alignment, inconsistent storage format)
  - algorithmic complexity: "O(n)" rewrites that still use `x in list`
  - stack-order logic
  - degenerate repetition when asked to write tests: it looped `max_len=69, 70, 71…` until hitting the token limit

## 3. General intelligence
| | Q4_K_M | Q8_0 |
|---|---|---|
| GSM8K (250, 8-shot CoT) | 79.6 ± 2.6 | 78.4 ± 2.6 |
| IFEval prompt-strict (541) | 58.2 ± 2.1 | 58.6 ± 2.1 |
| MMLU-Pro (140, 5-shot CoT) | 38.6 ± 4.0 | 40.0 ± 3.9 |

IFEval is the main weakness: about 4 in 10 prompts break at least one explicit format constraint. Plan for that in agent design: validate structured outputs and retry. It also invents historical facts freely.

## 4. KV cache: the important result
| K / V | HumanEval+ | Agent | PPL (Q8_0 = 10.14) | Same top token | Gen t/s @ 4k context |
|---|---|---|---|---|---|
| f16 / f16 | 84.1 | 14/20 | 10.33 | 94.2 % | 48.2 |
| **q8_0 / q8_0** | **83.5** | **14/20** | **10.34** | **94.2 %** | **44.4** |
| q8_0 / q4_0 | 82.9 | 14/20 | 10.35 | 93.9 % | **7.9** |
| q4_0 / q4_0 | **0.0** | **0/20** | **205.6** | 26.5 % | 46.4 |

- **A 4-bit K cache destroys Qwen2.5.** It happens on the CPU backend too and with Q8_0 weights, so it isn't a ROCm bug: Qwen2.5 has outlier key channels. The V cache tolerates 4-bit fine.
- The broken configuration also triggers **HTTP 500s** from llama-server ("output does not match the expected peg-native format"). A client that retries greedy requests will loop forever.
- **Mixed K/V types are about 6× slower at 4k context** (Flash Attention seems to have no GPU kernel for mixed types in this build). That rules them out for long agent contexts.

## Recommended serving configuration (7B coder)
```
llama-server -m qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf \
  -dev ROCm0 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --jinja ...
```
Replace `-ctk q4_0 -ctv q4_0` everywhere. Q4_K_M weights are fine; Q8_0 weights gain nothing measurable on these benchmarks.

## Operational notes
- `/data` is an HDD (~113 MB/s), so cold loads of 8 GB models take minutes. Health-check timeouts must allow for it.
- The server sanity gate (17×23 = 391) is what caught the q4_0 failure. Keep a known-answer check in every serving start-up.
