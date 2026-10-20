# Qwen2.5-Coder-7B-Instruct — accuracy & intelligence on RX 6700 (llama.cpp ROCm)

## Coding benchmarks (EvalPlus, greedy, pass@1 %)

| Config | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---|---|---|---|
| Published (Qwen report, BF16) | 88.4 | 84.1 | 83.5 | 71.7 |
| Q2.5-Coder-7B Q8_0, KV f16 | 90.2 | 85.4 | 84.4 | 73.5 |
| Q2.5-Coder-7B Q4_K_M, KV f16 | 89.0 | 84.1 | 83.6 | 73.3 |
| Q2.5-Coder-7B Q4_K_M, KV q8_0 | 88.4 | 83.5 | 85.2 | 73.5 |
| Q2.5-Coder-7B Q4_K_M, KV q8_0-q4_0 | 87.8 | 82.9 | 83.3 | 72.0 |
| Q2.5-Coder-7B Q4_K_M, KV q4_0 | 0.0 | 0.0 | — | — |
| Q3-Coder-30B-A3B Q4_K_S, KV q8_0-q8_0 | 93.9 | 89.6 | 88.1 | 74.9 |
| Qwen3.6-35B-A3B IQ4_XS, KV q8_0-q8_0 | 91.5 | 87.2 | 91.8 | 77.5 |
| Qwen3.5-9B Q4_K_M, KV q8_0-q8_0 | 87.2 | 83.5 | 77.2 | 63.8 |

## Agent task suite (20 hidden-test tasks)

| Config | Pass | Bugfix | Feature | Implement | Performance | Refactor | Write-tests | Truncated |
|---|---|---|---|---|---|---|---|---|
| Q2.5-Coder-7B Q8_0, KV f16 | **13/20** | 7/7 | 2/3 | 2/5 | 0/2 | 2/2 | 0/1 | 1 |
| Q2.5-Coder-7B Q4_K_M, KV f16 | **14/20** | 7/7 | 2/3 | 3/5 | 0/2 | 2/2 | 0/1 | 1 |
| Q2.5-Coder-7B Q4_K_M, KV q8_0 | **14/20** | 7/7 | 2/3 | 3/5 | 0/2 | 2/2 | 0/1 | 1 |
| Q2.5-Coder-7B Q4_K_M, KV q8_0-q4_0 | **14/20** | 7/7 | 2/3 | 3/5 | 0/2 | 2/2 | 0/1 | 1 |
| Q2.5-Coder-7B Q4_K_M, KV q4_0 | **0/20** | 0/7 | 0/3 | 0/5 | 0/2 | 0/2 | 0/1 | 2 |
| Q3-Coder-30B-A3B Q4_K_S, KV q8_0-q8_0 | **14/20** | 5/7 | 1/3 | 3/5 | 2/2 | 2/2 | 1/1 | 0 |
| Qwen3.6-35B-A3B IQ4_XS, KV q8_0-q8_0 | **18/20** | 7/7 | 3/3 | 4/5 | 2/2 | 2/2 | 0/1 | 0 |
| Qwen3.5-9B Q4_K_M, KV q8_0-q8_0 | **14/20** | 6/7 | 2/3 | 5/5 | 0/2 | 1/2 | 0/1 | 0 |

| Task | Q2.5-Coder-7B Q8_0/f16 | Q2.5-Coder-7B Q4_K_M/f16 | Q2.5-Coder-7B Q4_K_M/q8_0 | Q2.5-Coder-7B Q4_K_M/q8_0-q4_0 | Q2.5-Coder-7B Q4_K_M/q4_0 | Q3-Coder-30B-A3B Q4_K_S/q8_0-q8_0 | Qwen3.6-35B-A3B IQ4_XS/q8_0-q8_0 | Qwen3.5-9B Q4_K_M/q8_0-q8_0 |
|---|---|---|---|---|---|---|---|---|
| t01_pagination_off_by_one | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| t02_mutable_default | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| t03_traceback_fix | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| t04_inheritance_init | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| t05_timezone_bug | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| t06_sql_injection | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| t07_async_gather | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| t08_merge_intervals | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| t09_lru_cache | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| t10_log_parser | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| t11_word_wrap | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| t12_handler_validation | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| t13_cache_ttl | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| t14_cli_flags | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| t15_dataclass_validation | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| t16_rename_across_files | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| t17_signature_change | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| t18_quadratic_dupes | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| t19_deep_flatten | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| t20_write_tests | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |

## General reasoning & instruction following (lm-eval, %)

| Metric | q8_kvf16 | q4km_kvf16 |
|---|---|---|
| GSM8K (8-shot CoT, flexible) | 78.4 ± 2.6 | 79.6 ± 2.6 |
| IFEval instruction-level loose | 71.0 | 71.7 |
| IFEval prompt-level strict | 58.6 ± 2.1 | 58.2 ± 2.1 |
| MMLU-Pro (5-shot CoT, 10/subject) | 40.0 ± 3.9 | 38.6 ± 4.0 |

## Quantization fidelity vs Q8_0 (wikitext-2, 100×512 tokens)

| Config | PPL | Mean KLD | 99.9% KLD | Max KLD | Same top token | RMS Δp |
|---|---|---|---|---|---|---|
| q4km_kvf16-f16 | 10.331456 ± 0.186045 | 0.013480 ± 0.000134 | 0.227692 | 0.987052 | 94.208 ± 0.146 % | 3.118 ± 0.049 % |
| q4km_kvf16-q4_0 | 10.350006 ± 0.186452 | 0.015034 ± 0.000156 | 0.261607 | 1.027238 | 94.059 ± 0.148 % | 3.286 ± 0.052 % |
| q4km_kvq4_0-f16 | 203.468674 ± 4.646596 | 3.387302 ± 0.017825 | 16.186085 | 21.499884 | 26.451 ± 0.276 % | 47.507 ± 0.218 % |
| q4km_kvq4_0-q4_0 | 205.622403 ± 4.697592 | 3.398038 ± 0.017839 | 16.151030 | 22.541676 | 26.486 ± 0.276 % | 47.531 ± 0.218 % |
| q4km_kvq8_0-q4_0 | 10.353624 ± 0.186623 | 0.015247 ± 0.000154 | 0.262179 | 0.944701 | 93.878 ± 0.150 % | 3.321 ± 0.053 % |
| q4km_kvq8_0-q8_0 | 10.337695 ± 0.186291 | 0.013759 ± 0.000142 | 0.242459 | 1.021178 | 94.165 ± 0.147 % | 3.186 ± 0.056 % |
| q8_kvf16_base | 10.1441 +/- 0.18164 | — (base) | — | — | — | — |
| q8_kvq4_0-q4_0 | 189.955850 ± 4.330913 | 3.310139 ± 0.017671 | 16.001900 | 21.354530 | 27.110 ± 0.278 % | 47.164 ± 0.218 % |

## Speed by KV cache type (llama-bench, Q4_K_M, FA on, tokens/s)

| K / V | pp512 @ depth 0 | tg128 @ depth 0 | pp512 @ depth 4096 | tg128 @ depth 4096 |
|---|---|---|---|---|
| f16/f16 | 946.2 | 53.1 | 788.6 | 48.2 |
| q8_0/q8_0 | 938.8 | 51.1 | 783.1 | 44.4 |
| q8_0/q4_0 | 183.9 | 31.5 | 14.2 | 7.9 |
| f16/q4_0 | 159.8 | 31.0 | 7.0 | 4.7 |
| q4_0/q4_0 | 943.5 | 50.9 | 785.8 | 46.4 |
