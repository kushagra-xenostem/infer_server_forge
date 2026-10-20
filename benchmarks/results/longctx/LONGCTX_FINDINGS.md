# Qwen3.6-35B-A3B UD-IQ4_XS: long-context results (2026-10-06)

Server: `-ngl 99 --n-cpu-moe 24 -fa on -ctk q8_0 -ctv q8_0 --no-host --no-repack --cache-ram 0 -rea off -b 2048 -ub <N>`.
Haystack: real llama.cpp C++ with planted functions (known answers) + an absent-function trap. Greedy, thinking off.

| Context | Load VRAM | Prompt tokens | First prefill | Follow-up (cache reuse) | Accuracy | Driver SVM warnings |
|---|---|---|---|---|---|---|
| 32k (ub 2048) | 9.1 GB | 29.2k | 62 s (474 t/s) | ~4.5 s | 4/5 (two-hop wrong) | 0 |
| 64k (ub 2048) | 9.6 GB | 57.8k | 131 s (442 t/s) | ~6 s | **5/5** | 0 |
| 64k session, 16 turns → 42k | 9.6 GB | +2.5k/turn | 5–8 s per turn | — | **16/16** (incl. recall of turn 1) | 0 |
| 128k (ncmoe 24, ub 1024) | **10.2 / 10.2 GB** | ~120k | **GPU hang** | — | — | — |

## 128k failure
- At ncmoe 24 the 128k server loaded with VRAM essentially full (10,212 / 10,224 MiB) and passed the short sanity check.
- During the 120k prefill the amdgpu driver deadlocked: a kernel hung-task in `kfd_restore_wq → amdgpu_amdkfd_gpuvm_restore_process_bos`, later `svm_range_set_attr`. GPU at 0%; llama-server stuck in D state (`drm_suballoc_new`), unkillable. Needs a reboot or a root GPU reset.
- Lesson: never run with VRAM above ~95% after load. 128k needs more experts on the CPU (ncmoe 26–27), which costs ~0.8 GB of extra RAM on the CPU side. Untested.

## Conclusions so far
- **64k is solid on current hardware:** accurate, ~2 min for a cold 58k prompt, 5–8 s per agent turn with cache reuse, memory stable, zero driver warnings.
- Prompt-cache reuse works with `--cache-ram 0`: only new tokens are processed each turn.
- Weakness: two-step reasoning across distant code failed once at 32k (passed at 64k), so it isn't fully reliable.
