# 09 · Failures, postmortems and lessons

Real systems fail in specific ways. These are the ones this project hit, with root cause and fix. Several were found only because each experiment step ran a **verify gate** (known-answer sanity check, sample counts, VRAM checks, sandboxed execution).

| # | Symptom | Root cause | Fix / lesson |
|---|---|---|---|
| 1 | `device kernel image is invalid`; HIP probe fails | Code compiled for `gfx1030` on a `gfx1031` card | Build natively for `gfx1031` with TheRock ([02](02-gpu-stack.md)) |
| 2 | 30B MoE at ~0.5–1 tok/s | Partial `-ngl` moved attention to the CPU; KV in RAM; weights > RAM → swap | Attention on GPU, experts on CPU (`--n-cpu-moe`) |
| 3 | `ROCm error` + kernel `SVM mapping failed, exceeds resident system memory limit` | Pinned host buffers for 15 GB of CPU weights exceeded the amdgpu mapping cap (~6.8 GiB) | `--no-host`; keep CPU-side experts ≤ ~10 GB |
| 4 | Swap fills during load, 30 GB read from SSD for a 17 GB model | CPU weight repacking duplicated the memory-mapped experts | `--no-repack` |
| 5 | Model answers "17×23?" with *"I'm sorry, but I don't know what"* | **4-bit K cache**: Qwen keys have outlier channels; reproduces on CPU too | `-ctk q8_0`; sanity gate at every start |
| 6 | Benchmark hangs forever | Garbled output → llama-server HTTP 500 → evalplus retried the same greedy request in a loop | Robust generator that scores server errors as failures |
| 7 | Mixed K/V cache 6× slower at 4k context | No Flash Attention GPU kernel for mixed types in this build | Use matching types (q8_0/q8_0) |
| 8 | 3-hour stall, swap 100%, 100 s timeouts | `--cache-ram` default (8 GiB) stored old prompts in RAM, evicting expert pages | `--cache-ram 0` |
| 9 | **GPU driver deadlock** at 128k; process stuck in `D` state; reboot required | VRAM at 10,212 / 10,224 MiB; eviction/restore path deadlocked during a 120k prefill | VRAM guard: refuse to serve above 95% after load |
| 10 | Two benchmark runs killed each other's servers | A second agent session launched the same pipeline concurrently | `flock` single-instance lock on every GPU pipeline |
| 11 | `pkill -f pattern` killed the shell running it | Pattern matched the shell's own command line | Anchor patterns to the binary path; kill by PID |
| 12 | Cleanup filter would have removed `liblzma5`/`xz-utils` | "5.6" in a version string matched xz 5.6.x as well as ROCm 5.6 | Always simulate (`apt-get -s`) and match exact repo version tags |
| 13 | HF downloads stalled at 11 MB / 180 KB/s | Per-connection throttling + API rate limits (429) | Parallel range downloader with resume + SHA-256 verification ([`pdl.py`](../benchmarks/scripts/pdl.py)) |
| 14 | One benchmark answer empty → whole model rejected | Verify gate too strict (model rambled into the token limit once) | Allow ≤ 5% empty (scored as failures); still catches systematic breakage |
| 15 | Server gone after closing a terminal | Started from an interactive session that ended | Roadmap: systemd service with restart-on-failure |

## Principles that came out of it
1. **Gate every step.** A fixed known-answer prompt caught a configuration that scored 0% on every benchmark.
2. **Measure memory, not just speed.** Three of the worst failures (#3, #4, #8) were memory accounting problems that showed up first as slowness.
3. **Keep headroom.** 100% VRAM "works" until it hangs the kernel driver.
4. **Treat vendor numbers as claims.** Re-measure on your hardware, with your quantization and context.
5. **Simulate destructive operations first.**
