# Qwen3-Coder-30B-A3B Q4_K_S (16.25 GiB): getting it to run at a usable speed (2026-10-05)

Hardware: RX 6700 10 GB · 13.6 GiB RAM · 4-thread Ryzen 3 3200G · model on SSD.
Layout: experts 15.33 GiB (0.32 GiB per layer × 48 layers); attention, router and embeddings 0.92 GiB.

## Why it was ~1 tok/s before
- Old setup: `-ngl 18–24 -nkvo` put whole layers, attention included, plus the full KV cache in system RAM.
- With all experts on the CPU (15 GiB) the weights don't fit in RAM: the box swaps, re-reads ~30 GB from SSD, and runs at ~2 t/s.
- Default pinned host buffers and CPU weight repacking add copies that exceed RAM and the amdgpu system-memory mapping cap (ttm pages_limit ≈ 6.8 GiB) → "SVM mapping failed" → ROCm error/abort.

## Working recipe
```
llama-server -m <30B gguf> -dev ROCm0 -ngl 99 --n-cpu-moe <N> -fa on \
  -ctk q8_0 -ctv q8_0 -c <CTX> -b 2048 -ub <1024|2048> -t 4 --jinja \
  --no-host --no-repack          # both required on this box
```
Op offload must stay ON: prefill drops from 183 to 52 t/s without it.

| ncmoe | ctx | VRAM | prefill t/s (prompt size) | gen t/s (shallow → deep) | status |
|---|---|---|---|---|---|
| 26 | 8k (ub 2048) | 9.1 GB | 183 (1.4k) | 19 | stable |
| 30 | 32k (ub 2048) | 9.1 GB | **432 (20.5k)** | 15 → **10.5** @20k | **stable** |
| 32 | 48k (ub 1024) | 9.1 GB | 243 (41.9k) | 16 → 7.9 @42k | works; 2,045 SVM warnings (edge) |
| 35 | 64k | — | — | — | crash: GPU OOM + SVM cap on 45k prompt |
| 37 | 64k | — | — | — | crash on a 3k prompt (SVM cap) |

Ceiling: CPU-side experts must stay at or below ~10 GB (ncmoe ≤ ~32) because of the amdgpu mapping cap and the 13.6 GiB of RAM.
Long-context accuracy was NOT tested here. The haystack probe had no correct answer, and the model hallucinated a function name instead of saying "not present".
