# Model scan for a Cline coding agent on RX 6700 (10 GB) + 13.6 GiB RAM (2026-10-05)

Only free, open-weight models were considered. Every model below is Apache-2.0 unless noted.
Vendor scores come from the model cards: full precision with large contexts and the vendor's own agent scaffold. Use them for ranking, not as expected local results.

## Fit budget on this box (measured)
- Usable VRAM ≈ 9.5 GiB.
- CPU-side MoE expert weights ≤ ~9.6 GiB. Above that, the amdgpu mapping cap and RAM cause crashes or thrashing (see 30b_speed_probe.md).
- Required flags for CPU-offloaded experts: `--no-host --no-repack`, op offload ON.

## Candidates (configs read from each model's config.json)
| Model | Params (active) | Attention | KV/token q8_0 | 128k KV | SWE-bench Verified (vendor) | Fit |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (2026-04) | 36B MoE 256e/8 (~3B) | 10 full + 30 linear (Gated DeltaNet) | ~10.9 KB | ~1.3 GiB | **73.4** | UD-IQ4_XS 16.5 GiB → 128k plausible |
| Qwen3.5-35B-A3B (2026-02) | same arch | same | ~10.9 KB | ~1.3 GiB | 69.2 | superseded by 3.6 |
| Qwen3.5-9B (2026-02) | 9.7B dense | 8 full + 24 linear | ~17.4 KB | ~2.1 GiB | n/a (BFCL-V4 66.1, TAU2 79.1, LCB 65.6) | Q4_K_M 5.3 GiB, fully on GPU |
| GLM-4.7-Flash (2026-01) | 31B MoE 64e/4 + shared | MLA (kv_lora 512) | ~29 KB | ~3.5 GiB | 59.2 | bigger; possible later |
| Qwen3-Coder-30B-A3B (2025-07) | 30B MoE 128e/8 | 48 full GQA | ~52 KB | ~6.4 GiB | (not on card) | 32k max (measured) |
| gpt-oss-20b (2025-08) | 21B MoE 32e/4 (~3.6B) | 12 full + 12 sliding-128 | ~12 KB | ~1.5 GiB | 34.0 (per GLM card) | MXFP4 11.3 GiB, easy fit; weak agent score |
| Gemma-4-26B-A4B (2026-03) | 26B MoE (~4B) | 5 full + 25 sliding-1024 | ~21 KB | ~2.7 GiB | 17.4 (per Qwen3.6 card) | dropped: weak agent score |
| Qwen3-Coder-Next (2026-01) | ~80B MoE (qwen3_next, 512e/10) | hybrid | — | — | — | too large for this box |
| granite-4.2-8b/30b | dense GQA | full | large | — | — | poor long-context fit |

All of these architectures are supported by the local llama.cpp build (qwen35/qwen35moe, gemma4, gpt-oss, glm4moe), including the ROCm `gated_delta_net` kernel.

## Notes
- Qwen3.5/3.6 think by default. Disable per request with `chat_template_kwargs: {"enable_thinking": false}`. Test both modes, since thinking costs time on this hardware.
- Shortlist downloaded: Qwen3.6-35B-A3B UD-IQ4_XS, Qwen3.5-9B Q4_K_M (unsloth GGUFs, on SSD).
