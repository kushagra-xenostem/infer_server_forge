# 13 · The model: Qwen3.6-35B-A3B

| | |
|---|---|
| **Name** | Qwen3.6-35B-A3B (Alibaba Qwen team) |
| **Released** | 2026-04-15 · [huggingface.co/Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| **License** | **Apache-2.0**: free for personal and commercial use |
| **Type** | Natively **multimodal** (text + images) mixture-of-experts LLM, with a thinking (reasoning) mode and tool calling |
| **Architecture** | `Qwen3_5MoeForConditionalGeneration` (llama.cpp arch `qwen35moe`) |

## Language model

| Property | Value | Why it matters on this box |
|---|---|---|
| Total parameters | **36.0B** (HF metadata; "35B" in the name) | too big for 10 GB VRAM as a dense model |
| Active parameters per token | **~3B** | only ~3B of compute per token, so the CPU can handle many experts |
| Layers | **40** | |
| Layer layout | `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` | only **10 of 40 layers use full attention**; 30 use linear attention with a fixed-size state |
| Hidden size | 2048 | |
| Attention | 16 query heads, **2 KV heads**, head dim 256 | tiny KV cache: ~11 KB per token at q8_0 → 64k context ≈ 0.7 GB |
| Experts | **256 per layer, 8 routed + 1 shared** active per token; expert FFN size 512 | expert weights are the bulk of the model and the part offloaded to the CPU |
| Vocabulary | 248,320 tokens | |
| Native context | **262,144 tokens** (extensible to ~1M) | run here at **64k** because of the VRAM budget |

## Vision encoder (in `mmproj-F16.gguf`)

| Property | Value |
|---|---|
| Parameters | **446.6M** (separate from the 36B language model) |
| Depth / width | 27 transformer blocks, hidden 1152, 16 heads, FFN 4304 |
| Patch size | 16×16 px, merged 2×2 → **one image token per 32×32 px** |
| Projector | `qwen3vl_merger`, projects to the language model's 2048-dim space |
| Base image size | 768 px; dynamic resolution (more pixels → more tokens) |

A 1920×1080 screenshot ≈ (1920/32) × (1080/32) ≈ **2,000 image tokens**; this server caps images at 1,024 tokens (`--image-max-tokens 1024`).

## The files on this server

| File | Size | Source | SHA-256 |
|---|---|---|---|
| `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` | 17,730,509,792 bytes (16.5 GiB) | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | `649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3` |
| `mmproj-F16.gguf` (vision) | 899,283,680 bytes (858 MiB) | same repo | `8971ee4f331ff0a4c609374f32984b3d4e6dc086c0aa35f1d637fad1829e887f` |

Both were downloaded with a parallel, resumable downloader and **verified against Hugging Face's published SHA-256** ([`pdl.py`](../benchmarks/scripts/pdl.py)).

**Quantization:** `UD-IQ4_XS` is Unsloth's "dynamic" ~4.25-bit format. Most tensors are 4-bit importance-weighted, and sensitive layers are kept at higher precision. The vision projector stays at F16 (it is small, and vision quality is sensitive to quantization).

## Performance on this box

| | Vendor (full precision, 200K ctx, own scaffold) | Measured here (UD-IQ4_XS, 64k) |
|---|---|---|
| SWE-bench Verified | 73.4 | not run locally |
| Terminal-Bench 2.0 | 51.5 | — |
| LiveCodeBench v6 | 80.4 | — |
| HumanEval / HumanEval+ | — | **91.5 / 87.2** |
| MBPP / MBPP+ | — | **91.8 / 77.5** |
| Agent tasks (20, hidden tests) | — | **18/20** |
| Long-context needles at 58k | — | **5/5** |
| Generation speed | — | **~17 tok/s** (median 15.7 in real Cline use) |
| Prompt processing | — | 440–475 tok/s |
| Image understanding | — | 5/5 facts; 1080p screenshot ≈ 70 s ([14-vision](14-vision.md)) |

## Recommended sampling (from the model card), as configured here

| Mode | temperature | top_p | top_k | presence_penalty | How to start |
|---|---|---|---|---|---|
| **Non-thinking (default)** | 0.7 | 0.8 | 20 | 1.5 | `prod_server.sh start` |
| Thinking, precise coding | 0.6 | 0.95 | 20 | 0.0 | `prod_server.sh start think` (4,096-token reasoning budget) |

The gateway strips client sampling parameters so these values always apply.

Note: the model card advises keeping **≥128k context** to preserve full thinking-mode quality. At 64k, thinking mode works, but long reasoning traces compete with the code for context. That is one more reason to use non-thinking mode by default on this box.
