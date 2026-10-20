# 01 · Hardware

Every value below was read from the machine (`lscpu`, `/sys/class/dmi`, `lspci`, `lsblk`, `amd-smi`) unless marked *owner-reported*. Re-run [`tools/collect_env.sh`](../tools/collect_env.sh) to regenerate.

| Component | Part | Notes |
|---|---|---|
| CPU | **AMD Ryzen 3 3200G** (Zen+, 12 nm) | 4 cores / **4 threads** (no SMT), 3.6 GHz boost, 65 W TDP. Runs the MoE experts that don't fit in VRAM |
| iGPU | Radeon Vega 8 (`gfx902`) | Not used for inference. Reserves ~2 GB of system RAM (UMA frame buffer), which is why Linux sees 13.6 GiB instead of 16 |
| GPU | **Sapphire Radeon RX 6700 (non-XT), 10 GB GDDR6** | Navi 22, **`gfx1031`**, 36 CUs, PCIe ID `1002:73df`, subvendor `0x1da2` (Sapphire). Not on AMD's official ROCm support list ([02-gpu-stack](02-gpu-stack.md)) |
| RAM | 16 GB DDR4-2400 (*owner-reported speed*) | **13.6 GiB usable** (`MemTotal 14,236,460 kB`) |
| Motherboard | **MSI B450M PRO-VDH MAX (MS-7A38)** | BIOS dated 2025-04-08. The 3200G limits the GPU slot to PCIe 3.0 x8 |
| SSD | **WD Green 240 GB SATA** (WDS240G2G0B) | OS, venvs and **model files**. Expert weights are memory-mapped from here |
| HDD | **Toshiba 1 TB 7200 rpm** (DT01ABA100) | `/data`: ROCm (TheRock), llama.cpp, projects, results |
| OS | **Ubuntu 24.04.4 LTS (Noble)**, kernel 6.8.0-142-generic | In-kernel `amdgpu` driver + Ubuntu `linux-firmware` |

## Why this box is "resource-constrained"

| Resource | Budget | What it limits |
|---|---|---|
| VRAM | 10 GB (usable ≈ 9.5 GB; we keep load ≤ 95% after a driver hang at 100%) | Model weights on GPU + KV cache + compute buffers |
| System RAM | 13.6 GiB usable | CPU-side expert weights + OS + page cache |
| GPU↔RAM mapping cap | `ttm.pages_limit` ≈ **6.8 GiB** (kernel default, half of RAM) | How much host memory the GPU driver will map. Exceeding it caused `SVM mapping failed` crashes |
| CPU | 4 threads, dual-channel DDR4 | Speed of experts computed on the CPU; this is the decode bottleneck for MoE models |
| Disk | SATA SSD + HDD | Cold model loads: 8 GB from the HDD takes minutes, so models live on the SSD |

A common rule of thumb says you need ~24 GB of VRAM for a useful 30B-class coding model. This project runs a **36B-parameter MoE model at 64k context on 10 GB of VRAM** by deciding, tensor by tensor, what lives where. See [04-memory-engineering](04-memory-engineering.md).

## Memory map of the production configuration

```
RX 6700 VRAM (10 GB)                         System RAM (13.6 GiB)
┌──────────────────────────────────┐        ┌────────────────────────────────────┐
│ embeddings + all attention layers │        │ MoE experts of 24 of 40 layers      │
│ router/gating + norms             │        │ (memory-mapped from SSD, ~9-10 GB)  │
│ MoE experts of 16 of 40 layers    │        │                                     │
│ KV cache, 64k ctx, q8_0 (~0.7 GB) │        │ vision encoder 447M (~0.9 GB)       │
│                                   │        │ OS, gateway, VS Code server         │
│ compute buffers (ub 2048)         │        │ page cache                          │
│ ─────────────── 9.6 GB used ───── │        │ ── llama-server RSS 10.4-10.7 GB ── │
└──────────────────────────────────┘        └────────────────────────────────────┘
```
VRAM and RSS are measured; the split between the items inside each box is approximate (computed from GGUF tensor sizes).
