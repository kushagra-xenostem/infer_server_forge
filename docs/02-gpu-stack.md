# 02 · Making an "unsupported" GPU work with ROCm

## The problem

AMD's ROCm compute stack officially supports a short list of GPUs. In the RDNA2 consumer generation it targets **`gfx1030`** (RX 6800/6900). The **RX 6700 is `gfx1031`** and is *not* on the list: prebuilt ROCm libraries and PyTorch wheels ship no `gfx1031` kernels ([community guide](https://insiderllm.com/guides/rocm-not-detecting-gpu-amd-fix/), [example write-up](https://dev.to/lionthehoon/how-i-made-my-amd-rx6750xt-work-for-machine-learning-despite-rocm-limitations-553a)).

Typical symptoms on this card:
- `hipErrorNoBinaryForGpu` / **`invalid device function`** (PyTorch)
- llama.cpp: **`device kernel image is invalid`**
- the GPU is detected, but every kernel launch fails

## Approach 1: impersonate a supported GPU (early 2026)

The widespread workaround tells the ROCm runtime to treat the card as `gfx1030`:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

`gfx1030` and `gfx1031` share the RDNA2 ISA, so most `gfx1030` kernels run. This machine used that route first:
- ROCm **5.6** via AMD's `amdgpu-install` (with the `amdgpu-dkms` package), a PyTorch venv, and image-generation tools (ComfyUI, Fooocus)
- the override lived in `~/.bashrc`, alongside `PATH`/`LD_LIBRARY_PATH` entries for `/opt/rocm`

Drawbacks: an unofficial path with no `gfx1031`-tuned libraries, fragile across ROCm and PyTorch versions, and the old ROCm 5.6 packages were built for Ubuntu 22.04, not this 24.04 system.

## Approach 2 (final): native `gfx1031` with TheRock + a targeted llama.cpp build

**[TheRock](https://github.com/ROCm/TheRock)** is AMD's open build system for ROCm. Its nightly "dist" tarballs are built per GPU family, and the **`gfx103X-all`** family includes `gfx1031`. That gives native kernels without impersonation.

### 1. Install TheRock (self-contained, no system packages)
```bash
# illustrative: download the nightly dist tarball for the gfx103X family from TheRock's releases,
# then unpack it into a self-contained prefix (the build used here: 7.14.0a20260624)
mkdir -p /data/rocm/therock-7.14/install
tar xzf therock-dist-linux-gfx103X-all-7.14.0a20260624.tar.gz -C /data/rocm/therock-7.14/install
```
Result: ROCm **7.14.0**, HIP **7.14.60850**, AMD clang **23.0.0git** (ROCm llvm-project fork), hipBLAS/rocBLAS/hipBLASLt, `amd-smi`, all under one directory.

### 2. An environment script that keeps the stacks from mixing
[`gpu-bringup/the-rock-env.sh`](../gpu-bringup/the-rock-env.sh) puts TheRock first on `PATH`/`LD_LIBRARY_PATH` and **unsets `HSA_OVERRIDE_GFX_VERSION`, `ROCM_PATH`, `HIP_PATH`, `HIP_PLATFORM`**, so nothing falls back to the old `/opt/rocm` or the impersonation hack.

### 3. Prove the toolchain can build and run a `gfx1031` kernel
[`gpu-bringup/hip_validation.cpp`](../gpu-bringup/hip_validation.cpp) compiled with `--offload-arch=gfx1031`:
```
GPU: AMD Radeon RX 6700   Arch: gfx1031   Launch: no error   Sync: no error
Effective memory throughput: 242 GB/s     Max numerical error: 0     HIP validation: PASS
```
The first attempt was compiled for `gfx1030` and **failed on this card**. That is exactly the impersonation gap.

### 4. Build llama.cpp for `gfx1031` against TheRock
The shell history shows a series of attempts: first `-DAMDGPU_TARGETS=gfx1030`, then fights with CMake picking up headers from `/opt/rocm` and `/usr/include/hip`. The working configuration:
```bash
cd /data/projects/llama.cpp
cmake -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1031 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/data/rocm/therock-7.14/install/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/data/rocm/therock-7.14/install \
  -DHIP_ROOT_DIR=/data/rocm/therock-7.14/install \
  -DCMAKE_IGNORE_PATH=/opt/rocm
cmake --build build -j$(nproc)
```
- llama.cpp `0.3.0-dev`, build **10660**, commit **`6c84c7d5d`** (2026-08-27), GNU 13.3.0 host compiler
- verified with `ldd`: `libamdhip64.so.7`, `libhipblas.so.3`, `librocblas.so.5`, `libhipblaslt.so.1` all resolve to TheRock, not `/opt/rocm`
- `llama-cli --list-devices` → `ROCm0: AMD Radeon RX 6700 (10224 MiB)`

### 5. First benchmark (Qwen2.5-3B-Instruct Q4_K_M, `llama-bench`, 5 runs)

| Config | Prompt (pp512) | Generation (tg128) |
|---|---|---|
| GPU, all layers, Flash Attention off | 1818.6 ± 78.4 t/s | 92.8 ± 0.3 t/s |
| **GPU, all layers, Flash Attention on** | **1978.4 ± 55.1 t/s** | **97.5 ± 0.4 t/s** |
| CPU only (`-ngl 0`) control | 853.8 ± 110.9 t/s | 15.4 ± 0.0 t/s |

The 6× generation gap against the CPU-only control proves the work is really on the GPU.

### 6. Cleanup (Oct 2026)
Once the new stack was proven, the old one was removed so the two could never collide: 72 ROCm 5.6 packages (including `amdgpu-dkms` and its firmware overrides), `/opt/rocm-5.6.1`, AMD's apt sources, the `HSA_OVERRIDE` lines in `.bashrc`, and the broken Python 3.10 venv. Each package removal was **simulated first**. A naive version filter matched `liblzma5`/`xz-utils` (version 5.6.x), which would have broken `apt` itself; the final list was version-tag exact. After the reboot: in-kernel `amdgpu` + Ubuntu `linux-firmware` (all 12 `navy_flounder_*` blobs present), zero GPU errors, HIP validation PASS.

## PyTorch on this card

| Wheel | Result |
|---|---|
| `torch 2.10.0+rocm7.0` | kernel test failed: `invalid device function` (no gfx1031 code) |
| `torch 2.12.0+rocm7.14.0a20260612` (TheRock wheels, `rocm-sdk-libraries-gfx103X-all`) | works; used for Transformers experiments |

The production inference path does **not** use PyTorch. llama.cpp's HIP backend is lighter, supports GGUF quantization and CPU/GPU tensor placement, and was the only practical way to run 30B+ models here.

## Monitoring
```bash
amd-smi                 # GPU summary
watch -n 0.5 amd-smi monitor -g 0
```
