#!/bin/bash
# Shared paths and helpers for the llm-serving experiments. Source, don't execute.

export PROJ=/data/projects/llm-serving
export LLAMA=/data/projects/llama.cpp
export PORT=${PORT:-8080}

export M_Q4KM=$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct-GGUF/snapshots/13fb94bfda8c8cf22497dc57b78f391a9acb426a/qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf
export M_Q8=/data/models/gguf/qwen2.5-coder-7b-instruct-q8_0.gguf

# TheRock ROCm environment (sets PATH/LD_LIBRARY_PATH); keep our venv on PATH.
rock_env() {
    export VIRTUAL_ENV=$PROJ/.venv
    source /data/projects/rocm-tests/scripts/the-rock-env.sh
    export HIP_VISIBLE_DEVICES=0
    export ROCR_VISIBLE_DEVICES=0
}

# VRAM used on GPU 0 in MiB.
vram_used_mib() {
    (rock_env; amd-smi metric -g 0 --mem-usage --json 2>/dev/null) |
        python3 -c 'import json,sys
d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d
d=d.get("gpu_data",[d])[0] if "gpu_data" in d else d
u=d["mem_usage"]["used_vram"]; print(int(float(u["value"] if isinstance(u,dict) else u)))'
}

# Fail if anything could skew a run: stale llama procs, busy port, VRAM in use.
preflight() {
    local ok=0
    if pgrep -f '^[^ ]*build/bin/llama-' >/dev/null; then
        echo "PREFLIGHT FAIL: stale llama process:"; pgrep -af '^[^ ]*build/bin/llama-'; ok=1
    fi
    if ss -ltn | grep -q ":$PORT "; then echo "PREFLIGHT FAIL: port $PORT busy"; ok=1; fi
    local v; v=$(vram_used_mib)
    if [ "${v:-99999}" -gt 300 ]; then echo "PREFLIGHT FAIL: VRAM used ${v} MiB"; ok=1; fi
    local avail; avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    echo "preflight: vram_used=${v}MiB ram_avail=${avail}MiB disk_free=$(df -h /data | awk 'NR==2{print $4}')"
    return $ok
}

# Stop servers we started and confirm VRAM returned to idle.
cleanup_gpu() {
    pkill -f '^[^ ]*build/bin/llama-' 2>/dev/null
    for _ in $(seq 1 20); do pgrep -f '^[^ ]*build/bin/llama-' >/dev/null || break; sleep 1; done
    pkill -9 -f '^[^ ]*build/bin/llama-' 2>/dev/null
    sleep 2
    local v; v=$(vram_used_mib)
    echo "cleanup: vram_used=${v}MiB"
    [ "${v:-99999}" -le 300 ]
}
