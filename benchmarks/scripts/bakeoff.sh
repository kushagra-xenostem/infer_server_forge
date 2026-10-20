#!/bin/bash
# Bake-off step 1: same coding suite (HumanEval+/MBPP+/agent tasks) for each candidate at its deployment config.
# Benchmarks run non-thinking (-rea off) for comparability; greedy decoding.
set -u
source "$(dirname "$0")/common.sh"
cd $PROJ
exec 9>$PROJ/logs/.phase3.lock
flock -n 9 || { echo "another GPU pipeline is running"; exit 1; }
step() { echo; echo "======== $* ($(date -u +%H:%M:%S))"; }
# memory watchdog log: server RSS, available RAM, swap (the 8 GiB --cache-ram default once filled swap)
( while true; do
    S=$(pgrep -f '^[^ ]*build/bin/llama-server' | head -1)
    echo "$(date -u +%T) rss=$([ -n "$S" ] && echo $(( $(ps -o rss= -p $S)/1024 )) || echo -)MB avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB swap_used=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)MB"
    sleep 60; done ) >> logs/mem_bakeoff.log 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

G=$HOME/models/gguf
M30=$(ls $HOME/.cache/huggingface/hub/models--n00b001--Qwen3-Coder-30B-A3B-Instruct-Q4_K_S-GGUF/snapshots/*/*.gguf)
M9=$G/Qwen3.5-9B-Q4_K_M.gguf
M35=$G/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
MOE="--no-host --no-repack -b 2048 -ub 2048"

wait_file() {  # wait until pdl.py finished and verified (no .parts file left)
    until [ -f "$1" ] && [ ! -f "$1.parts" ]; do sleep 60; done
    [ -f "$1.bad" ] && { echo "hash mismatch for $1"; return 1; }
    echo "ready: $1"
}

step "Qwen3-Coder-30B-A3B Q4_K_S (ncmoe 30, 32k)"
CTX=32768 SERVE_ARGS="--n-cpu-moe 30 $MOE" scripts/coding_suite.sh "$M30" q30coder q8_0-q8_0 || echo "FAILED q30coder"

step "Qwen3.5-9B Q4_K_M (full GPU, 64k, non-thinking)"
wait_file $M9 && CTX=65536 SERVE_ARGS="-rea off" scripts/coding_suite.sh "$M9" q35_9b q8_0-q8_0 || echo "FAILED q35_9b"

step "Qwen3.6-35B-A3B UD-IQ4_XS (MoE offload, 64k, non-thinking)"
# first find the smallest --n-cpu-moe that loads and passes sanity
wait_file $M35 && for N in 18 21 24 27; do
    if CTX=65536 SERVE_ARGS="--n-cpu-moe $N $MOE -rea off" scripts/serve.sh "$M35" probe35 q8_0-q8_0 65536 --n-cpu-moe $N $MOE -rea off; then
        echo "q36_35b fits with ncmoe=$N"; cleanup_gpu; FIT=$N; break
    fi
    cleanup_gpu
done
[ -n "${FIT:-}" ] && CTX=65536 SERVE_ARGS="--n-cpu-moe $FIT $MOE -rea off" scripts/coding_suite.sh "$M35" q36_35b q8_0-q8_0 || echo "FAILED q36_35b"

step "report"
python3 scripts/report.py > /dev/null && echo "report written"
preflight
echo "BAKEOFF STEP1 COMPLETE"
