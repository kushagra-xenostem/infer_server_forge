#!/bin/bash
# Probe one 30B-A3B MoE offload config: load time, memory, prompt/generation speed, quality sanity.
# Usage: probe_30b.sh <n_cpu_moe> <ctx> [extra llama-server args...]
set -u
source "$(dirname "$0")/common.sh"
N=$1; CTX=$2; shift 2
M30=$(ls $HOME/.cache/huggingface/hub/models--n00b001--Qwen3-Coder-30B-A3B-Instruct-Q4_K_S-GGUF/snapshots/*/*.gguf)
TAG="ncmoe${N}_c${CTX}$(echo "$*" | tr -d ' -')"
LOG=$PROJ/logs/probe30_$TAG.log
preflight >/dev/null || { echo "preflight failed"; preflight; exit 1; }
SWAP0=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
rock_env
T0=$(date +%s)
$LLAMA/build/bin/llama-server -m "$M30" --alias q30 --host 127.0.0.1 --port $PORT \
    -dev ROCm0 -ngl 99 --n-cpu-moe $N -fa on -c $CTX -np 1 -t 4 -ctk q8_0 -ctv q8_0 \
    --jinja --no-webui --no-host --no-repack "$@" > $LOG 2>&1 &
SPID=$!
until curl -sf 127.0.0.1:$PORT/health >/dev/null; do
    kill -0 $SPID 2>/dev/null || { echo "$TAG: server died"; grep -iE "error|fail|alloc" $LOG | tail -5; exit 1; }
    [ $(( $(date +%s) - T0 )) -gt 900 ] && { echo "$TAG: load timeout"; cleanup_gpu >/dev/null; exit 1; }
    sleep 2
done
LOAD=$(( $(date +%s) - T0 ))

req() {  # prompt_text max_tokens -> prints "pp_tps tg_tps n_prompt content"
    printf '%s' "$1" | python3 $PROJ/scripts/req.py "$2"
}
SAN=$(req "What is 17*23? Reply with only the number." 10)
# ~3k-token code prompt: realistic agent-style prefill
LONG="$(cat $PROJ/tasks/suite/*/repo/*.py | head -c 11000)

Summarize what each module above does in one line each."
SHORT=$(req "Write a Python function that checks if a string is a palindrome." 200)
BIG=$(req "$LONG" 200)
if [ -n "${HUGE_CHARS:-}" ]; then
    HUGE="$(cat $LLAMA/src/llama-*.cpp | head -c $HUGE_CHARS)

Which function in the code above handles loading the KV cache state from a file? Answer with just its name."
    HUGE_R=$(req "$HUGE" 60)
fi
SWAP1=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
echo "$TAG load=${LOAD}s vram=$(vram_used_mib)MiB rss=$(( $(ps -o rss= -p $SPID)/1024 ))MB avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB swapΔ=$((SWAP1-SWAP0))MB"
echo "  sanity : $SAN"
echo "  short  : pp/tg/n_prompt = $SHORT" | cut -c1-110
echo "  3k-ctx : pp/tg/n_prompt = $BIG" | cut -c1-110
[ -n "${HUGE_R:-}" ] && echo "  huge   : pp/tg/n_prompt = $HUGE_R" | cut -c1-130
cleanup_gpu >/dev/null
