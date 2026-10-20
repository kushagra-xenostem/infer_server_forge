#!/bin/bash
# Start llama-server for an eval run and verify it before returning.
# Usage: serve.sh <model.gguf> <alias> [kv_type=f16 | K-V e.g. q8_0-q4_0] [ctx=8192] [extra llama-server args...]
# SKIP_SANITY=1 skips the known-answer gate (only for measuring a known-broken config).
set -u
source "$(dirname "$0")/common.sh"

MODEL=$1; ALIAS=$2; KV=${3:-f16}; CTX=${4:-8192}; shift $(( $# < 4 ? $# : 4 ))
LOG=$PROJ/logs/server_${ALIAS}_kv${KV}.log

preflight || { echo "serve: preflight failed, not starting"; exit 1; }
[ -f "$MODEL" ] || { echo "serve: model not found: $MODEL"; exit 1; }

rock_env
$LLAMA/build/bin/llama-server \
    -m "$MODEL" --alias "$ALIAS" \
    --host 127.0.0.1 --port $PORT \
    -dev ROCm0 -ngl 99 -fa on \
    -c $CTX -np 1 -t 4 -b 2048 -ub 512 \
    -ctk ${KV%-*} -ctv ${KV#*-} \
    --jinja --seed 42 --no-webui --metrics --cache-ram 0 \
    "$@" > "$LOG" 2>&1 &
SPID=$!
echo "serve: pid $SPID log $LOG"

for _ in $(seq 1 600); do  # 10 min: /data is an HDD (~113 MB/s), cold 8 GB loads are slow
    curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
    if ! kill -0 $SPID 2>/dev/null; then
        echo "serve: server died during load"; tail -20 "$LOG"; exit 1
    fi
    sleep 1
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "serve: health timeout"; tail -20 "$LOG"; exit 1; }

# Sanity: deterministic known answer, and model must be on the GPU.
ANS=$(curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is 17*23? Reply with only the number."}],"temperature":0,"max_tokens":10}' |
    python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())')
V=$(vram_used_mib)
echo "serve: sanity answer='$ANS' vram_used=${V}MiB"
[ "$ANS" = "391" ] || [ "${SKIP_SANITY:-0}" = 1 ] || { echo "serve: SANITY FAIL"; exit 1; }
[ "$V" -gt 2000 ] || { echo "serve: model not on GPU (vram ${V}MiB)"; exit 1; }
echo "serve: READY"
