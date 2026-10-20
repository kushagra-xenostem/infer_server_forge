#!/bin/bash
# Production llama-server for Cline: Qwen3.6-35B-A3B UD-IQ4_XS, 64k context.
# Usage: prod_server.sh start [think] | stop | status
#   think: thinking mode with a reasoning budget (default: non-thinking)
# Every flag here comes from a measured failure on this box:
#   -ctk/-ctv q8_0        4-bit K cache destroyed Qwen2.5 quality
#   --no-host --no-repack pinned host buffers / repacking blew the amdgpu SVM cap and RAM
#   --cache-ram 0         default 8 GiB prompt-RAM cache filled swap over hours
#   VRAM guard (<95%)     128k at 100% VRAM deadlocked the amdgpu driver (needed a reboot)
set -u
source "$(dirname "$0")/common.sh"
MODEL=$HOME/models/gguf/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
LOG=$PROJ/logs/prod_server.log
CTX=${CTX:-65536}; NCMOE=${NCMOE:-24}; UB=${UB:-2048}
VRAM_TOTAL=10224; VRAM_MAX_PCT=95
# Vision: Qwen3.6 is natively multimodal; the projector runs on the CPU (--no-mmproj-offload) so it
# costs no VRAM (only ~0.9 GB of RAM). Set VISION=0 to disable.
MMPROJ=$HOME/models/gguf/mmproj-F16.gguf; VISION=${VISION:-1}
# gateway listens on the Tailscale address only (Cline runs on another machine in the tailnet);
# llama-server itself stays on 127.0.0.1 behind it
GW_HOST=${GW_HOST:-$(tailscale ip -4 2>/dev/null | head -1)}; GW_HOST=${GW_HOST:-127.0.0.1}

case "${1:-status}" in
start)
    if pgrep -f '^[^ ]*build/bin/llama-server' >/dev/null; then
        if curl -sf -m 3 127.0.0.1:$PORT/health >/dev/null && curl -sf -m 3 $GW_HOST:8081/health >/dev/null; then
            echo "Already running and healthy. Nothing to do; Cline can use http://$GW_HOST:8081/v1"
            echo "(To restart: $0 stop && $0 start)"
            exit 0
        fi
        echo "A llama-server is running but not healthy. Run: $0 stop   then: $0 start"
        exit 1
    fi
    preflight || { echo "not starting: preflight failed"; exit 1; }
    if [ "${2:-}" = think ]; then
        MODE=(--reasoning on --reasoning-budget 4096 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0)
    else
        MODE=(--reasoning off --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5)
    fi
    VIS=()
    [ "$VISION" = 1 ] && [ -f "$MMPROJ" ] && VIS=(--mmproj "$MMPROJ" --no-mmproj-offload --image-max-tokens ${IMG_TOKENS:-1024})
    rock_env
    nohup $LLAMA/build/bin/llama-server -m "$MODEL" --alias qwen3.6-35b-a3b \
        --host 127.0.0.1 --port $PORT -dev ROCm0 -ngl 99 --n-cpu-moe $NCMOE -fa on \
        -c $CTX -np 1 -t 4 -b 2048 -ub $UB -ctk q8_0 -ctv q8_0 \
        --no-host --no-repack --cache-ram 0 --jinja --metrics --no-webui "${MODE[@]}" "${VIS[@]}" \
        >> "$LOG" 2>&1 &
    SPID=$!
    echo "starting pid $SPID (ctx=$CTX ncmoe=$NCMOE mode=${2:-nothink} vision=$([ ${#VIS[@]} -gt 0 ] && echo cpu || echo off)) ..."
    for _ in $(seq 1 300); do
        curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
        kill -0 $SPID 2>/dev/null || { echo "server died during load"; tail -5 "$LOG"; exit 1; }
        sleep 1
    done
    V=$(vram_used_mib)
    if [ $((V * 100)) -gt $((VRAM_TOTAL * VRAM_MAX_PCT)) ]; then
        echo "VRAM GUARD: ${V} MiB used after load (> ${VRAM_MAX_PCT}%); stopping to avoid a driver hang"
        cleanup_gpu; exit 1
    fi
    ANS=$(curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"What is 17*23? Reply with only the number."}],"temperature":0,"max_tokens":10,"chat_template_kwargs":{"enable_thinking":false}}' |
        python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())')
    [ "$ANS" = "391" ] || { echo "SANITY FAIL: got '$ANS'"; cleanup_gpu; exit 1; }
    echo "READY on 127.0.0.1:$PORT  vram=${V}MiB  sanity=391"
    # logging gateway for Cline on :8081
    if ! ss -ltn | grep -q ":8081 "; then
        nohup $PROJ/.venv/bin/uvicorn --app-dir $PROJ/scripts gateway:app --host $GW_HOST --port 8081 \
            >> $PROJ/logs/gateway_server.log 2>&1 &
        echo $! > $PROJ/logs/gateway.pid
        for _ in $(seq 1 20); do curl -sf $GW_HOST:8081/health >/dev/null && break; sleep 0.5; done
    fi
    echo "gateway: http://$GW_HOST:8081/v1  (health $(curl -s -m 3 -o /dev/null -w '%{http_code}' $GW_HOST:8081/health))"
    ;;
stop)
    [ -f $PROJ/logs/gateway.pid ] && kill $(cat $PROJ/logs/gateway.pid) 2>/dev/null && rm -f $PROJ/logs/gateway.pid
    cleanup_gpu ;;
status)
    S=$(pgrep -af '^[^ ]*build/bin/llama-server')
    if [ -n "$S" ]; then
        echo "running: $(echo "$S" | grep -oE -- '--alias [^ ]+|-c [0-9]+|--reasoning [a-z]+|--mmproj [^ ]+' | tr '\n' ' ')"
        echo "vram=$(vram_used_mib)MiB  server health=$(curl -s -m 3 -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/health)  gateway health=$(curl -s -m 3 -o /dev/null -w '%{http_code}' $GW_HOST:8081/health)"
    else
        echo "not running"
    fi ;;
esac
