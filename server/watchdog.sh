#!/bin/bash
# Restart llama-server when it stops answering. A process can be alive but hung (docs/16: a hung upstream
# keeps its slot forever; the gateway deadline only protects clients), so liveness is a real request:
# /health must answer within 10 s, three checks in a row, before a restart.
# Not covered: a GPU driver deadlock (docs/09 #9) leaves the process in D state; only a reboot clears it.
set -u
source "$(dirname "$0")/common.sh"
STATE=$PROJ/logs/watchdog.fails
LOG=$PROJ/logs/watchdog.log
fails=$(cat "$STATE" 2>/dev/null || echo 0)

if curl -sf -m 10 "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo 0 > "$STATE"
    exit 0
fi
# loading after a (re)start answers 503 "Loading model": not a failure while systemd is still starting it
if systemctl is-active --quiet llm-serving && [ "$(systemctl show -p ActiveState --value llm-serving)" = activating ]; then
    exit 0
fi
fails=$((fails + 1))
echo "$fails" > "$STATE"
echo "$(date -Is) health check failed ($fails/3)" >> "$LOG"
if [ "$fails" -ge 3 ]; then
    echo "$(date -Is) restarting llm-serving" >> "$LOG"
    echo 0 > "$STATE"
    sudo -n systemctl restart llm-serving >> "$LOG" 2>&1 || echo "$(date -Is) restart failed (sudo rule missing?)" >> "$LOG"
fi
