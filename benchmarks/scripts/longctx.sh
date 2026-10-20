#!/bin/bash
# Step 1: long-context validation of Qwen3.6-35B-A3B at 32k / 64k / 128k.
set -u
source "$(dirname "$0")/common.sh"
cd $PROJ
exec 9>$PROJ/logs/.phase3.lock
flock -n 9 || { echo "another GPU pipeline is running"; exit 1; }
step() { echo; echo "======== $* ($(date -u +%H:%M:%S))"; }
M35=$HOME/models/gguf/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
OUT=$PROJ/results/longctx; mkdir -p $OUT
PY=$PROJ/.venv/bin/python
( while true; do
    S=$(pgrep -f '^[^ ]*build/bin/llama-server' | head -1)
    echo "$(date -u +%T) vram=$(vram_used_mib)MiB rss=$([ -n "$S" ] && echo $(( $(ps -o rss= -p $S)/1024 )) || echo -)MB avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB swap_used=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)MB"
    sleep 30; done ) >> logs/mem_longctx.log 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null; cleanup_gpu >/dev/null' EXIT
svm() { journalctl -k --since "$1" --no-pager 2>/dev/null | grep -c "SVM mapping failed"; }

run_cfg() {  # ctx ncmoe ub -> starts server, returns 0 if READY
    scripts/serve.sh "$M35" q36_35b q8_0-q8_0 $1 --n-cpu-moe $2 --no-host --no-repack -b 2048 -ub $3 -rea off
}

step "32k (ncmoe 24, ub 2048)"
T0=$(date '+%Y-%m-%d %H:%M:%S')
run_cfg 32768 24 2048 && $PY scripts/longctx.py needle 30000 $OUT/needle_32k.json
echo "SVM warnings: $(svm "$T0")"; cleanup_gpu

step "64k (ncmoe 24, ub 2048): needle + 16-turn session"
T0=$(date '+%Y-%m-%d %H:%M:%S')
if run_cfg 65536 24 2048; then
    $PY scripts/longctx.py needle 60000 $OUT/needle_64k.json
    $PY scripts/longctx.py session 16 3000 $OUT/session_64k.json
fi
echo "SVM warnings: $(svm "$T0")"; cleanup_gpu

step "128k: find a config that loads"
FIT=""
for cfg in "24 1024" "26 1024" "28 1024" "26 512"; do
    set -- $cfg
    T0=$(date '+%Y-%m-%d %H:%M:%S')
    if run_cfg 131072 $1 $2; then FIT="$1 $2"; break; fi
    echo "128k with ncmoe=$1 ub=$2 failed (SVM warnings: $(svm "$T0"))"; cleanup_gpu
done
if [ -n "$FIT" ]; then
    echo "128k fits with ncmoe/ub = $FIT"
    $PY scripts/longctx.py needle 120000 $OUT/needle_128k.json
    echo "SVM warnings after needle: $(svm "$T0")"
    $PY scripts/longctx.py session 30 3800 $OUT/session_128k.json
    echo "SVM warnings after session: $(svm "$T0")"
else
    echo "128k: no config loaded"
fi
cleanup_gpu
preflight
echo "LONGCTX COMPLETE"
