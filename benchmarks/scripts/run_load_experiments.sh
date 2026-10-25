#!/bin/bash
# Simulator experiments for docs/15. Two independent streams run in parallel (the sim is mostly sleeping).
# Each stream writes its own directory (runs-a/, runs-b/) so parallel appends never interleave; tables: benchmarks/scripts/load_report.py
# Usage: benchmarks/scripts/run_load_experiments.sh [stream-a|stream-b|stream-c|all]
# (stream-c was added after stream-b showed that the hold starves other sessions; 'all' runs a+b, then c)
set -eu
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/Scripts/python.exe}
[ -x "$PY" ] || PY=.venv/bin/python
SIM="$PY -m llmserve.cli sim --cost live"

stream_a() {
    # E-LOAD: admission control vs none, live-calibrated service times
    $SIM --out benchmarks/results/load/runs-a --sessions 1,2,4,8 --tag e-load-noadm
    $SIM --out benchmarks/results/load/runs-a --sessions 1,2,4,8 --max-queue 2 --tag e-load-q2
    # E-SCHED part 1: queue order and anticipatory hold, one slot, 64k context (queue large: no rejections)
    $SIM --out benchmarks/results/load/runs-a --sessions 2,4 --max-queue 8 --policy fifo --tag e-sched-fifo
    $SIM --out benchmarks/results/load/runs-a --sessions 2,4 --max-queue 8 --policy affinity --tag e-sched-affinity
}

stream_b() {
    $SIM --out benchmarks/results/load/runs-b --sessions 2,4 --max-queue 8 --hold 5 --tag e-sched-hold5
    $SIM --out benchmarks/results/load/runs-b --sessions 2,4 --max-queue 8 --hold 15 --tag e-sched-hold15
    $SIM --out benchmarks/results/load/runs-b --sessions 2,4 --max-queue 8 --hold 25 --tag e-sched-hold25
    # E-NP2: two slots (32k context each) vs one slot with the same 30k client context cap
    $SIM --out benchmarks/results/load/runs-b --sessions 2,4 --max-queue 8 --max-context 30000 --tag e-np2-np1ctx30k
    for beta in 0.3 0.6 1.0; do
        $SIM --out benchmarks/results/load/runs-b --sessions 2,4 --max-queue 8 --max-context 30000 --slots 2 --beta $beta --tag e-np2-beta$beta
    done
}

stream_c() {
    # E-SCHED part 2: the anticipatory hold again, with the fairness guard (no hold once anyone waited > 60 s)
    $SIM --out benchmarks/results/load/runs-c --sessions 2,4 --max-queue 8 --hold 15 --hold-max-wait 60 --tag e-sched-hold15-guard
    $SIM --out benchmarks/results/load/runs-c --sessions 2,4 --max-queue 8 --hold 25 --hold-max-wait 60 --tag e-sched-hold25-guard
}

seeds() {
    # robustness: the three main contenders again with other seeds (different think times and lengths)
    for seed in "$@"; do
        o="--out benchmarks/results/load/runs-d-$seed --seed $seed"
        $SIM $o --sessions 2,4 --max-queue 8 --tag e-seed-fifo
        $SIM $o --sessions 2,4 --max-queue 8 --max-context 30000 --slots 2 --beta 0.6 --tag e-seed-np2
        $SIM $o --sessions 2,4 --max-queue 8 --hold 15 --hold-max-wait 60 --tag e-seed-hold15-guard
    done
}

case "${1:-all}" in
    stream-a) stream_a ;;
    stream-b) stream_b ;;
    stream-c) stream_c ;;
    seed-1) seeds 1 ;;
    seed-2) seeds 2 ;;
    all) stream_a & stream_b & wait; stream_c ;;
esac
