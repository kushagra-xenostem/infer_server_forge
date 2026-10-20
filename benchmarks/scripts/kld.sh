#!/bin/bash
# Phase 3b: perplexity + KL-divergence vs Q8_0 on wikitext-2 (100 x 512-token chunks).
# Base logits (~8 GB) are a temp artifact and are deleted at the end.
set -u
source "$(dirname "$0")/common.sh"
cd $PROJ; rock_env
BASE=$PROJ/data/kld_base_q8.bin
WT=$PROJ/data/wikitext-2-raw/wiki.test.raw
OUT=$PROJ/results/kld; mkdir -p $OUT
PPL="$LLAMA/build/bin/llama-perplexity -f $WT -c 512 --chunks 100 -dev ROCm0 -ngl 99 -fa on -t 4 -b 2048"
trap 'rm -f $BASE; cleanup_gpu' EXIT

preflight || exit 1
echo "== base: q8_0 kv f16"
$PPL -m $M_Q8 --kl-divergence-base $BASE > $OUT/q8_kvf16_base.log 2>&1 || { tail $OUT/q8_kvf16_base.log; exit 1; }
grep -E "Final estimate" $OUT/q8_kvf16_base.log
ls -lh $BASE

for cfg in "q8 $M_Q8 q4_0-q4_0" "q4km $M_Q4KM f16-f16" "q4km $M_Q4KM q8_0-q8_0" "q4km $M_Q4KM q8_0-q4_0" "q4km $M_Q4KM f16-q4_0" "q4km $M_Q4KM q4_0-f16" "q4km $M_Q4KM q4_0-q4_0"; do
    set -- $cfg; A=$1; M=$2; KV=$3
    preflight >/dev/null || { echo "preflight failed before $A $KV"; exit 1; }
    echo "== $A kv $KV"
    $PPL -m $M -ctk ${KV%-*} -ctv ${KV#*-} --kl-divergence-base $BASE --kl-divergence > $OUT/${A}_kv${KV}.log 2>&1 || { tail $OUT/${A}_kv${KV}.log; exit 1; }
    grep -E "^(Mean PPL|Mean    KLD|Maximum KLD|99.9%   KLD|Same top p|RMS Δp)" $OUT/${A}_kv${KV}.log
done
