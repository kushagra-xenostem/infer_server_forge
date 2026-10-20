#!/bin/bash
# Full coding suite for one serving config: serve -> HumanEval+/MBPP+ -> agent tasks -> record -> cleanup.
# Usage: coding_suite.sh <model.gguf> <alias> <kv_type> [datasets="humaneval mbpp"]
# Env: SERVE_ARGS="extra llama-server flags" (e.g. MoE offload, -rea off), CTX (default 8192), RUN_TAG (suffix)
set -u
source "$(dirname "$0")/common.sh"
MODEL=$1; A=$2; KV=$3; RUN=${A}_kv${KV}${RUN_TAG:-}; DATASETS=${4:-humaneval mbpp}
cd $PROJ
trap 'cleanup_gpu' EXIT

echo "##### $RUN: start $(date -u +%H:%M:%S)"
scripts/serve.sh "$MODEL" "$A" "$KV" "${CTX:-8192}" ${SERVE_ARGS:-} || exit 1
mkdir -p results/evalplus/$RUN
python3 scripts/record.py --snapshot results/evalplus/$RUN/server.json

for ds in $DATASETS; do
    # codegen_robust == evalplus OpenAI codegen (verified identical), with resume + 600 s timeout, no infinite retry
    .venv/bin/python scripts/codegen_robust.py $ds $A results/evalplus/$RUN \
        >> logs/codegen_${ds}_${RUN}.log 2>&1 || { echo "codegen $ds failed"; exit 1; }
    tail -1 logs/codegen_${ds}_${RUN}.log
    F=results/evalplus/$RUN/$ds/${A}_openai_temp_0.0.jsonl
    # every task present; a few empty answers (model rambled to the token limit) count as failures,
    # but >5% empty means something systematic (server/template/cache) is broken
    read N E <<< $(python3 -c "import json;r=[json.loads(l) for l in open('$F')];print(len({x['task_id'] for x in r}), sum(len(x['solution'].strip())<15 for x in r))")
    echo "$ds: $N unique samples, $E empty"
    EXP=$([ $ds = humaneval ] && echo 164 || echo 378)
    [ "$N" = "$EXP" ] || { echo "VERIFY FAIL: $ds sample count $N != $EXP"; exit 1; }
    [ $((E * 20)) -le "$EXP" ] || { echo "VERIFY FAIL: $ds has $E empty solutions (>5%)"; exit 1; }
done

# agent suite while the server is still up
.venv/bin/python scripts/agent_tasks.py run $RUN > logs/agent_${RUN}.log 2>&1 || { echo "agent run failed"; exit 1; }
python3 scripts/record.py 4 agent_suite $A $KV "$(cat results/agent/$RUN/summary.json)" "20 custom tasks"
cleanup_gpu || { echo "VERIFY FAIL: VRAM not released"; exit 1; }
trap - EXIT

# execute generated code in the sandbox (server already stopped, so CPU is free)
for ds in $DATASETS; do
    scripts/sandbox.sh results/evalplus/$RUN python -m evalplus.evaluate --dataset $ds \
        --samples $ds/${A}_openai_temp_0.0.jsonl --parallel 4 > logs/eval_${ds}_${RUN}.log 2>&1 || { echo "eval $ds failed"; exit 1; }
    M=$(python3 - <<EOF
import json
d = json.load(open("results/evalplus/$RUN/$ds/${A}_openai_temp_0.0_eval_results.json"))["eval"]
n = len(d); b = sum(v[0]["base_status"] == "pass" for v in d.values())
p = sum(v[0]["base_status"] == "pass" and v[0]["plus_status"] == "pass" for v in d.values())
print(json.dumps({"$ds": round(b / n, 3), "${ds}_plus": round(p / n, 3), "n": n}))
EOF
)
    echo "$ds: $M"
    SERVER_INFO=results/evalplus/$RUN/server.json python3 scripts/record.py 3 $ds $A $KV "$M" "evalplus greedy; server stopped before eval"
done
echo "##### $RUN: done $(date -u +%H:%M:%S)"
