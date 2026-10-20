#!/bin/bash
# Phase 2: general reasoning / instruction following via lm-eval against llama-server.
# Usage: run_lmeval.sh <alias> <run_name>
set -eu
cd /data/projects/llm-serving
A=$1; RUN=$2
for spec in "gsm8k_cot_llama 250" "ifeval none" "mmlu_pro 10"; do
    set -- $spec; TASK=$1; LIM=$2
    LIMARG=""; [ "$LIM" != none ] && LIMARG="--limit $LIM"
    .venv/bin/lm_eval --model local-chat-completions \
        --model_args model=$A,base_url=http://127.0.0.1:8080/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,tokenizer_backend=None \
        --tasks $TASK --apply_chat_template --fewshot_as_multiturn $LIMARG \
        --gen_kwargs temperature=0 --seed 42 \
        --output_path results/lmeval/$RUN --log_samples > logs/lmeval_${RUN}_${TASK}.log 2>&1
    echo "$TASK done rc=$?"
done
