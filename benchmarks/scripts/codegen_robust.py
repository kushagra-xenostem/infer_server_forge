#!/usr/bin/env python3
"""EvalPlus-equivalent greedy codegen that tolerates server errors.

Mirrors evalplus 0.3.1 OpenAIChatDecoder (same system prompt, instruction prefix,
max_tokens=768, top_p=0.95, sanitizer), but a failed request (e.g. HTTP 500 when
llama-server cannot parse a garbled chat output) is recorded as an empty solution
(= task failure) instead of being retried forever.

Usage: codegen_robust.py <dataset> <alias> <out_root> [limit]
"""
import json
import os
import sys

import httpx
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.sanitize import sanitize

SYSTEM = "You are a helpful assistant good at coding."
PREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"


def main():
    ds, alias, root = sys.argv[1:4]
    data = get_human_eval_plus() if ds == "humaneval" else get_mbpp_plus()
    if len(sys.argv) > 4:
        data = dict(list(data.items())[: int(sys.argv[4])])
    out_dir = os.path.join(root, ds)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{alias}_openai_temp_0.0")
    errors = 0
    # resume: keep tasks already present in both files (greedy decoding, so redoing them would be identical)
    done = set()
    if os.path.exists(base + ".jsonl") and os.path.exists(base + ".raw.jsonl"):
        s_ids = [json.loads(l)["task_id"] for l in open(base + ".jsonl") if l.strip()]
        r_ids = [json.loads(l)["task_id"] for l in open(base + ".raw.jsonl") if l.strip()]
        done = set(s_ids) & set(r_ids)
        if set(s_ids) != set(r_ids) or len(s_ids) != len(set(s_ids)):
            done = set()  # inconsistent files: start over
    mode = "a" if done else "w"
    if done:
        for path in (base + ".jsonl", base + ".raw.jsonl"):
            rows = [l for l in open(path) if l.strip() and json.loads(l)["task_id"] in done]
            open(path, "w").writelines(rows)
        print(f"resuming: {len(done)} tasks already done", flush=True)
    with httpx.Client(timeout=600) as cl, open(base + ".jsonl", mode) as fs, open(base + ".raw.jsonl", mode) as fr:
        for task_id, task in data.items():
            if task_id in done:
                continue
            prompt = task["prompt"].strip() + "\n"
            r = cl.post(f"http://127.0.0.1:{os.environ.get('PORT', '8080')}/v1/chat/completions", json={
                "model": alias, "temperature": 0, "top_p": 0.95, "max_tokens": 768, "n": 1,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": PREFIX + f"\n```python\n{prompt.strip()}\n```"}]})
            if r.status_code == 200:
                raw = r.json()["choices"][0]["message"]["content"]
            else:
                errors += 1
                raw = ""
                print(f"{task_id}: HTTP {r.status_code} {r.text[:120]}", flush=True)
            fs.write(json.dumps({"task_id": task_id, "solution": sanitize(raw, entrypoint=task["entry_point"]) if raw else ""}) + "\n")
            fr.write(json.dumps({"task_id": task_id, "solution": raw, "http_status": r.status_code}) + "\n")
            fs.flush(); fr.flush()
    print(json.dumps({"tasks": len(data), "http_errors": errors}))


if __name__ == "__main__":
    main()
