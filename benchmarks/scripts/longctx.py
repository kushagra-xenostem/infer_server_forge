#!/usr/bin/env python3
"""Long-context tests against a running llama-server (port 8080).

needle  <target_tokens> <out.json>
    Real C++ (llama.cpp sources) as a haystack, padded to ~target tokens. Synthetic functions with
    known return values are planted at 10/50/90% depth, plus a two-hop chain (A at 5% calls B at 95%).
    Questions share the haystack prefix, so only the first one pays full prefill (prompt-cache reuse).
    A question about a function that does not exist must be answered NOT FOUND.
session <turns> <tokens_per_turn> <out.json>
    Simulated agent session: each turn appends a new code chunk + question to the conversation.
    Measures per-turn prefill (only new tokens, if the cache works), latency and correctness.
"""
import glob
import json
import random
import re
import sys
import time

import httpx

URL = "http://127.0.0.1:8080"
SRC = sorted(glob.glob("/data/projects/llama.cpp/src/*.cpp") + glob.glob("/data/projects/llama.cpp/common/*.cpp")
             + glob.glob("/data/projects/llama.cpp/tools/server/*.cpp") + glob.glob("/data/projects/llama.cpp/ggml/src/*.c"))
SYSTEM = "You are a precise code assistant. Answer with only the requested value, no explanation."
cl = httpx.Client(timeout=7200)


def ntok(text):
    return len(cl.post(f"{URL}/tokenize", json={"content": text}).json()["tokens"])


def corpus():
    for p in SRC:
        yield f"// ===== file: {p.split('/data/projects/llama.cpp/')[1]} =====\n" + open(p, errors="replace").read()


def haystack(target):
    """Whole files concatenated up to ~target tokens (estimate by chars, then measure exactly)."""
    parts, chars = [], 0
    for f in corpus():
        parts.append(f)
        chars += len(f)
        if chars > target * 3.2:  # C++ is ~3–3.5 chars/token
            break
    text = "\n".join(parts)
    while ntok(text) > target:
        text = text[: int(len(text) * 0.97)]
        text = text[: text.rfind("\n")]
    return text


def plant(text, frac, snippet):
    i = text.rfind("\n", 0, int(len(text) * frac)) + 1
    return text[:i] + snippet + "\n" + text[i:]


def ask(messages, max_tokens=48):
    t = time.perf_counter()
    r = cl.post(f"{URL}/v1/chat/completions", json={"messages": messages, "temperature": 0, "max_tokens": max_tokens})
    wall = time.perf_counter() - t
    if r.status_code != 200:
        return {"error": r.status_code, "body": r.text[:200], "wall_s": round(wall, 1)}
    j = r.json()
    tm = j["timings"]
    return {"answer": j["choices"][0]["message"]["content"].strip(), "wall_s": round(wall, 1),
            "prompt_n": tm["prompt_n"], "cached": tm.get("cache_n"), "prefill_tps": round(tm["prompt_per_second"], 1),
            "prefill_s": round(tm["prompt_ms"] / 1000, 1), "gen_tps": round(tm["predicted_per_second"], 1)}


def needle(target, out):
    rnd = random.Random(target)
    names = [f"nx_{w}_{rnd.randint(100, 999)}" for w in ("ledger", "quorum", "spindle", "harbor", "basalt")]
    vals = [rnd.randint(1000, 9999) for _ in range(4)]
    snips = {0.10: f"static int {names[0]}(void) {{ return {vals[0]}; }}",
             0.50: f"static int {names[1]}(void) {{ return {vals[1]}; }}",
             0.90: f"static int {names[2]}(void) {{ return {vals[2]}; }}",
             0.05: f"static int {names[3]}(void) {{ return {names[4]}() + 17; }}",
             0.95: f"static int {names[4]}(void) {{ return {vals[3]}; }}"}
    text = haystack(target - 600)
    for frac in sorted(snips, reverse=True):
        text = plant(text, frac, snips[frac])
    prompt_tokens = ntok(text)
    qs = [("depth10", f"What integer does the function {names[0]} return?", str(vals[0])),
          ("depth50", f"What integer does the function {names[1]} return?", str(vals[1])),
          ("depth90", f"What integer does the function {names[2]} return?", str(vals[2])),
          ("twohop", f"What integer does the function {names[3]} return? Follow any calls it makes.", str(vals[3] + 17)),
          ("absent", "What integer does the function nx_cobalt_404 return? If that function does not appear in the code, answer exactly NOT FOUND.", "NOT FOUND")]
    res = {"target": target, "haystack_tokens": prompt_tokens, "questions": []}
    for qid, q, want in qs:
        r = ask([{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": f"{text}\n\nQuestion: {q}"}])
        got = r.get("answer", "")
        r.update(id=qid, expected=want,
                 correct=(want in got) if want != "NOT FOUND" else ("NOT FOUND" in got.upper()))
        res["questions"].append(r)
        print(json.dumps(r), flush=True)
    res["accuracy"] = f'{sum(q["correct"] for q in res["questions"])}/{len(qs)}'
    json.dump(res, open(out, "w"), indent=1)
    print("ACCURACY", res["accuracy"], "haystack_tokens", prompt_tokens, flush=True)


def session(turns, per_turn, out):
    """Growing conversation: new code chunk each turn + a question about something only in that chunk."""
    files = list(corpus())
    rnd = random.Random(7)
    msgs = [{"role": "system", "content": "You are a coding agent. Answer each question with only the requested value."}]
    log = []
    fi = 0
    for t in range(1, turns + 1):
        chunk = ""
        while len(chunk) < per_turn * 3.2 and fi < len(files):
            chunk += files[fi]; fi += 1
        chunk = chunk[: int(per_turn * 3.2)]
        name, val = f"turn{t}_marker_{rnd.randint(100, 999)}", rnd.randint(1000, 9999)
        chunk = plant(chunk, rnd.uniform(0.2, 0.8), f"static int {name}(void) {{ return {val}; }}")
        # every 4th turn, ask about a marker from an early turn (tests recall across the whole session)
        if t % 4 == 0 and log:
            old = log[0]
            q, want = f"Earlier in this session there was a function {old['name']}. What integer does it return?", str(old["val"])
        else:
            q, want = f"What integer does the function {name} return?", str(val)
        msgs.append({"role": "user", "content": f"Here is more code from the repository:\n{chunk}\n\nQuestion: {q}"})
        r = ask(msgs, max_tokens=32)
        msgs.append({"role": "assistant", "content": r.get("answer", "")})
        r.update(turn=t, name=name, val=val, expected=want, correct=want in r.get("answer", ""))
        log.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("name",)}), flush=True)
        if "error" in r:
            break
    json.dump(log, open(out, "w"), indent=1)
    print("SESSION", f'{sum(x["correct"] for x in log)}/{len(log)} correct', flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "needle":
        needle(int(sys.argv[2]), sys.argv[3])
    elif mode == "session":
        session(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
