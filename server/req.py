#!/usr/bin/env python3
"""One chat request to the local server; prompt on stdin (argv caps at 128 KB).
Usage: req.py <max_tokens>  -> prints "pp_tps tg_tps n_prompt content_head"
"""
import sys
import json
import urllib.request

body = {"messages": [{"role": "user", "content": sys.stdin.read()}], "temperature": 0, "max_tokens": int(sys.argv[1])}
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
    {"Content-Type": "application/json"}), timeout=3600))
t = r["timings"]
print(f'{t["prompt_per_second"]:.1f} {t["predicted_per_second"]:.1f} {t["prompt_n"]} {json.dumps(r["choices"][0]["message"]["content"][:80])}')
