#!/usr/bin/env python3
"""Print a compact per-turn table from a longctx.py session JSON. Usage: show_session.py <session.json>"""
import sys
import json

for r in json.load(open(sys.argv[1])):
    ctx = (r.get("cached") or 0) + (r.get("prompt_n") or 0)
    ans = r.get("answer", r.get("error"))
    print(f"turn {r['turn']:2d}  ctx={ctx:6d}  new={r.get('prompt_n')}  prefill={r.get('prefill_s')}s  "
          f"wall={r['wall_s']}s  gen={r.get('gen_tps')}t/s  {'OK ' if r['correct'] else 'BAD'} ans={ans} want={r['expected']}")
