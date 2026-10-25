#!/usr/bin/env python3
"""Append one result row, with the exact serving config, to results/summary.jsonl.

Usage: record.py <phase> <benchmark> <model_alias> <kv_type> <metrics-json> [notes]
Server config is read live from llama-server /props so every number is traceable.
"""
import os
import sys
import json
import hashlib
import datetime
import subprocess
import urllib.request

PROJ = "/data/projects/llm-serving"
PORT = os.environ.get("PORT", "8080")


def file_id(path):
    # size + hash of first 16 MiB: cheap but distinguishes model files reliably
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(16 << 20))
    return f"{st.st_size}:{h.hexdigest()[:16]}"


def server_props():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5) as r:
            p = json.load(r)
        return {
            "model_path": p.get("model_path"),
            "n_ctx": p.get("default_generation_settings", {}).get("n_ctx"),
            "build": p.get("build_info"),
        }
    except Exception as e:
        return {"error": str(e)}


def snapshot(path):
    cmdline = subprocess.run(
        ["pgrep", "-af", "^[^ ]*build/bin/llama-server"], capture_output=True, text=True
    ).stdout.strip()
    json.dump({"props": server_props(), "cmdline": cmdline}, open(path, "w"))


def main():
    if sys.argv[1] == "--snapshot":
        return snapshot(sys.argv[2])
    phase, bench, alias, kv, metrics = sys.argv[1:6]
    notes = sys.argv[6] if len(sys.argv) > 6 else ""
    if os.environ.get("SERVER_INFO"):
        # snapshot taken while the server was up (results recorded after it stopped)
        snap = json.load(open(os.environ["SERVER_INFO"]))
        props, cmdline = snap["props"], snap["cmdline"]
    else:
        props = server_props()
        cmdline = subprocess.run(
            ["pgrep", "-af", "^[^ ]*build/bin/llama-server"], capture_output=True, text=True
        ).stdout.strip()
    row = {
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "phase": phase,
        "benchmark": bench,
        "model": alias,
        "kv_type": kv,
        "metrics": json.loads(metrics),
        "server": props,
        "model_file_id": file_id(props["model_path"]) if props.get("model_path") else None,
        "server_cmdline": cmdline.split(" ", 1)[-1] if cmdline else None,
        "llama_commit": subprocess.run(
            ["git", "-C", "/data/projects/llama.cpp", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip(),
        "notes": notes,
    }
    with open(f"{PROJ}/results/summary.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print("recorded:", bench, alias, kv, row["metrics"])


if __name__ == "__main__":
    main()
