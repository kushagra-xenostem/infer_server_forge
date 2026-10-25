#!/usr/bin/env python3
"""Agent task suite: validate the suite, or run a model against it.

  agent_tasks.py validate                 # original must FAIL, reference must PASS
  agent_tasks.py run <run_name> [--url http://127.0.0.1:8080/v1]

Model output contract: full contents of each changed file as
    ### FILE: <path>
    ```python
    ...
    ```
Hidden tests are copied in AFTER model edits, so the model cannot overwrite them.
All test execution happens inside the no-network sandbox (scripts/sandbox.sh).
"""
import re
import sys
import json
import time
import httpx
import shutil
import pathlib
import subprocess

PROJ = pathlib.Path("/data/projects/llm-serving")
SUITE = PROJ / "tasks" / "suite"
SANDBOX = PROJ / "scripts" / "sandbox.sh"

SYSTEM = "You are an autonomous coding agent working in a Python repository. You make correct, minimal, complete changes."
FORMAT = (
    "Respond with the complete new contents of every file you create or modify, each in exactly this format:\n"
    "### FILE: <relative path>\n```python\n<full file content>\n```\n"
    "Do not include files you did not change. Do not write explanations outside these blocks."
)
FILE_RE = re.compile(r"### FILE:\s*`?([^\s`]+)`?\s*\n```[a-zA-Z0-9_+-]*\n(.*?)\n```", re.S)

# t20: broken variants of slugify; model-written tests should catch them.
_SLUG = (SUITE / "t20_write_tests" / "repo" / "slugify.py")
MUTANTS = {
    "no_accent_fold": ('t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()',
                       't = text.encode("ascii", "ignore").decode()'),
    "no_lower": ("t = t.lower()", "t = t"),
    "no_run_collapse": ('r"[^a-z0-9]+"', 'r"[^a-z0-9]"'),
    "no_strip": ('t = t.strip("-")', "t = t"),
    "no_rstrip_after_trunc": ('return t[:max_len].rstrip("-")', "return t[:max_len]"),
}
MUTANT_KILL_REQUIRED = 4


def tasks():
    for d in sorted(SUITE.iterdir()):
        if (d / "task.json").exists():
            yield d, json.loads((d / "task.json").read_text())


def build_prompt(tdir, meta):
    files = "\n\n".join(
        f"### FILE: {p.relative_to(tdir / 'repo')}\n```python\n{p.read_text()}```"
        for p in sorted((tdir / "repo").rglob("*.py"))
    )
    return f"Task:\n{meta['instruction']}\n\nRepository files:\n\n{files}\n\n{FORMAT}"


def parse_files(text):
    out = {}
    for path, body in FILE_RE.findall(text):
        if path.startswith("/") or ".." in pathlib.PurePosixPath(path).parts:
            continue
        out[path] = body + "\n"
    return out


def stage(work, tdir, meta, edits):
    """Create sandbox dirs for one task. Returns list of (dir_name, expect_pass)."""
    if work.exists():
        shutil.rmtree(work)
    if meta["category"] == "write-tests":
        model_test = edits.get("test_slugify.py")
        cases = [("orig", True)] + [(f"mut_{m}", False) for m in MUTANTS]
        for name, _ in cases:
            d = work / name
            shutil.copytree(tdir / "repo", d)
            if model_test:
                (d / "test_slugify.py").write_text(model_test)
            if name.startswith("mut_"):
                old, new = MUTANTS[name[4:]]
                src = (d / "slugify.py").read_text()
                assert old in src, f"mutant {name} does not apply"
                (d / "slugify.py").write_text(src.replace(old, new))
        return cases if model_test else []
    d = work / "case"
    shutil.copytree(tdir / "repo", d)
    for path, body in edits.items():
        p = d / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    shutil.copytree(tdir / "tests", d, dirs_exist_ok=True)
    return [("case", True)]


DRIVER = r'''
import json, subprocess, sys, pathlib
res = {}
for d in sorted(p for p in pathlib.Path(".").glob("*/*") if p.is_dir()):
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x"],
                           cwd=d, capture_output=True, text=True, timeout=60)
        res[str(d)] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]}
    except subprocess.TimeoutExpired:
        res[str(d)] = {"rc": "timeout", "tail": ""}
json.dump(res, open("pytest_results.json", "w"), indent=1)
'''


def run_sandbox(root):
    (root / "_driver.py").write_text(DRIVER)
    subprocess.run([str(SANDBOX), str(root), "python", "_driver.py"], check=True)
    return json.loads((root / "pytest_results.json").read_text())


def grade(root, plan):
    res = run_sandbox(root)
    out = {}
    for tid, cases in plan.items():
        if not cases:
            out[tid] = {"pass": False, "reason": "no parseable files in response"}
            continue
        r = {name: res.get(f"{tid}/{name}", {}).get("rc") for name, _ in cases}
        if any(name.startswith("mut_") for name, _ in cases):
            orig_ok = r["orig"] == 0
            killed = sum(1 for n, _ in cases if n.startswith("mut_") and r[n] not in (0, None))
            out[tid] = {"pass": orig_ok and killed >= MUTANT_KILL_REQUIRED, "orig_pass": orig_ok,
                        "mutants_killed": killed, "mutants_total": len(MUTANTS)}
        else:
            ok = r["case"] == 0
            out[tid] = {"pass": ok}
            if not ok:
                out[tid]["tail"] = res.get(f"{tid}/case", {}).get("tail", "")[-600:]
    return out


def cmd_validate():
    root = PROJ / "results" / "agent" / "_validate"
    if root.exists():
        shutil.rmtree(root)
    plan_orig, plan_ref = {}, {}
    for tdir, meta in tasks():
        ref = {p.name: p.read_text() for p in (tdir / "solution").glob("*.py")}
        if meta["category"] == "write-tests":
            plan_ref[meta["id"]] = stage(root / "ref" / meta["id"], tdir, meta, ref)
            continue
        plan_orig[meta["id"]] = stage(root / "orig" / meta["id"], tdir, meta, {})
        plan_ref[meta["id"]] = stage(root / "ref" / meta["id"], tdir, meta, ref)
    g_orig = grade(root / "orig", plan_orig)
    g_ref = grade(root / "ref", plan_ref)
    bad = 0
    for tid in sorted(g_ref):
        o = g_orig.get(tid, {}).get("pass")
        rf = g_ref[tid]["pass"]
        ok = rf and o is not True
        bad += not ok
        extra = {k: v for k, v in g_ref[tid].items() if k.startswith("mut")}
        print(f"{'OK ' if ok else 'BAD'} {tid:28s} original_passes={o} reference_passes={rf} {extra or ''}")
    shutil.rmtree(root)
    print(f"\n{len(g_ref) - bad}/{len(g_ref)} tasks valid")
    return bad == 0


def cmd_run(run_name, url):
    root = PROJ / "results" / "agent" / run_name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    plan, log = {}, []
    with httpx.Client(timeout=300) as cl:
        for tdir, meta in tasks():
            t0 = time.perf_counter()
            resp = cl.post(f"{url}/chat/completions", json={
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": build_prompt(tdir, meta)}],
                "temperature": 0, "seed": 42, "max_tokens": 2048,
            })
            dt = time.perf_counter() - t0
            if resp.status_code != 200:
                # e.g. HTTP 500 when the server can't parse garbled output: counts as a failed task
                r = {"choices": [{"message": {"content": ""}, "finish_reason": f"http_{resp.status_code}"}]}
            else:
                r = resp.json()
            text = r["choices"][0]["message"]["content"]
            edits = parse_files(text)
            plan[meta["id"]] = stage(root / meta["id"], tdir, meta, edits)
            log.append({"id": meta["id"], "category": meta["category"], "latency_s": round(dt, 2),
                        "usage": r.get("usage"), "files": sorted(edits), "finish": r["choices"][0].get("finish_reason"),
                        "response": text})
            print(f"{meta['id']:28s} {dt:6.1f}s files={sorted(edits)}", flush=True)
    graded = grade(root, plan)
    for row in log:
        row.update(graded[row["id"]])
    (root / "responses.json").write_text(json.dumps(log, indent=1))
    n = sum(r["pass"] for r in log)
    by_cat = {}
    for r in log:
        c = by_cat.setdefault(r["category"], [0, 0])
        c[0] += r["pass"]; c[1] += 1
    summary = {"pass": n, "total": len(log), "pass_rate": round(n / len(log), 3),
               "by_category": {k: f"{a}/{b}" for k, a, b in ((k, *v) for k, v in sorted(by_cat.items()))},
               "format_ok": sum(bool(r["files"]) for r in log),
               "truncated": sum(r["finish"] == "length" for r in log),
               "http_errors": sum(str(r["finish"]).startswith("http_") for r in log)}
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    for r in log:
        print(f"{'PASS' if r['pass'] else 'FAIL'} {r['id']}" + ("" if r["pass"] else f"  {r.get('reason') or ''}{r.get('tail','')[-200:]!r}" if r.get('tail') else ""))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        sys.exit(0 if cmd_validate() else 1)
    elif sys.argv[1] == "run":
        url = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "http://127.0.0.1:8080/v1"
        cmd_run(sys.argv[2], url)
