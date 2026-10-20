#!/usr/bin/env python3
"""Build results/REPORT.md (+ results/report_data.json) from all recorded runs."""
import glob
import json
import pathlib
import re

R = pathlib.Path("/data/projects/llm-serving/results")
PUBLISHED = {"humaneval": 88.4, "humaneval_plus": 84.1, "mbpp": 83.5, "mbpp_plus": 71.7}
CONFIGS = [("q8", "f16"), ("q4km", "f16"), ("q4km", "q8_0"), ("q4km", "q8_0-q4_0"), ("q4km", "q4_0")]
LABEL = {"q8": "Q2.5-Coder-7B Q8_0", "q4km": "Q2.5-Coder-7B Q4_K_M", "q30coder": "Q3-Coder-30B-A3B Q4_K_S",
         "q35_9b": "Qwen3.5-9B Q4_K_M", "q36_35b": "Qwen3.6-35B-A3B IQ4_XS"}


class _Label(dict):
    def __missing__(self, k):
        return k


LABEL = _Label(LABEL)


def rows():
    return [json.loads(l) for l in open(R / "summary.jsonl")]


def latest(bench, model, kv):
    hit = [r for r in rows() if r["benchmark"] == bench and r["model"] == model and r["kv_type"] == kv]
    return hit[-1]["metrics"] if hit else None


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def lmeval(run):
    out = {}
    for f in sorted(glob.glob(str(R / "lmeval" / run / "*" / "results_*.json"))):
        d = json.load(open(f))
        for task, m in d["results"].items():
            if task == "gsm8k_cot_llama":
                out["GSM8K (8-shot CoT, flexible)"] = (m["exact_match,flexible-extract"], m["exact_match_stderr,flexible-extract"], m.get("sample_len"))
            elif task == "ifeval":
                out["IFEval prompt-level strict"] = (m["prompt_level_strict_acc,none"], m.get("prompt_level_strict_acc_stderr,none"), m.get("sample_len"))
                out["IFEval instruction-level loose"] = (m["inst_level_loose_acc,none"], None, m.get("sample_len"))
            elif task == "mmlu_pro":
                out["MMLU-Pro (5-shot CoT, 10/subject)"] = (m["exact_match,custom-extract"], m.get("exact_match_stderr,custom-extract"), None)
    return out


def kld():
    out = {}
    for f in sorted(glob.glob(str(R / "kld" / "*.log"))):
        name = pathlib.Path(f).stem
        t = open(f).read()
        g = lambda pat: (m.group(1) if (m := re.search(pat, t)) else None)
        out[name] = {
            "ppl": g(r"Mean PPL\(Q\)\s*:\s*([\d.]+\s*±\s*[\d.]+)") or g(r"Final estimate: PPL = ([\d.]+ \+/- [\d.]+)"),
            "kld_mean": g(r"Mean\s+KLD:\s*([\d.\-]+\s*±\s*[\d.]+)"),
            "kld_99_9": g(r"99\.9%\s+KLD:\s*([\d.\-]+)"),
            "kld_max": g(r"Maximum KLD:\s*([\d.\-]+)"),
            "same_top": g(r"Same top p:\s*([\d.]+ ± [\d.]+ %)"),
            "rms_dp": g(r"RMS Δp\s*:\s*([\d.]+ ± [\d.]+ %)"),
        }
    return out


def agent_detail(run):
    p = R / "agent" / run / "responses.json"
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    global CONFIGS
    for r in rows():  # include every recorded (model, kv) pair, keeping the base order first
        if (r["model"], r["kv_type"]) not in CONFIGS:
            CONFIGS.append((r["model"], r["kv_type"]))
    L = ["# Qwen2.5-Coder-7B-Instruct — accuracy & intelligence on RX 6700 (llama.cpp ROCm)", ""]
    data = {"coding": {}, "agent": {}, "lmeval": {}, "kld": kld()}

    L += ["## Coding benchmarks (EvalPlus, greedy, pass@1 %)", "",
          "| Config | HumanEval | HumanEval+ | MBPP | MBPP+ |", "|---|---|---|---|---|",
          f"| Published (Qwen report, BF16) | {PUBLISHED['humaneval']} | {PUBLISHED['humaneval_plus']} | {PUBLISHED['mbpp']} | {PUBLISHED['mbpp_plus']} |"]
    for m, kv in CONFIGS:
        he, mb = latest("humaneval", m, kv), latest("mbpp", m, kv)
        if not (he or mb):
            continue
        data["coding"][f"{m}_kv{kv}"] = {"humaneval": he, "mbpp": mb}
        L.append(f"| {LABEL[m]}, KV {kv} | {pct(he and he['humaneval'])} | {pct(he and he['humaneval_plus'])} | "
                 f"{pct(mb and mb['mbpp'])} | {pct(mb and mb['mbpp_plus'])} |")

    L += ["", "## Agent task suite (20 hidden-test tasks)", "",
          "| Config | Pass | Bugfix | Feature | Implement | Performance | Refactor | Write-tests | Truncated |",
          "|---|---|---|---|---|---|---|---|---|"]
    per_task = {}
    for m, kv in CONFIGS:
        a = latest("agent_suite", m, kv)
        if not a:
            continue
        data["agent"][f"{m}_kv{kv}"] = a
        c = a["by_category"]
        L.append(f"| {LABEL[m]}, KV {kv} | **{a['pass']}/{a['total']}** | {c.get('bugfix')} | {c.get('feature')} | "
                 f"{c.get('implement')} | {c.get('performance')} | {c.get('refactor')} | {c.get('write-tests')} | {a.get('truncated')} |")
        for r in agent_detail(f"{m}_kv{kv}") or []:
            per_task.setdefault(r["id"], {})[f"{LABEL[m]}/{kv}"] = "✓" if r["pass"] else "✗"
    if per_task:
        cols = list(next(iter(per_task.values())).keys())
        L += ["", "| Task | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for t, v in sorted(per_task.items()):
            L.append(f"| {t} | " + " | ".join(v.get(c, "—") for c in cols) + " |")

    L += ["", "## General reasoning & instruction following (lm-eval, %)", ""]
    runs = [f"{m}_kv{kv}" for m, kv in CONFIGS if (R / "lmeval" / f"{m}_kv{kv}").exists()]
    for run in runs:
        data["lmeval"][run] = lmeval(run)
    metrics = sorted({k for v in data["lmeval"].values() for k in v})
    if metrics:
        L += ["| Metric | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
        for k in metrics:
            cells = []
            for run in runs:
                v = data["lmeval"][run].get(k)
                cells.append("—" if not v else f"{100 * v[0]:.1f}" + (f" ± {100 * v[1]:.1f}" if v[1] else ""))
            L.append(f"| {k} | " + " | ".join(cells) + " |")

    if data["kld"]:
        L += ["", "## Quantization fidelity vs Q8_0 (wikitext-2, 100×512 tokens)", "",
              "| Config | PPL | Mean KLD | 99.9% KLD | Max KLD | Same top token | RMS Δp |", "|---|---|---|---|---|---|---|"]
        for k, v in data["kld"].items():
            v = {kk: (re.sub(r"\s+", " ", vv) if isinstance(vv, str) else vv) for kk, vv in v.items()}
            L.append(f"| {k} | {v['ppl']} | {v['kld_mean'] or '— (base)'} | {v['kld_99_9'] or '—'} | {v['kld_max'] or '—'} | {v['same_top'] or '—'} | {v['rms_dp'] or '—'} |")

    bench = R / "bench_kv.jsonl"
    if bench.exists():
        data["speed"] = {}
        for l in open(bench):
            b = json.loads(l)
            key = f"{b['type_k']}/{b['type_v']}"
            col = f"{'pp512' if b['n_prompt'] else 'tg128'} @ depth {b['n_depth']}"
            data["speed"].setdefault(key, {})[col] = round(b["avg_ts"], 1)
        cols = ["pp512 @ depth 0", "tg128 @ depth 0", "pp512 @ depth 4096", "tg128 @ depth 4096"]
        L += ["", "## Speed by KV cache type (llama-bench, Q4_K_M, FA on, tokens/s)", "",
              "| K / V | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for k, v in data["speed"].items():
            L.append(f"| {k} | " + " | ".join(str(v.get(c, "—")) for c in cols) + " |")

    (R / "REPORT.md").write_text("\n".join(L) + "\n")
    (R / "report_data.json").write_text(json.dumps(data, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
