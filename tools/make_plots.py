#!/usr/bin/env python3
"""Regenerate every chart in plots/ from the raw data in benchmarks/results/.

Usage:  python tools/make_plots.py        (needs matplotlib)
Writes plots/<name>-light.png and plots/<name>-dark.png (README uses <picture> to pick by theme).
Palette: dataviz reference palette (validated categorical slots 1-3 + reserved status colors).
"""
import json
import pathlib
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "benchmarks" / "results"
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="#e4e3df",
                  s=["#2a78d6", "#eb6834", "#1baf7a"], good="#0ca30c", critical="#d03b3b", muted="#9a9890"),
    "dark": dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7", grid="#383835",
                 s=["#3987e5", "#d95926", "#199e70"], good="#0ca30c", critical="#d03b3b", muted="#6f6e68"),
}


def setup(t, w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(t["grid"])
    ax.tick_params(colors=t["text2"], labelsize=9)
    ax.grid(color=t["grid"], linewidth=0.6); ax.set_axisbelow(True)
    return fig, ax


def finish(fig, ax, t, name, theme, title, subtitle, xlabel=None, ylabel=None, legend=True, legend_below=False):
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=t["text"])
    ax.set_title("\n".join(textwrap.wrap(subtitle, 92)), loc="left", fontsize=9, color=t["text2"], pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=t["text2"], fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=t["text2"], fontsize=9)
    if legend and ax.get_legend_handles_labels()[0]:
        if legend_below:
            ax.legend(frameon=False, fontsize=9, labelcolor=t["text"], loc="upper center",
                      bbox_to_anchor=(0.5, -0.09), ncol=3)
        else:
            ax.legend(frameon=False, fontsize=9, labelcolor=t["text"], loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}-{theme}.png", facecolor=t["surface"])
    plt.close(fig)


def summary():
    rows = [json.loads(l) for l in open(RES / "summary.jsonl")]
    last = {}
    for r in rows:
        last[(r["benchmark"], r["model"], r["kv_type"])] = r["metrics"]
    return last


# ------------------------------------------------------------------ 1. bake-off
def bakeoff(theme):
    t = THEMES[theme]; S = summary()
    models = [("Qwen2.5-Coder-7B", "q4km", "f16"), ("Qwen3.5-9B", "q35_9b", "q8_0-q8_0"),
              ("Qwen3-Coder-30B-A3B", "q30coder", "q8_0-q8_0"), ("Qwen3.6-35B-A3B", "q36_35b", "q8_0-q8_0")]
    series = [("HumanEval+", lambda m, k: 100 * S[("humaneval", m, k)]["humaneval_plus"]),
              ("MBPP+", lambda m, k: 100 * S[("mbpp", m, k)]["mbpp_plus"]),
              ("Agent tasks (20)", lambda m, k: 100 * S[("agent_suite", m, k)]["pass"] / 20)]
    fig, ax = setup(t, 9, 5.2)
    h = 0.24
    for si, (lab, fn) in enumerate(series):
        ys = [i + (1 - si) * (h + 0.03) for i in range(len(models))]
        vals = [fn(m, k) for _, m, k in models]
        ax.barh(ys, vals, height=h, color=t["s"][si], label=lab, zorder=3)
        for y, v in zip(ys, vals):
            ax.text(v + 0.8, y, f"{v:.0f}" if lab.startswith("Agent") else f"{v:.1f}",
                    va="center", fontsize=8, color=t["text2"])
    ax.set_yticks(range(len(models))); ax.set_yticklabels([m[0] for m in models], color=t["text"], fontsize=10)
    ax.set_xlim(0, 105); ax.grid(axis="y", visible=False)
    finish(fig, ax, t, "bakeoff", theme, "Coding bake-off: 4 free models on the same box",
           "pass rate %, greedy decoding, non-thinking mode, each model at its best fitting config (RX 6700 10 GB)",
           xlabel="% passed")


# ------------------------------------------------------------------ 2. KV cache quality
def kv_quality(theme):
    t = THEMES[theme]
    k = json.load(open(RES / "report_data.json"))["kld"]
    order = [("q4km_kvf16-f16", "K f16 / V f16"), ("q4km_kvq8_0-q8_0", "K q8_0 / V q8_0"),
             ("q4km_kvf16-q4_0", "K f16 / V q4_0"), ("q4km_kvq8_0-q4_0", "K q8_0 / V q4_0"),
             ("q4km_kvq4_0-f16", "K q4_0 / V f16"), ("q4km_kvq4_0-q4_0", "K q4_0 / V q4_0")]
    vals = [float(re.match(r"[\d.]+", k[key]["same_top"]).group()) for key, _ in order]
    fig, ax = setup(t, 9, 4.6)
    cols = [t["critical"] if "K q4_0" in lab else t["s"][0] for _, lab in order]
    ax.barh(range(len(order)), vals, color=cols, height=0.55, zorder=3)
    for i, v in enumerate(vals):
        broken = "K q4_0" in order[i][1]
        ax.text(v + 1, i, f"{v:.1f}%" + ("  ✗ broken (perplexity > 200)" if broken else ""), va="center", fontsize=9,
                color=t["text"] if broken else t["text2"])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([o[1] for o in order], color=t["text"])
    ax.invert_yaxis(); ax.set_xlim(0, 120); ax.grid(axis="y", visible=False)
    finish(fig, ax, t, "kv_cache_quality", theme, "A 4-bit K cache breaks Qwen2.5; a 4-bit V cache is fine",
           "Qwen2.5-Coder-7B Q4_K_M · agreement of top token with Q8_0 reference on wikitext-2 (100×512 tokens) · red = K cache at q4_0",
           xlabel="same top token as reference (%)", legend=False)


# ------------------------------------------------------------------ 3. KV cache speed
def kv_speed(theme):
    t = THEMES[theme]
    rows = [json.loads(l) for l in open(RES / "bench_kv.jsonl")]
    cfgs = ["f16/f16", "q8_0/q8_0", "q4_0/q4_0", "q8_0/q4_0", "f16/q4_0"]
    get = lambda c, d: next(r["avg_ts"] for r in rows if f'{r["type_k"]}/{r["type_v"]}' == c and r["n_depth"] == d and r["n_gen"] > 0)
    fig, ax = setup(t, 9, 4.6)
    h = 0.36
    for si, (d, lab) in enumerate([(0, "empty context"), (4096, "at 4k context")]):
        ys = [i + (si - 0.5) * (h + 0.03) for i in range(len(cfgs))]
        vals = [get(c, d) for c in cfgs]
        ax.barh(ys, vals, height=h, color=t["s"][si], label=f"generation, {lab}", zorder=3)
        for y, v in zip(ys, vals):
            ax.text(v + 0.6, y, f"{v:.1f}", va="center", fontsize=8, color=t["text2"])
    ax.set_yticks(range(len(cfgs))); ax.set_yticklabels([f"K/V {c}" for c in cfgs], color=t["text"])
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlim(0, 62)
    finish(fig, ax, t, "kv_cache_speed", theme, "Mixed K/V cache types fall off a cliff at longer context",
           "Qwen2.5-Coder-7B Q4_K_M, llama-bench tg128, Flash Attention on · mixed types have no fast GPU kernel in this build",
           xlabel="tokens / second")


# ------------------------------------------------------------------ 4. long-context session
def longctx(theme):
    t = THEMES[theme]
    s = json.load(open(RES / "longctx" / "session_64k.json"))
    ctx = [(r.get("cached") or 0) + (r.get("prompt_n") or 0) for r in s]
    wall = [r["wall_s"] for r in s]
    fig, ax = setup(t, 9, 4.4)
    ax.plot([c / 1000 for c in ctx], wall, color=t["s"][0], linewidth=2, marker="o", markersize=5, zorder=3,
            label="seconds per turn")
    ax.annotate(f"turn 16: {wall[-1]:.1f} s at {ctx[-1]/1000:.0f}k tokens, 16/16 answers correct",
                (ctx[-1] / 1000, wall[-1]), xytext=(-260, 18), textcoords="offset points", fontsize=9, color=t["text"],
                arrowprops=dict(arrowstyle="-", color=t["text2"], lw=0.8))
    ax.set_ylim(0, 12); ax.set_xlim(0, 46)
    finish(fig, ax, t, "longctx_session", theme, "Agent turns stay fast as the conversation grows",
           "Qwen3.6-35B-A3B, 64k context, simulated 16-turn coding session (+~2.5k tokens of code per turn) · prompt cache reuses earlier turns",
           xlabel="conversation size (thousand tokens)", ylabel="seconds per turn", legend=False)


# ------------------------------------------------------------------ 5. SWE-bench landscape
def swe(theme):
    t = THEMES[theme]
    cat = json.load(open(RES / "model_catalog.json"))
    paid = [(m["name"], m["swe_verified"], 0) for m in cat["paid_reference"]["models"]]
    openw = [(m["name"], m["swe_verified"], 1) for m in cat["models"]
             if m.get("swe_verified") and "Official" in m.get("swe_source", "") and "mini-SWE" in m["swe_source"]]
    ours = [("Qwen3.6-35B-A3B* (runs here)", 73.4, 2), ("Qwen3-Coder-30B-A3B† (runs here)", 51.6, 2)]
    rows = sorted(paid + openw + ours, key=lambda r: r[1])
    fig, ax = setup(t, 9, 6.4)
    labels = {0: "paid / proprietary", 1: "free open-weight (too large for this box)", 2: "runs on this box"}
    seen = set()
    for i, (n, v, g) in enumerate(rows):
        ax.barh(i, v, color=t["s"][g], height=0.62, zorder=3, hatch="//" if "*" in n else None,
                edgecolor=t["surface"], linewidth=0, label=None if g in seen else labels[g])
        seen.add(g)
        ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=8, color=t["text2"])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], color=t["text"], fontsize=9)
    ax.set_xlim(0, 88); ax.grid(axis="y", visible=False)
    finish(fig, ax, t, "swe_bench_landscape", theme, "Where a free local model stands on SWE-bench Verified",
           "official standardized runs (mini-SWE-agent, Feb 2026) · *Qwen-reported, own scaffold, full precision · †official, OpenHands scaffold",
           xlabel="% of 500 real GitHub issues resolved", legend_below=True)


# ------------------------------------------------------------------ 6. model fit map
def fit(theme):
    t = THEMES[theme]
    cat = json.load(open(RES / "model_catalog.json"))["models"]
    groups = [("tested", "tested here", t["s"][0]), ("bringup", "GPU bring-up", t["s"][0]),
              ("fits", "fits (not benchmarked)", t["s"][2]), ("split_slow", "dense > VRAM: slow when split", t["s"][1]),
              ("too_large", "too large for this box", t["muted"])]
    ms = sorted(cat, key=lambda m: m["params_b"])
    fig, ax = setup(t, 9, 7.8)
    for i, m in enumerate(ms):
        g = next(x for x in groups if x[0] == m["status"])
        ax.scatter(m["params_b"], i, s=60 if m["status"] in ("tested", "bringup") else 40, color=g[2], zorder=3,
                   marker="o" if m["status"] != "bringup" else "D", edgecolors=t["surface"], linewidths=1.5)
        ax.text(m["params_b"] * 1.15, i, m["name"], va="center", fontsize=8,
                color=t["text"] if m["status"] == "tested" else t["text2"],
                fontweight="bold" if m["status"] == "tested" else "normal")
    for key, lab, col in groups:
        ax.scatter([], [], color=col, label=lab, marker="D" if key == "bringup" else "o")
    ax.axvspan(0.3, 40, color=t["s"][2], alpha=0.06, zorder=0)
    ax.text(0.35, len(ms) - 0.6, "practical range on 10 GB VRAM + 13.6 GiB RAM", fontsize=8, color=t["text2"])
    ax.set_xscale("log"); ax.set_xlim(0.3, 8000); ax.set_yticks([]); ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    finish(fig, ax, t, "model_fit", theme, f"{len(ms)} free models screened for one 10 GB GPU",
           "total parameters (log scale, from Hugging Face metadata) · bold = benchmarked on this server",
           xlabel="total parameters (billions, log scale)")


# ------------------------------------------------------------------ 7. real Cline session
def cline(theme):
    t = THEMES[theme]
    rows = [json.loads(l) for l in open(RES / "gateway" / "cline_session_metrics.jsonl")]
    rows = [r for r in rows if r["status"] == 200 and r.get("ttft_s") is not None]
    ctx = [((r["prompt_n"] or 0) + (r["cached_n"] or 0)) / 1000 for r in rows]
    ttft = [r["ttft_s"] for r in rows]
    fig, ax = setup(t, 9, 4.4)
    ax.scatter(ctx, ttft, s=28, color=t["s"][0], alpha=0.85, zorder=3, edgecolors=t["surface"], linewidths=1)
    med = sorted(ttft)[len(ttft) // 2]
    ax.axhline(med, color=t["text2"], linewidth=1, linestyle="--", zorder=2)
    ax.annotate(f"median time to first token: {med:.1f} s", (8, med), xytext=(8, med + 9), fontsize=9,
                color=t["text"], arrowprops=dict(arrowstyle="-", color=t["text2"], lw=0.8))
    ax.set_ylim(0, max(10, max(ttft) * 1.1))
    finish(fig, ax, t, "cline_session", theme, f"Real Cline session: {len(rows)} agent requests, 0 errors",
           "Qwen3.6-35B-A3B via the logging gateway · each dot = one request (tool calls, file reads, edits)",
           xlabel="context size of the request (thousand tokens)", ylabel="time to first token (s)", legend=False)


# ------------------------------------------------------------------ 8. speed journey
def journey(theme):
    t = THEMES[theme]
    rows = [("30B · layers split, KV in RAM\n(owner-reported)", 1.0, t["muted"], "~0.5–1"),
            ("30B · experts on CPU, 32k ctx", 15.0, t["s"][1], "15"),
            ("30B · experts on CPU, 8k ctx", 19.0, t["s"][1], "19"),
            ("35B · Qwen3.6, 64k ctx", 17.4, t["s"][0], "17.4"),
            ("35B · real Cline session (median)", 15.7, t["s"][0], "15.7")]
    fig, ax = setup(t, 9, 4.4)
    for i, (lab, v, c, txt) in enumerate(rows):
        ax.barh(i, v, color=c, height=0.6, zorder=3)
        ax.text(v + 0.3, i, f"{txt} tok/s", va="center", fontsize=9, color=t["text"])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], color=t["text"], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 24); ax.grid(axis="y", visible=False)
    finish(fig, ax, t, "speed_journey", theme, "From ~1 tok/s to ~17 tok/s, on a bigger model",
           "generation speed on the same RX 6700 10 GB + 13.6 GiB RAM box · fix: attention on GPU, MoE experts on CPU, no pinned/repacked copies",
           xlabel="tokens / second (generation)", legend=False)


if __name__ == "__main__":
    for theme in THEMES:
        for f in (bakeoff, kv_quality, kv_speed, longctx, swe, fit, cline, journey):
            f(theme)
    print("wrote", len(list(OUT.glob("*.png"))), "files to", OUT)
