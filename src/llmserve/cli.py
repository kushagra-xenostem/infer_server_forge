"""Command-line entry point: `llmserve <command>`.

llmserve sim   --sessions 1,2,4 [--max-queue N]   load the local simulator (gateway + calibrated mock)
llmserve load  --sessions 1,2                      load the real endpoint from .env (LLM_BASE_URL, ...)
llmserve fit   <gateway.jsonl>                     fit the service-time model to a gateway log
"""

import re
import json
import time
import asyncio
import argparse
from pathlib import Path
from dataclasses import asdict
from llmserve import __version__
from aiobs import ResourceSampler
from llmserve.sim import SimConfig, run_sim
from llmserve.costmodel import PRESETS, CostModel, load_rows
from llmserve.loadgen import Runner, Target, Workload, summarize, write_jsonl

CLINE_LOG = Path("benchmarks/results/gateway/cline_session_metrics.jsonl")


def read_env(path: str | Path) -> dict[str, str]:
    """KEY = value lines; spaces around '=' and quotes allowed (the .env files in this portfolio use both)."""
    env: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m and not line.lstrip().startswith("#"):
            env[m.group(1)] = m.group(2).strip("\"'")
    return env


def workload_from_args(a: argparse.Namespace) -> Workload:
    common = {
        "kind": a.workload,
        "turns": a.turns,
        "max_gen": a.max_gen,
        "max_context": a.max_context,
        "first_context": (a.first_context[0], a.first_context[1]),
        "think_s": (a.think[0], a.think[1]),
    }
    if CLINE_LOG.exists():
        return Workload.from_log(CLINE_LOG, **common)
    return Workload(**common)


def add_workload_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sessions", default="1", help="comma-separated concurrency levels, e.g. 1,2,4")
    p.add_argument("--workload", choices=["agent", "chat"], default="agent")
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--max-gen", type=int, default=1500)
    p.add_argument("--max-context", type=int, default=60000)
    p.add_argument("--first-context", type=int, nargs=2, default=[3000, 12000], metavar=("MIN", "MAX"))
    p.add_argument("--think", type=float, nargs=2, default=[2.0, 20.0], metavar=("MIN", "MAX"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="benchmarks/results/load", help="directory for JSONL results")
    p.add_argument("--tag", default="", help="label stored with every row")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmserve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"llmserve {__version__}")
    sub = parser.add_subparsers(dest="command")

    sim = sub.add_parser("sim", help="run the load generator against the local simulator")
    add_workload_args(sim)
    sim.add_argument("--max-queue", type=int, default=None, help="enable admission control with this queue bound")
    sim.add_argument("--deadline", type=float, default=None, help="per-request deadline, simulated seconds")
    sim.add_argument("--time-scale", type=float, default=0.25)
    sim.add_argument("--cost", choices=sorted(PRESETS), default="live", help="service-time model preset")
    sim.add_argument("--slots", type=int, default=1, help="llama-server -np")
    sim.add_argument("--beta", type=float, default=0.6, help="batched-decode slowdown per extra slot")
    sim.add_argument("--policy", choices=["fifo", "affinity"], default="fifo")
    sim.add_argument("--hold", type=float, default=0.0, help="anticipatory hold, simulated seconds")
    sim.add_argument("--hold-max-wait", type=float, default=60.0, help="hold fairness guard (1e9 = off)")

    load = sub.add_parser("load", help="run the load generator against the real endpoint")
    add_workload_args(load)
    load.add_argument("--env", default=".env")

    fit = sub.add_parser("fit", help="fit the service-time model to a gateway JSONL log")
    fit.add_argument("log", nargs="?", default=str(CLINE_LOG))

    a = parser.parse_args(argv)
    if a.command is None:
        parser.print_help()
        return 0
    if a.command == "fit":
        print(json.dumps(asdict(CostModel.fit(load_rows(a.log))), indent=2))
        return 0

    w = workload_from_args(a)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for n in [int(x) for x in a.sessions.split(",")]:
        meta = {
            "command": a.command,
            "tag": a.tag,
            "sessions": n,
            "workload": w.kind,
            "turns": w.turns,
            "max_gen": w.max_gen,
            "seed": a.seed,
            "stamp": stamp,
        }
        if a.command == "sim":
            meta.update(
                max_queue=a.max_queue,
                deadline_s=a.deadline,
                time_scale=a.time_scale,
                cost=a.cost,
                slots=a.slots,
                beta=a.beta,
                policy=a.policy,
                hold_s=a.hold,
                hold_max_wait_s=a.hold_max_wait,
                max_context=w.max_context,
            )
            cfg = SimConfig(
                sessions=n,
                max_queue=a.max_queue,
                deadline_s=a.deadline,
                time_scale=a.time_scale,
                seed=a.seed,
                cost=PRESETS[a.cost],
                n_slots=a.slots,
                batch_beta=a.beta,
                queue_policy=a.policy,
                hold_s=a.hold,
                hold_max_wait_s=a.hold_max_wait,
            )
            summary, results = run_sim(w, cfg)
        else:
            env = read_env(a.env)
            target = Target(
                base_url=env["LLM_BASE_URL"], api_key=env.get("LLM_API_KEY"), model=env.get("LLM_MODEL", "default")
            )
            runner = Runner(target, w, sessions=n, seed=a.seed)
            with ResourceSampler(interval_s=1.0) as res:  # shows the load generator itself is not the bottleneck
                results = asyncio.run(runner.run())
            summary = {**summarize(results, runner.wall_s), "client": res.summary()}
        out = Path(a.out)
        write_jsonl(out / f"{a.command}.jsonl", results, meta)
        with (out / f"{a.command}_summary.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({**meta, **summary}) + "\n")
        print(json.dumps({"sessions": n, **summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
