"""Local simulator: this repo's gateway in front of the calibrated mock upstream, driven by the load generator.

Everything runs on localhost, so it is reproducible and needs no GPU. All times are simulated seconds:
the mock and the client both run `time_scale` times faster, and the gateway's own timeouts are scaled to
match. The one distortion: Retry-After is whole real seconds, so at time_scale 0.25 it is rounded up to a
multiple of 4 simulated seconds.
"""

import asyncio
import tempfile
from typing import Any
from pathlib import Path
from llmserve.servers import serve
from llmserve.costmodel import CostModel
from dataclasses import dataclass, replace
from llmserve.gateway import Settings, create_app
from llmserve.mock_upstream import MockState, create_mock
from llmserve.loadgen import Result, Runner, Target, Workload, summarize


@dataclass(frozen=True)
class SimConfig:
    sessions: int = 1
    max_queue: int | None = None  # None = no admission control
    queue_timeout_s: float = 300.0
    deadline_s: float | None = None
    time_scale: float = 0.25
    seed: int = 0
    cost: CostModel = CostModel()
    ctx_size: int = 65536  # total context, split across slots (the real server: 64k)
    n_slots: int = 1  # llama-server -np
    batch_beta: float = 0.6  # assumption, see mock_upstream
    queue_policy: str = "fifo"
    hold_s: float = 0.0
    hold_max_wait_s: float = 60.0


def run_sim(
    workload: Workload, cfg: SimConfig, log_dir: str | Path | None = None
) -> tuple[dict[str, Any], list[Result]]:
    state = MockState(
        cost=cfg.cost, time_scale=cfg.time_scale, ctx_size=cfg.ctx_size, n_slots=cfg.n_slots, batch_beta=cfg.batch_beta
    )
    log_dir = Path(log_dir or tempfile.mkdtemp(prefix="llmserve-sim-"))
    settings = Settings(
        log_path=str(log_dir / "gateway.jsonl"),
        log_preview=False,
        max_queue=cfg.max_queue,
        queue_timeout_s=cfg.queue_timeout_s * cfg.time_scale,
        deadline_s=None if cfg.deadline_s is None else cfg.deadline_s * cfg.time_scale,
        max_inflight=cfg.n_slots,
        queue_policy=cfg.queue_policy,
        hold_s=cfg.hold_s * cfg.time_scale,
        hold_max_wait_s=cfg.hold_max_wait_s * cfg.time_scale,
    )
    with serve(create_mock(state)) as mock_url, serve(create_app(replace(settings, upstream=mock_url))) as gw_url:
        runner = Runner(
            Target(base_url=f"{gw_url}/v1", model="mock-model", timeout_s=3600),
            workload,
            sessions=cfg.sessions,
            seed=cfg.seed,
            time_scale=cfg.time_scale,
        )
        results = asyncio.run(runner.run())
    return summarize(results, runner.wall_s), results
