"""Background CPU / memory sampling with psutil."""

import time
import psutil
import threading
from typing import Any
from types import TracebackType


class ResourceSampler:
    """`with ResourceSampler() as r: ...` samples this process (and the machine) every `interval_s`."""

    def __init__(self, interval_s: float = 0.5, pid: int | None = None) -> None:
        self.interval_s = interval_s
        self.proc = psutil.Process(pid)
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.t0 = 0.0
        self.t1 = 0.0

    def _sample(self) -> None:
        self.samples.append(
            {
                "t": time.perf_counter() - self.t0,
                "proc_cpu_pct": self.proc.cpu_percent(None),
                "proc_rss_mb": self.proc.memory_info().rss / 2**20,
                "sys_cpu_pct": psutil.cpu_percent(None),
                "sys_mem_used_pct": psutil.virtual_memory().percent,
            }
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> "ResourceSampler":
        self.t0 = time.perf_counter()
        self.proc.cpu_percent(None)  # prime the counters: the first call always returns 0
        psutil.cpu_percent(None)
        self._thread.start()
        return self

    def __exit__(self, et: type[BaseException] | None, e: BaseException | None, tb: TracebackType | None) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()  # always at least one sample, even for a block shorter than the interval
        self.t1 = time.perf_counter()

    def summary(self) -> dict[str, Any]:
        def col(k: str) -> list[float]:
            return [s[k] for s in self.samples]

        if not self.samples:
            return {"samples": 0}
        return {
            "samples": len(self.samples),
            "duration_s": round(self.t1 - self.t0, 3),
            "proc_rss_mb_peak": round(max(col("proc_rss_mb")), 1),
            "proc_cpu_pct_mean": round(sum(col("proc_cpu_pct")) / len(self.samples), 1),
            "sys_cpu_pct_mean": round(sum(col("sys_cpu_pct")) / len(self.samples), 1),
            "sys_mem_used_pct_peak": round(max(col("sys_mem_used_pct")), 1),
        }
