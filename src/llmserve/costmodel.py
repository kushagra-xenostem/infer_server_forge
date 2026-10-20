"""Service-time model of the real server, fitted to llama.cpp `timings` from real traffic.

prefill_s = overhead_s + new_tokens / prefill_tps + cached_tokens * cached_s_per_token
decode_s_per_token = decode_s_per_token + context_tokens * decode_s_per_ctx_token

The defaults are the least-squares fit to the 98-request Cline session in
benchmarks/results/gateway/cline_session_metrics.jsonl (median relative error of prefill 6%).
Refit with `CostModel.fit(rows)`.
"""

import json
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Iterable, Mapping


@dataclass(frozen=True)
class CostModel:
    overhead_s: float = 1.062
    prefill_tps: float = 582.0
    cached_s_per_token: float = 1.25e-5
    decode_s_per_token: float = 0.0598
    decode_s_per_ctx_token: float = 1.37e-7

    def prefill_s(self, new_tokens: int, cached_tokens: int) -> float:
        return self.overhead_s + new_tokens / self.prefill_tps + cached_tokens * self.cached_s_per_token

    def token_s(self, context_tokens: int) -> float:
        """Seconds to decode one token with `context_tokens` already in the KV cache."""
        return self.decode_s_per_token + context_tokens * self.decode_s_per_ctx_token

    def decode_s(self, context_tokens: int, n_tokens: int) -> float:
        # sum of token_s over a growing context
        return n_tokens * self.decode_s_per_token + self.decode_s_per_ctx_token * (
            n_tokens * context_tokens + n_tokens * (n_tokens - 1) / 2
        )

    @classmethod
    def fit(cls, rows: Iterable[Mapping[str, Any]]) -> "CostModel":
        """Least-squares fit from gateway log rows (prompt_n, cached_n, prefill_s, gen_tps)."""
        rows = list(rows)
        pre = [r for r in rows if r.get("prefill_s") and r.get("prompt_n") is not None]
        a, b, c = _lstsq([[1.0, r["prompt_n"], r.get("cached_n") or 0] for r in pre], [r["prefill_s"] for r in pre])
        dec = [r for r in rows if r.get("gen_tps")]
        d, e = _lstsq(
            [[1.0, (r.get("cached_n") or 0) + (r.get("prompt_n") or 0)] for r in dec], [1 / r["gen_tps"] for r in dec]
        )
        return cls(
            overhead_s=a, prefill_tps=1 / b, cached_s_per_token=c, decode_s_per_token=d, decode_s_per_ctx_token=e
        )


# The Cline-log fit (the class defaults): 98 real requests, 2026-10-06.
CLINE_2026_10_06 = CostModel()
# Cold prefill refit to the live black-box runs of 2026-10-18 (1-2 session runs; out-of-sample median error
# +2.9% on the 4-session run, vs +26% for the Oct 6 fit). Decode and cache-hit terms keep the Oct 6 fit,
# which matched decode within 1.5%. Source: benchmarks/results/load/analysis.json ("validation").
LIVE_2026_10_18 = CostModel(overhead_s=0.733, prefill_tps=699.7)
PRESETS = {"cline": CLINE_2026_10_06, "live": LIVE_2026_10_18}


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _lstsq(x: list[list[float]], y: list[float]) -> list[float]:
    """Ordinary least squares via the normal equations (small, well-conditioned problems only)."""
    n = len(x[0])
    if len(x) < n:
        raise ValueError(f"need at least {n} rows, got {len(x)}")
    # scale columns so the normal equations stay well conditioned
    scale = [max(abs(row[j]) for row in x) or 1.0 for j in range(n)]
    xs = [[row[j] / scale[j] for j in range(n)] for row in x]
    m = [
        [sum(r[i] * r[j] for r in xs) for j in range(n)] + [sum(r[i] * t for r, t in zip(xs, y, strict=True))]
        for i in range(n)
    ]
    for col in range(n):  # Gauss-Jordan with partial pivoting
        piv = max(range(col, n), key=lambda i: abs(m[i][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-12:
            raise ValueError("singular system: columns are collinear")
        for i in range(n):
            if i != col:
                f = m[i][col] / m[col][col]
                m[i] = [vi - f * vc for vi, vc in zip(m[i], m[col], strict=True)]
    return [m[i][n] / m[i][i] / scale[i] for i in range(n)]
