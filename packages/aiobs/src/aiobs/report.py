"""Percentiles, summaries, and one report written as both JSON and Markdown."""

import sys
import json
import math
import time
import platform
from typing import Any
from pathlib import Path
from collections.abc import Mapping, Sequence


def percentiles(xs: Sequence[float], qs: Sequence[float] = (0.5, 0.95, 0.99)) -> dict[str, float | None]:
    """Nearest-rank percentiles (no interpolation: every value reported was actually observed)."""
    s = sorted(x for x in xs if x is not None and not math.isnan(x))
    out: dict[str, float | None] = {}
    for q in qs:
        key = f"p{round(q * 100):g}"
        out[key] = s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))] if s else None
    return out


def summarize(xs: Sequence[float]) -> dict[str, Any]:
    s = [x for x in xs if x is not None and not math.isnan(x)]
    if not s:
        return {"n": 0}
    return {"n": len(s), "mean": sum(s) / len(s), "min": min(s), "max": max(s), **percentiles(s)}


def environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _md_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, Mapping):
        return ", ".join(f"{k}={_md_value(x)}" for k, x in v.items())
    return str(v)


def write_report(stem: str | Path, sections: Mapping[str, Any], title: str = "Report") -> tuple[Path, Path]:
    """Write `<stem>.json` and `<stem>.md` with the same numbers. Each section is a dict (shown as a
    two-column table) or a list of dicts (shown as one table, columns from the first row)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    data = {"environment": environment(), **sections}
    json_path, md_path = stem.with_suffix(".json"), stem.with_suffix(".md")
    json_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    lines = [f"# {title}", ""]
    for name, sec in data.items():
        lines += [f"## {name}", ""]
        if isinstance(sec, Mapping):
            lines += ["| key | value |", "|---|---|"] + [f"| {k} | {_md_value(v)} |" for k, v in sec.items()]
        elif isinstance(sec, Sequence) and not isinstance(sec, str) and sec and isinstance(sec[0], Mapping):
            cols = list(sec[0])
            lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            lines += ["| " + " | ".join(_md_value(row.get(c)) for c in cols) + " |" for row in sec]
        else:
            lines.append(_md_value(sec))
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
