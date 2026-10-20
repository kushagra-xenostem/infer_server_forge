import json
import time
import pytest
from pathlib import Path
from aiobs import Event, EventLog, ResourceSampler, TokenCounter, percentiles, span, summarize, write_report


def test_event_roundtrip(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "e" / "events.jsonl")
    log.emit("fault", "upstream_drop", detect_s=1.5)
    log.write(Event(kind="metric", name="ttft", duration_s=0.2, attrs={"n": 3}))
    evs = log.read()
    assert [e.name for e in evs] == ["upstream_drop", "ttft"]
    assert evs[0].attrs == {"detect_s": 1.5} and evs[1].duration_s == 0.2 and evs[0].schema == 1


def test_span_times_and_logs_errors(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    with span(log, "ok", doc="a") as extra:
        time.sleep(0.02)
        extra["pages"] = 2
    with pytest.raises(ValueError), span(log, "bad"):
        raise ValueError("boom")
    ok, bad = log.read()
    assert ok.status == "ok" and ok.attrs == {"doc": "a", "pages": 2} and ok.duration_s and ok.duration_s >= 0.02
    assert bad.status == "error" and bad.attrs["error"] == "ValueError"


def test_span_without_a_log_is_a_no_op() -> None:
    with span(None, "x"):
        pass


def test_token_counter_llama_timings_and_openai_usage() -> None:
    t = TokenCounter()
    t.add(
        usage={"completion_tokens": 10},
        timings={"prompt_n": 100, "cache_n": 900, "prompt_ms": 500, "predicted_ms": 1000},
    )
    t.add(usage={"prompt_tokens": 1000, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 100}})
    assert t.calls == 2 and t.prompt_tokens == 2000 and t.cached_tokens == 1000 and t.completion_tokens == 30
    assert t.cached_share == 0.5 and t.prefill_s == 0.5 and t.decode_tps == 30.0
    assert TokenCounter().cached_share is None


def test_percentiles_are_nearest_rank() -> None:
    assert percentiles([]) == {"p50": None, "p95": None, "p99": None}
    xs = list(range(1, 101))
    assert percentiles(xs) == {"p50": 50, "p95": 95, "p99": 99}
    assert percentiles([3.0, 1.0, 2.0], (0.5,)) == {"p50": 2.0}
    assert summarize([1.0, float("nan"), 3.0])["mean"] == 2.0 and summarize([])["n"] == 0


def test_resource_sampler_always_has_a_sample() -> None:
    with ResourceSampler(interval_s=10) as r:
        pass
    s = r.summary()
    assert s["samples"] == 1 and s["proc_rss_mb_peak"] > 0


def test_resource_sampler_samples_during_work() -> None:
    with ResourceSampler(interval_s=0.02) as r:
        end = time.perf_counter() + 0.2
        while time.perf_counter() < end:
            pass
    assert r.summary()["samples"] >= 3


def test_write_report_json_and_markdown(tmp_path: Path) -> None:
    j, m = write_report(
        tmp_path / "r" / "bench",
        {"summary": {"ttft_p50": 1.23456, "n": 4}, "rows": [{"sessions": 1, "ok": 6}, {"sessions": 2, "ok": 12}]},
        title="Bench",
    )
    data = json.loads(j.read_text())
    assert data["summary"]["n"] == 4 and data["rows"][1]["ok"] == 12 and "python" in data["environment"]
    md = m.read_text()
    assert "# Bench" in md and "| ttft_p50 | 1.235 |" in md and "| 2 | 12 |" in md
