import json
import pytest
from pathlib import Path
from llmserve.cli import main, read_env
from llmserve.mock_upstream import n_tokens
from llmserve.sim import SimConfig, run_sim
from llmserve.loadgen import VOCAB, Result, Workload, filler, percentile, summarize

TINY = Workload(kind="agent", turns=3, first_context=(500, 500), new_tokens=(50,), gen_tokens=(20,), think_s=(0.0, 0.0))


def test_filler_words_are_one_mock_token_each() -> None:
    import random

    assert all(len(w) == 3 for w in VOCAB)
    text = filler(random.Random(0), 250)
    assert n_tokens(text + " ") == 250


def test_single_agent_session_reuses_its_prefix() -> None:
    summary, results = run_sim(TINY, SimConfig(sessions=1, time_scale=0.01))
    assert summary["ok"] == 3 and summary["errors"] == 0
    turns = sorted(results, key=lambda r: r.turn)
    assert turns[0].prompt_cached == 0
    for r in turns[1:]:  # every later turn finds the previous conversation, reply included, in the cache
        assert r.prompt_cached and r.prompt_new is not None and r.prompt_new < 90  # 50 new words + the count prompt


def test_two_sessions_on_one_slot_evict_each_others_cache() -> None:
    _, results = run_sim(TINY, SimConfig(sessions=2, time_scale=0.01))
    later = [r for r in results if r.turn > 0]
    # with one KV cache, interleaved sessions often find the other session's prompt resident
    assert any((r.prompt_new or 0) > 500 for r in later)


def test_chat_requests_are_independent() -> None:
    w = Workload(kind="chat", turns=3, new_tokens=(40,), gen_tokens=(10,), think_s=(0.0, 0.0))
    summary, results = run_sim(w, SimConfig(sessions=1, time_scale=0.01))
    assert summary["ok"] == 3
    # only the shared template header + '[session 0 turn ' is reused (~6 tokens), as with a real chat template
    assert all((r.prompt_cached or 0) < 12 for r in results)


def test_rejected_clients_honour_retry_after_and_retry() -> None:
    w = Workload(kind="chat", turns=1, new_tokens=(40,), gen_tokens=(400,), think_s=(0.0, 0.0))
    summary, results = run_sim(w, SimConfig(sessions=3, max_queue=0, time_scale=0.05))
    assert summary["rejected_429"] >= 1
    last = {}
    for r in sorted(results, key=lambda r: r.attempts):
        last[(r.session, r.turn)] = r
    assert any(r.attempts > 1 for r in last.values())  # someone was turned away and came back
    assert all(r.status == 200 for r in last.values())
    assert summary["ok"] == 3 and summary["gave_up"] == 0


def test_summarize_counts_only_the_last_attempt() -> None:
    rs = [
        Result(0, 0, 0.0, 429, retry_after_s=4, attempts=1),
        Result(0, 0, 4.0, 200, ttft_s=1.0, e2e_s=3.0, gen_tokens=10, attempts=2),
        Result(1, 0, 0.0, 429, retry_after_s=4, attempts=1),
        Result(2, 0, 0.0, 200, ttft_s=2.0, e2e_s=5.0, gen_tokens=10, error="deadline_exceeded"),
    ]
    s = summarize(rs, wall_s=10)
    assert (s["requests"], s["ok"], s["rejected_429"], s["gave_up"], s["errors"]) == (3, 1, 2, 1, 1)
    assert s["ttft_s"]["p50"] == 1.0 and s["gen_tok_per_s"] == 1.0


def test_percentile() -> None:
    assert percentile([], 0.5) is None
    assert percentile([3, 1, 2], 0.5) == 2
    assert percentile(list(range(100)), 0.99) == 99


def test_read_env_handles_spaces_and_quotes(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("# comment\nLLM_BASE_URL = \"https://x/v1\"\nLLM_API_KEY=abc\n  LLM_MODEL = 'm'\n", encoding="utf-8")
    assert read_env(p) == {"LLM_BASE_URL": "https://x/v1", "LLM_API_KEY": "abc", "LLM_MODEL": "m"}


def test_cli_sim_writes_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "sim",
            "--sessions",
            "1",
            "--turns",
            "2",
            "--max-gen",
            "20",
            "--first-context",
            "200",
            "200",
            "--think",
            "0",
            "0",
            "--time-scale",
            "0.01",
            "--out",
            str(tmp_path),
            "--tag",
            "t",
        ]
    )
    assert rc == 0
    rows = [json.loads(line) for line in (tmp_path / "sim.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["tag"] == "t" and rows[0]["status"] == 200
    summary = json.loads((tmp_path / "sim_summary.jsonl").read_text())
    assert summary["ok"] == 2
    assert '"sessions": 1' in capsys.readouterr().out
