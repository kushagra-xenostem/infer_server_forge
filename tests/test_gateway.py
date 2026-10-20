import json
import httpx
import pytest
from pathlib import Path
from llmserve.gateway import SSEStats, describe, timings_fields
from llmserve.mock_upstream import MockState, n_tokens, prompt_text

BODY = {"messages": [{"role": "user", "content": "tok " * 100}], "max_tokens": 6}


def log_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_stream_passes_bytes_through_unchanged(gateway_factory, mock_url: str) -> None:  # type: ignore[no-untyped-def]
    direct = httpx.post(f"{mock_url}/v1/chat/completions", json={**BODY, "stream": True}).text
    with gateway_factory() as gw:
        via = httpx.post(f"{gw}/v1/chat/completions", json={**BODY, "stream": True}).text
    # identical apart from the timings, which differ because the second request hits the cache
    strip = [line for line in direct.splitlines() if "timings" not in line]
    assert [line for line in via.splitlines() if "timings" not in line] == strip


def test_stream_log_line(gateway_factory, log_path: Path) -> None:  # type: ignore[no-untyped-def]
    with gateway_factory() as gw:
        httpx.post(f"{gw}/v1/chat/completions", json={**BODY, "stream": True})
    (rec,) = log_lines(log_path)
    assert rec["status"] == 200
    assert rec["stream"] is True
    assert rec["finish"] == "length"
    assert rec["gen_n"] == 6
    assert rec["prompt_n"] == n_tokens(prompt_text(BODY)) and rec["cached_n"] == 0
    assert isinstance(rec["ttft_s"], float)
    assert "error" not in rec


def test_non_stream_log_line(gateway_factory, log_path: Path) -> None:  # type: ignore[no-untyped-def]
    with gateway_factory() as gw:
        r = httpx.post(f"{gw}/v1/chat/completions", json=BODY)
    assert r.status_code == 200
    (rec,) = log_lines(log_path)
    assert rec["status"] == 200 and rec["stream"] is False and rec["gen_n"] == 6 and rec["tool_calls"] == 0


def test_sampling_params_are_stripped_by_default(gateway_factory, mock_state: MockState) -> None:  # type: ignore[no-untyped-def]
    with gateway_factory() as gw:
        httpx.post(f"{gw}/v1/chat/completions", json={**BODY, "temperature": 0, "top_p": 0.1, "seed": 7})
    assert "temperature" not in mock_state.last_body and "top_p" not in mock_state.last_body
    assert mock_state.last_body["seed"] == 7  # seed is not a sampling param; it passes through


def test_keep_sampling_passes_them_through(gateway_factory, mock_state: MockState) -> None:  # type: ignore[no-untyped-def]
    with gateway_factory(keep_sampling=True) as gw:
        httpx.post(f"{gw}/v1/chat/completions", json={**BODY, "temperature": 0})
    assert mock_state.last_body["temperature"] == 0


def test_auth(gateway_factory) -> None:  # type: ignore[no-untyped-def]
    with gateway_factory(api_keys=frozenset({"k1"})) as gw:
        assert httpx.get(f"{gw}/v1/models").status_code == 401
        assert httpx.get(f"{gw}/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401
        r = httpx.get(f"{gw}/v1/models", headers={"Authorization": "Bearer k1"})
        assert r.status_code == 200 and r.json()["data"][0]["id"] == "mock-model"
        assert httpx.get(f"{gw}/health").status_code == 200  # health stays open for probes
        bad = httpx.post(f"{gw}/v1/chat/completions", json=BODY)
        assert bad.status_code == 401 and bad.json()["error"]["type"] == "invalid_api_key"


def test_no_keys_means_no_auth(gateway_factory) -> None:  # type: ignore[no-untyped-def]
    with gateway_factory() as gw:
        assert httpx.get(f"{gw}/v1/models").status_code == 200


@pytest.mark.parametrize("stream", [False, True])
def test_upstream_down_is_a_502(gateway_factory, log_path: Path, stream: bool) -> None:  # type: ignore[no-untyped-def]
    with gateway_factory(upstream="http://127.0.0.1:9", connect_timeout_s=2) as gw:
        r = httpx.post(f"{gw}/v1/chat/completions", json={**BODY, "stream": stream}, timeout=10)
        assert httpx.get(f"{gw}/v1/models").status_code == 502
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "upstream_unavailable"
    assert log_lines(log_path)[0]["status"] == 502


@pytest.mark.parametrize("stream", [False, True])
def test_upstream_error_status_is_kept(
    gateway_factory,
    mock_state: MockState,
    log_path: Path,  # type: ignore[no-untyped-def]
    stream: bool,
) -> None:
    mock_state.faults.error_status = 500
    with gateway_factory() as gw:
        r = httpx.post(f"{gw}/v1/chat/completions", json={**BODY, "stream": stream})
    assert r.status_code == 500  # the old gateway answered 200 and streamed the error body
    rec = log_lines(log_path)[0]
    assert rec["status"] == 500 and "injected" in str(rec["error"])


def test_upstream_drop_mid_stream_ends_with_an_error_event(
    gateway_factory,
    mock_state: MockState,  # type: ignore[no-untyped-def]
    log_path: Path,
) -> None:
    mock_state.faults.drop_after_tokens = 2
    with gateway_factory() as gw:
        r = httpx.post(f"{gw}/v1/chat/completions", json={**BODY, "stream": True})
    last = [line for line in r.text.splitlines() if line.startswith("data: ")][-1]
    assert json.loads(last[6:])["error"]["type"] == "upstream_error"
    assert "mid-stream" in str(log_lines(log_path)[0]["error"])


def test_malformed_chunk_is_forwarded_and_counted(
    gateway_factory,
    mock_state: MockState,  # type: ignore[no-untyped-def]
    log_path: Path,
) -> None:
    mock_state.faults.malformed_chunk = True
    with gateway_factory() as gw:
        r = httpx.post(f"{gw}/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 200
    assert log_lines(log_path)[0]["bad_sse_lines"] == 1


def test_invalid_json_body_is_a_400(gateway_factory) -> None:  # type: ignore[no-untyped-def]
    with gateway_factory() as gw:
        r = httpx.post(f"{gw}/v1/chat/completions", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_describe_handles_multimodal_content_and_preview_flag() -> None:
    body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url"}]}],
        "tools": [{}],
    }
    assert describe(body, preview=True)["last_user_preview"] == "hello "
    rec = describe(body, preview=False)
    assert "last_user_preview" not in rec and rec["has_tools"] is True and rec["n_messages"] == 1


def test_timings_fields() -> None:
    t = {
        "prompt_n": 10,
        "cache_n": 90,
        "prompt_ms": 1234.0,
        "prompt_per_second": 8.1,
        "predicted_n": 5,
        "predicted_per_second": 16.66,
    }
    assert timings_fields(t, None) == {
        "prompt_n": 10,
        "cached_n": 90,
        "prefill_s": 1.23,
        "prefill_tps": 8.1,
        "gen_n": 5,
        "gen_tps": 16.7,
    }
    assert timings_fields(None, {"completion_tokens": 3})["gen_n"] == 3


def test_sse_stats_split_chunks_and_tool_calls() -> None:
    s = SSEStats()
    ev = {"choices": [{"delta": {"tool_calls": [{"id": "a"}, {"index": 0}]}, "finish_reason": None}]}
    raw = f"data: {json.dumps(ev)}\n\ndata: " + json.dumps({"choices": [{"finish_reason": "tool_calls"}]}) + "\n\n"
    s.feed(raw[:17].encode())  # a chunk boundary in the middle of a JSON line
    s.feed(raw[17:].encode())
    assert s.tool_calls == 1 and s.finish == "tool_calls" and s.bad_lines == 0


def test_health_reports_upstream_loading(gateway_factory, mock_state: MockState) -> None:  # type: ignore[no-untyped-def]
    import time

    mock_state.faults.loading_until = time.monotonic() + 60
    with gateway_factory(api_keys=frozenset({"k"})) as gw:
        r = httpx.get(f"{gw}/health")
    assert r.status_code == 503 and r.json()["error"]["message"] == "Loading model"


def test_log_preview_off_keeps_prompts_out_of_the_log(gateway_factory, log_path: Path) -> None:  # type: ignore[no-untyped-def]
    secret = "my-private-prompt"
    with gateway_factory(log_preview=False) as gw:
        httpx.post(f"{gw}/v1/chat/completions", json={"messages": [{"role": "user", "content": secret}]})
    assert secret not in log_path.read_text(encoding="utf-8")


def test_passthrough_forwards_post_bodies(gateway_factory, mock_state: MockState) -> None:  # type: ignore[no-untyped-def]
    with gateway_factory() as gw:
        r = httpx.post(f"{gw}/mock/faults", json={"extra_latency_s": 0.5})
    assert r.status_code == 200 and mock_state.faults.extra_latency_s == 0.5
