import json
import time
import httpx
import pytest
from llmserve.mock_upstream import MockState, n_tokens, prompt_text

MSG = [{"role": "user", "content": "tok " * 400}]


def chat(url: str, **body: object) -> httpx.Response:
    return httpx.post(f"{url}/v1/chat/completions", json={"messages": MSG, "max_tokens": 8, **body}, timeout=10)


def sse_events(text: str) -> list[dict[str, object]]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


def test_second_request_with_same_prefix_hits_the_cache(mock_url: str) -> None:
    first = chat(mock_url).json()["timings"]
    assert first["cache_n"] == 0
    second = chat(mock_url).json()["timings"]
    assert second["cache_n"] == first["prompt_n"]  # the whole prompt was resident
    assert second["prompt_n"] == 0
    assert second["prompt_ms"] < first["prompt_ms"]


def test_reported_timings_follow_the_cost_model(mock_url: str, mock_state: MockState) -> None:
    t = chat(mock_url).json()["timings"]
    n = t["prompt_n"]
    assert t["prompt_ms"] == pytest.approx(mock_state.cost.prefill_s(n, 0) * 1000, abs=0.1)
    assert t["predicted_ms"] == pytest.approx(mock_state.cost.decode_s(n, 8) * 1000, abs=0.1)


def test_stream_has_one_token_per_chunk_and_final_timings(mock_url: str) -> None:
    r = chat(mock_url, stream=True, max_tokens=5)
    assert r.status_code == 200
    assert r.text.endswith("data: [DONE]\n\n")
    evs = sse_events(r.text)
    assert len(evs) == 6
    assert evs[-1]["choices"][0]["finish_reason"] == "length"  # type: ignore[index]
    assert evs[-1]["timings"]["predicted_n"] == 5  # type: ignore[index]


def test_context_overflow_is_a_400(mock_url: str, mock_state: MockState) -> None:
    mock_state.ctx_size = 100
    r = chat(mock_url)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "exceed_context_size_error"


def test_faults_set_at_runtime(mock_url: str) -> None:
    assert httpx.post(f"{mock_url}/mock/faults", json={"error_status": 500}).json()["error_status"] == 500
    assert chat(mock_url).status_code == 500
    assert "unknown" in httpx.post(f"{mock_url}/mock/faults", json={"nope": 1}).json()["error"]


def test_loading_returns_503_until_ready(mock_url: str, mock_state: MockState) -> None:
    mock_state.faults.loading_until = time.monotonic() + 60
    assert httpx.get(f"{mock_url}/health").status_code == 503
    assert chat(mock_url).status_code == 503
    mock_state.faults.loading_until = 0
    assert httpx.get(f"{mock_url}/health").status_code == 200


def test_malformed_chunk_fault(mock_url: str, mock_state: MockState) -> None:
    mock_state.faults.malformed_chunk = True
    r = chat(mock_url, stream=True, max_tokens=4)
    bad = [line for line in r.text.splitlines() if line.startswith("data: {") and not line.endswith("}")]
    assert len(bad) == 1


def test_drop_mid_stream_fault(mock_url: str, mock_state: MockState) -> None:
    mock_state.faults.drop_after_tokens = 2
    with pytest.raises(httpx.RemoteProtocolError):
        chat(mock_url, stream=True, max_tokens=10)


def test_short_token_sleeps_still_add_up(mock_state: MockState, mock_url: str) -> None:
    # Regression: per-token sleeps shorter than the OS timer tick (~15.6 ms on Windows) used to take no time,
    # so a whole stream arrived at once. 40 tokens at time_scale 0.05 are ~3 ms sleeps each, 0.12 s in total.
    mock_state.time_scale = 0.05
    t0 = time.perf_counter()
    arrivals = []
    with httpx.stream(
        "POST", f"{mock_url}/v1/chat/completions", json={"messages": MSG, "max_tokens": 40, "stream": True}, timeout=30
    ) as r:
        for line in r.iter_lines():
            if '"content"' in line:
                arrivals.append(time.perf_counter() - t0)
    expected = mock_state.cost.decode_s(n_tokens(prompt_text({"messages": MSG})), 39) * 0.05
    # the bug made the span ~0; a busy test machine can only make it longer, so check the lower bound
    assert arrivals[-1] - arrivals[0] >= 0.75 * expected
