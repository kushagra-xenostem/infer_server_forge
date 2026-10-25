import pytest
from pathlib import Path
from llmserve.costmodel import CostModel, _lstsq, load_rows

CLINE_LOG = Path(__file__).parents[1] / "benchmarks/results/gateway/cline_session_metrics.jsonl"


def test_lstsq_recovers_exact_coefficients() -> None:
    xs = [[1.0, float(a), float(b)] for a in range(5) for b in range(0, 50000, 10000)]
    ys = [2.0 + 3.0 * x[1] + 1e-5 * x[2] for x in xs]
    a, b, c = _lstsq(xs, ys)
    assert a == pytest.approx(2.0)
    assert b == pytest.approx(3.0)
    assert c == pytest.approx(1e-5)


def test_lstsq_rejects_collinear_and_too_few_rows() -> None:
    with pytest.raises(ValueError, match="singular"):
        _lstsq([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="at least"):
        _lstsq([[1.0, 2.0, 3.0]], [1.0])


def test_fit_recovers_a_synthetic_model() -> None:
    true = CostModel(
        overhead_s=0.5, prefill_tps=400, cached_s_per_token=2e-5, decode_s_per_token=0.05, decode_s_per_ctx_token=2e-7
    )
    rows = []
    for new in (10, 200, 3000, 20000):
        for cached in (0, 5000, 40000):
            rows.append(
                {
                    "prompt_n": new,
                    "cached_n": cached,
                    "prefill_s": true.prefill_s(new, cached),
                    "gen_tps": 1 / true.token_s(new + cached),
                }
            )
    fit = CostModel.fit(rows)
    assert fit.overhead_s == pytest.approx(0.5)
    assert fit.prefill_tps == pytest.approx(400)
    assert fit.decode_s_per_token == pytest.approx(0.05)
    assert fit.decode_s_per_ctx_token == pytest.approx(2e-7)


def test_defaults_are_the_fit_of_the_real_cline_log() -> None:
    fit = CostModel.fit(load_rows(CLINE_LOG))
    d = CostModel()
    assert fit.overhead_s == pytest.approx(d.overhead_s, rel=0.01)
    assert fit.prefill_tps == pytest.approx(d.prefill_tps, rel=0.01)
    assert fit.decode_s_per_token == pytest.approx(d.decode_s_per_token, rel=0.01)


def test_decode_s_is_the_sum_of_token_s() -> None:
    m = CostModel()
    assert m.decode_s(1000, 50) == pytest.approx(sum(m.token_s(1000 + i) for i in range(50)))
    # the measured shape: ~16.7 tok/s on an empty context, slower at 60k
    assert 1 / m.token_s(0) == pytest.approx(16.7, abs=0.1)
    assert 1 / m.token_s(60000) < 1 / m.token_s(0)
