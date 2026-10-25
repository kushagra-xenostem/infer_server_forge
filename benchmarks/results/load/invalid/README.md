# Invalid simulator runs (kept for the record, not used anywhere)

These sim runs (2026-10-18, tags `simval-small`, `e-load-noadm`) were made before failure #19 was fixed: the
mock's per-token sleeps (~15 ms at time_scale 0.25) were shorter than the Windows timer tick, and asyncio
fires such timers at once, so decode took ~0 s and every stream arrived in one burst (e2e ≈ TTFT).
Fixed by pacing sleeps on an absolute timeline (`Pacer` in `src/llmserve/mock_upstream.py`); regression test
`test_short_token_sleeps_still_add_up`.
