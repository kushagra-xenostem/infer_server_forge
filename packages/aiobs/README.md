# aiobs

A small observability kit shared by the portfolio projects, so each one does not rewrite its telemetry.
Standard library + `psutil` only.

| Module | What it gives you |
|---|---|
| `aiobs.events` | `Event` (one JSON line: time, kind, name, duration, attributes) and `EventLog` (append-only JSONL) |
| `aiobs.timing` | `span(log, name, **attrs)`: a context manager that times a block and logs it, including on error |
| `aiobs.tokens` | `TokenCounter`: sums `usage` (OpenAI) and `timings` (llama.cpp) across calls; cached-token share |
| `aiobs.resources` | `ResourceSampler`: a background thread sampling CPU % and RSS; peak and mean on exit |
| `aiobs.report` | `percentiles`, `summarize`, and `write_report` (the same numbers as JSON + a Markdown table) |

Install from another project (no PyPI release):

```bash
uv add "aiobs @ git+https://github.com/<owner>/<repo>#subdirectory=packages/aiobs"
# or, side by side on disk:
uv add --editable ../<this repo>/packages/aiobs
```

```python
from aiobs import EventLog, ResourceSampler, TokenCounter, span, write_report

log = EventLog("runs/events.jsonl")
tokens = TokenCounter()
with ResourceSampler(interval_s=0.5) as res, span(log, "extract", doc="a.pdf"):
    reply = client.chat(...)
    tokens.add(usage=reply["usage"], timings=reply.get("timings"))
write_report("runs/report", {"tokens": tokens.as_dict(), "resources": res.summary()})
```
