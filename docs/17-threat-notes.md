# 17 · Threat notes

What can go wrong when other machines talk to this server, and what stops it. Scope: the gateway and the
model server. The model's own behaviour (prompt injection into Cline, unsafe generated code) is covered by
Cline's approval prompts and the sandbox in `benchmarks/Dockerfile.sandbox`.

| Threat | Entry point | Control | Status |
|---|---|---|---|
| Someone on the internet uses the GPU | the HTTP port | llama-server listens on 127.0.0.1 only; the gateway binds the Tailscale interface only (07) | ✅ deployed |
| Someone inside the tailnet uses it without permission | the gateway | Bearer keys (`GW_API_KEYS`); `/health` is the only open path | ✅ in `llmserve` (tested); the deployed gateway already enforces keys (measured: 401 in 0.12 s) |
| One client starves everyone (huge prompts, floods) | `/v1/chat/completions` | bounded queue → 429 + Retry-After; per-request deadline; oversized prompts rejected by llama-server before prefill (measured: 80k tokens → 400 in 0.5 s) | ✅ (`llmserve`), the deployed gateway rate-limits too |
| A client that hangs up keeps the GPU busy | streaming responses | disconnect → slot released and upstream closed, which stops generation | ✅ tested (≤ 2 s; 5 ms in the fault run); live server confirmed to stop generating |
| Prompts leak through logs | `logs/gateway.jsonl` | `GW_LOG_PREVIEW=0` drops the 160-character preview; published logs have prompts removed | ✅ |
| Secrets in the repo | `.env` | `.env` and `.env.*` git-ignored; only `.env.example` is tracked | ✅ |
| Identifiers in published data | `benchmarks/results/*.jsonl` | addresses and account names are redacted in docs; **older llama-bench JSON still contains a home-directory path with the account name** | ⚠ owner decision (see the cleanup list) |
| A hung server stays hung | llama-server | gateway deadline answers clients; systemd + watchdog restart the server | ⚠ written, not deployed (16) |
| GPU driver deadlock | amdgpu at ~100% VRAM | VRAM guard refuses to serve above 95% after load | ✅ (09 #9); a reboot is the only recovery |

Not in scope: TLS inside the tailnet (WireGuard already encrypts it), multi-tenant isolation between key
holders (all keys are equal), and per-key quotas (a later gateway feature if the server is shared widely).
