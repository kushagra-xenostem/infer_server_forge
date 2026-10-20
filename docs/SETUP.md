# Local Coding Agent Setup: Qwen3.6-35B-A3B + Cline

## Server (Linux, RX 6700)

Model: `~/models/gguf/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`

```bash
cd /data/projects/llm-serving/scripts
./prod_server.sh start         # thinking off (default)
./prod_server.sh start think   # thinking on (4096-token budget)
./prod_server.sh status
./prod_server.sh stop
```

Ready when you see `READY ... sanity=391` and `gateway ... health 200` (~1–2 min).

- llama-server: `127.0.0.1:8080` (private)
- Gateway for Cline: `<TAILSCALE_IP>:8081` (Tailscale only, logs to `logs/gateway.jsonl`)

## Client (your PC)

1. Connect Tailscale, then test:
   ```
   curl.exe http://<TAILSCALE_IP>:8081/v1/models
   ```
2. Cline → ⚙️ → API Configuration:

   | Field | Value |
   |---|---|
   | API Provider | OpenAI Compatible |
   | Base URL | `http://<TAILSCALE_IP>:8081/v1` |
   | API Key | `local` |
   | Model ID | `qwen3.6-35b-a3b` |
   | Context Window | `65536` |
   | Max Output Tokens | `8192` |
   | Supports Images | **on** (vision runs on the CPU: ~20 s for a small image, ~70 s for a full-HD screenshot) |
   | Temperature | default |

3. Enable **auto-compact** in Cline settings.
4. Work on a branch: `git checkout -b cline-work`.

## Rules

- Context max is **64k**. Don't raise it: 128k hung the GPU driver.
- First prompt in a session: ~1–2 min. Later turns: ~5–10 s.
- Never use a 4-bit K cache (`-ctk q4_0`), because it breaks the model.
- If anything is stuck, run `./prod_server.sh stop`, then `./prod_server.sh start`.
- If `stop` hangs or the GPU won't free memory, run `sudo reboot`.
