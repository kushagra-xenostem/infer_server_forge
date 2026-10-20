# 08 · Using it from VS Code with Cline

[Cline](https://github.com/cline/cline) is an open-source autonomous coding agent for VS Code. It reads files, runs commands, edits code and iterates. Here it runs on the developer's PC and uses the home server as its model.

## Install
1. VS Code → Extensions → search **"Cline"** (publisher `saoudrizwan`, extension id `saoudrizwan.claude-dev`) → Install. This project used **v4.1.22**.
2. Open the Cline panel from the activity bar.

## Configure (Cline → ⚙️ → API Configuration)

| Field | Value |
|---|---|
| API Provider | **OpenAI Compatible** |
| Base URL | `http://<TAILSCALE_IP>:8081/v1` |
| API Key | any non-empty text, e.g. `local` (not checked; access is controlled by Tailscale) |
| Model ID | `qwen3.6-35b-a3b` |
| Context Window Size | **`65536`** (must match the server's `-c`) |
| Max Output Tokens | `8192` |
| Supports Images | **on**: the server reads images ([14-vision](14-vision.md)); ~20 s for a small image, ~75 s for a full-HD screenshot |
| Supports Prompt Caching | off (llama-server reuses the prompt cache automatically) |
| Temperature | leave default (the gateway applies Qwen's recommended sampling) |

Then enable Cline's **auto-compact / auto-condense** so long conversations are summarized before they reach 64k.

> A mismatch here is the most common mistake: with the default 128k window, Cline won't compact in time and the server rejects requests above 64k.

## Daily workflow
1. On the server, after a reboot: `server/prod_server.sh start` (~1–2 min until `READY … health 200`). Running it again when the server is already up just reports "Already running and healthy".
2. On the PC: Tailscale connected → open your repo → `git checkout -b cline-work`.
3. Give Cline a task. The first message of a task is slower (it reads files into context); later turns take seconds.
4. Review Cline's diffs before committing. Known weak spots: tricky regex and spec edge cases.

## Thinking mode
`server/prod_server.sh start think` enables Qwen's reasoning with a 4,096-token budget (Qwen's precise-coding sampling: temp 0.6, top-p 0.95). It's slower per turn. Use it for hard debugging; the default non-thinking mode is faster for routine edits.
