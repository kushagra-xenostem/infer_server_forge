# 07 · Private remote access with Tailscale

The server sits at home; Cline runs on a laptop/PC. They connect over **[Tailscale](https://tailscale.com)**, a WireGuard-based mesh VPN. Nothing is exposed to the public internet and no ports are forwarded on the router.

## Setup
1. **Server (Ubuntu):** install Tailscale (`curl -fsSL https://tailscale.com/install.sh | sh`), then `sudo tailscale up` and sign in. The server gets a stable `100.x.y.z` address (version 1.102.3 here).
2. **Client (Windows):** install Tailscale from tailscale.com/download and sign in with the **same account**. It runs in the tray and starts with Windows.
3. **Check:** in the Tailscale admin console both devices show as connected. Here the PC connects **directly** (peer-to-peer), not through a relay.
4. **Test from the client:**
   ```powershell
   curl.exe http://<TAILSCALE_IP>:8081/v1/models
   # → {"data":[{"id":"qwen3.6-35b-a3b",...}]}
   ```

## Security model
| Layer | Protection |
|---|---|
| Network | Only devices signed into the owner's tailnet can reach the server; traffic is WireGuard-encrypted |
| Binding | The gateway binds to the Tailscale interface IP only (not `0.0.0.0`), so it isn't reachable from the home LAN |
| llama-server | Bound to `127.0.0.1`, reachable only through the gateway |
| Firewall | `ufw` disabled; exposure is controlled by binding + Tailscale |

The Tailscale address is stable across reboots. It only changes if the device is removed and re-added to the tailnet. MagicDNS can replace the IP with the hostname.

> Addresses, account names and hostnames are redacted in this repo as `<TAILSCALE_IP>`.
