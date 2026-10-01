# Controlled demo helper

Run these commands from the repository root with a working GPU environment.
This optional helper is separate from normal local usage and Hugging Face hosting.

## Public controlled demo

The public demo path keeps the SpaceFlow service private on localhost and exposes only an authenticated gateway.

Create a local secret file that is already ignored by `.gitignore`:

```bash
cat > .env.public-demo <<'EOF'
SQ_PUBLIC_USER=spaceflow
SQ_PUBLIC_PASSWORD=replace-with-a-shared-password
EOF
chmod 600 .env.public-demo
```

Start the demo helper:

```bash
bash sq_ui/scripts/run_public_demo.sh
```

The script builds the UI with `VITE_PUBLIC_DEMO=1`, starts `spaceflow_service.py` on `127.0.0.1:11480`, starts `public_demo_gateway.py` on `127.0.0.1:11481`, and runs `cloudflared tunnel --url http://127.0.0.1:11481` when `cloudflared` is on `PATH`.

Public mode defaults:

- `SQ_SPACEFLOW_MAX_ACTIVE_RUNS=1`
- `SQ_SPACEFLOW_RETENTION_HOURS=48`
- `SQ_SPACEFLOW_MAX_STORAGE_GB=40`
- `SQ_PUBLIC_MAX_UPLOAD_MB=64`

Use the printed `https://...trycloudflare.com` URL plus the shared user/password for selected users. Quick tunnel URLs change when the tunnel restarts.

### Long-Lived Cloudflare Tunnel

For a stable public URL, use a named Cloudflare Tunnel instead of the generated `trycloudflare.com` quick tunnel. This requires a Cloudflare account and a domain using Cloudflare DNS.

Authenticate and create the tunnel once:

```bash
cloudflared tunnel login
cloudflared tunnel create spaceflow-demo
cloudflared tunnel route dns spaceflow-demo spaceflow.example.com
cloudflared tunnel list
```

Create `~/.cloudflared/spaceflow-demo.yml`, replacing the UUID and hostname:

```yaml
tunnel: <TUNNEL-UUID>
credentials-file: /absolute/path/to/your/.cloudflared/<TUNNEL-UUID>.json

ingress:
  - hostname: spaceflow.example.com
    service: http://127.0.0.1:11481
  - service: http_status:404
```

Run the local demo gateway without starting a quick tunnel:

```bash
tmux new -s spaceflow-public
SQ_PUBLIC_SKIP_QUICK_TUNNEL=1 bash sq_ui/scripts/run_public_demo.sh
```

In another tmux window, run the named tunnel:

```bash
cloudflared tunnel --config ~/.cloudflared/spaceflow-demo.yml run spaceflow-demo
```

