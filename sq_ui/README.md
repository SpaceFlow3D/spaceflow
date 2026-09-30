# SpaceFlow UI

Web UI for editing superquadrics and launching SpaceFlow runs from this checkout.

Default service port:

- SpaceFlow runs/assets: `http://127.0.0.1:11438`

## SpaceFlow Service

```bash
source .venv/bin/activate
export SQ_SPACEFLOW_PYTHON="$(pwd)/.venv/bin/python"
export SQ_SPACEFLOW_STORAGE_ROOT="$(pwd)/spaceflow_runtime"
"$SQ_SPACEFLOW_PYTHON" sq_ui/scripts/spaceflow_service.py
```

Health check:

```bash
curl -s http://127.0.0.1:11438/spaceflow/health | head
```

The editor and asset saving work without a GPU. Generation needs the environment,
checkpoints, and Blender described in the [project README](../README.md).
Set `SQ_SPACEFLOW_FORCE_LOCAL=1` when already running on an allocated GPU node.
When launching from a login node, set `SQ_SPACEFLOW_SLURM_ACCOUNT` and
`SQ_SPACEFLOW_SLURM_PARTITION` for your allocation.

## Public Controlled Demo

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

## Frontend

```bash
cd sq_ui/app
npm ci --include=optional
npm run dev -- --host 127.0.0.1
```

Open the printed Vite URL, usually `http://<host>:5173`.
The development server proxies `/spaceflow` requests to the backend on port
11438. Override `VITE_DEV_PROXY_SPACEFLOW` when the backend uses another port.
For remote use, forward the editor port over SSH.

## Open NPZ Files Directly

With the Vite dev server running:

```text
http://<host>:5173/?npz=/absolute/path/to/file.npz
```

By default the dev server allows `.npz` files from the repo, `spaceflow_runtime/`,
the shared sibling `../spaceflow_runtime/`, and the course dataset folder at
`../spaceflow/datasets/` when those paths exist. Override with
`SQ_UI_NPZ_ROOTS=/path/a:/path/b`.
