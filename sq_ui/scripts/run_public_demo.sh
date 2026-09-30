#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # Match the course runtime used by run.sh when it is available.
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${SQ_PUBLIC_CONDA_ENV:-/work/courses/3dv/team3/guideflow3d/envs/guideflow3d}"
fi

if [ -f ".env.public-demo" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env.public-demo"
    set +a
fi

if [ -z "${SQ_PUBLIC_PASSWORD:-}" ]; then
    echo "Set SQ_PUBLIC_PASSWORD in .env.public-demo before starting the public demo." >&2
    echo "Example file:" >&2
    echo "  SQ_PUBLIC_USER=spaceflow" >&2
    echo "  SQ_PUBLIC_PASSWORD=replace-with-shared-password" >&2
    exit 1
fi

export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)")")}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-6.1;7.5;8.0;8.6;9.0;12.0}"
export BLENDER_HOME="${BLENDER_HOME:-/work/courses/3dv/team3/spaceflow-minimal/blender-3.0.1-linux-x64/blender}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export SQ_SPACEFLOW_PYTHON="${SQ_SPACEFLOW_PYTHON:-/work/courses/3dv/team3/guideflow3d/envs/guideflow3d/bin/python}"
export SQ_SPACEFLOW_STORAGE_ROOT="${SQ_SPACEFLOW_STORAGE_ROOT:-/work/courses/3dv/team3/spaceflow_runtime}"
export SQ_SPACEFLOW_ASSET_ROOT="${SQ_SPACEFLOW_ASSET_ROOT:-$SQ_SPACEFLOW_STORAGE_ROOT/sq_ui_assets}"
export SQ_SPACEFLOW_RUN_ROOT="${SQ_SPACEFLOW_RUN_ROOT:-$SQ_SPACEFLOW_STORAGE_ROOT/sq_ui_runs}"
export SQ_SPACEFLOW_HOST="${SQ_SPACEFLOW_HOST:-127.0.0.1}"
export SQ_SPACEFLOW_PORT="${SQ_SPACEFLOW_PORT:-11480}"
export SQ_SPACEFLOW_PUBLIC_DEMO=1
export SQ_SPACEFLOW_MAX_ACTIVE_RUNS="${SQ_SPACEFLOW_MAX_ACTIVE_RUNS:-1}"
export SQ_SPACEFLOW_RETENTION_HOURS="${SQ_SPACEFLOW_RETENTION_HOURS:-48}"
export SQ_SPACEFLOW_MAX_STORAGE_GB="${SQ_SPACEFLOW_MAX_STORAGE_GB:-40}"

export SQ_PUBLIC_HOST="${SQ_PUBLIC_HOST:-127.0.0.1}"
export SQ_PUBLIC_PORT="${SQ_PUBLIC_PORT:-11481}"
export SQ_PUBLIC_USER="${SQ_PUBLIC_USER:-spaceflow}"
export SQ_PUBLIC_BACKEND="${SQ_PUBLIC_BACKEND:-http://127.0.0.1:$SQ_SPACEFLOW_PORT}"
export SQ_PUBLIC_STATIC_ROOT="${SQ_PUBLIC_STATIC_ROOT:-$PWD/sq_ui/app/dist}"
export SQ_PUBLIC_MAX_UPLOAD_MB="${SQ_PUBLIC_MAX_UPLOAD_MB:-64}"
export SQ_PUBLIC_SKIP_QUICK_TUNNEL="${SQ_PUBLIC_SKIP_QUICK_TUNNEL:-0}"

port_in_use() {
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${1}$"
}

if port_in_use "$SQ_SPACEFLOW_PORT"; then
    echo "Port $SQ_SPACEFLOW_PORT is already in use. Stop the previous SpaceFlow demo/backend before starting a new one." >&2
    exit 1
fi

if port_in_use "$SQ_PUBLIC_PORT"; then
    echo "Port $SQ_PUBLIC_PORT is already in use. Stop the previous public gateway before starting a new one." >&2
    exit 1
fi

echo "[sq-public-demo] Building public UI..."
(cd sq_ui/app && VITE_PUBLIC_DEMO=1 npm run build)

echo "[sq-public-demo] Starting private SpaceFlow backend on $SQ_SPACEFLOW_HOST:$SQ_SPACEFLOW_PORT..."
"$SQ_SPACEFLOW_PYTHON" sq_ui/scripts/spaceflow_service.py &
SPACEFLOW_PID=$!

echo "[sq-public-demo] Starting authenticated gateway on $SQ_PUBLIC_HOST:$SQ_PUBLIC_PORT..."
"$SQ_SPACEFLOW_PYTHON" sq_ui/scripts/public_demo_gateway.py &
GATEWAY_PID=$!

cleanup() {
    kill "$GATEWAY_PID" "$SPACEFLOW_PID" 2>/dev/null || true
    wait "$GATEWAY_PID" "$SPACEFLOW_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 1

if ! kill -0 "$SPACEFLOW_PID" 2>/dev/null; then
    echo "[sq-public-demo] Private SpaceFlow backend failed to start." >&2
    exit 1
fi

if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
    echo "[sq-public-demo] Public gateway failed to start." >&2
    exit 1
fi

case "$SQ_PUBLIC_SKIP_QUICK_TUNNEL" in
    1|true|TRUE|yes|YES|on|ON)
        echo "[sq-public-demo] Skipping Cloudflare quick tunnel."
        echo "[sq-public-demo] Local authenticated gateway is running at http://127.0.0.1:$SQ_PUBLIC_PORT"
        echo "[sq-public-demo] Point a named Cloudflare Tunnel at http://127.0.0.1:$SQ_PUBLIC_PORT."
        wait
        ;;
    *)
        if command -v cloudflared >/dev/null 2>&1; then
            echo "[sq-public-demo] Starting Cloudflare quick tunnel..."
            cloudflared tunnel --url "http://127.0.0.1:$SQ_PUBLIC_PORT"
        else
            echo "[sq-public-demo] cloudflared is not installed or not on PATH." >&2
            echo "[sq-public-demo] In another terminal, install/use cloudflared and run:" >&2
            echo "  cloudflared tunnel --url http://127.0.0.1:$SQ_PUBLIC_PORT" >&2
            echo "[sq-public-demo] Local authenticated gateway is running at http://127.0.0.1:$SQ_PUBLIC_PORT" >&2
            wait
        fi
        ;;
esac
