#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Activate the documented Python environment before starting this script.
if [ -z "${SQ_SPACEFLOW_PYTHON:-}" ]; then
    SQ_SPACEFLOW_PYTHON="$(command -v python || command -v python3 || true)"
fi
export SQ_SPACEFLOW_PYTHON
if [ -z "$SQ_SPACEFLOW_PYTHON" ] || ! "$SQ_SPACEFLOW_PYTHON" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)'; then
    echo 'Activate a Python 3.10–3.12 editor environment, or the Python 3.10 generation environment, before starting the UI.' >&2
    exit 1
fi
command -v node >/dev/null && command -v npm >/dev/null || {
    echo 'Install Node.js 22.12 or later (Node 24 is recommended) and npm.' >&2
    exit 1
}
if [ ! -d "$REPO_ROOT/sq_ui/app/node_modules" ]; then
    echo 'Install the editor first: cd sq_ui/app && npm ci --include=optional' >&2
    exit 1
fi
if ! "$SQ_SPACEFLOW_PYTHON" tools/doctor.py --editor; then
    exit 1
fi
export SQ_SPACEFLOW_STORAGE_ROOT="${SQ_SPACEFLOW_STORAGE_ROOT:-$REPO_ROOT/spaceflow_runtime}"
export SQ_SPACEFLOW_ASSET_ROOT="${SQ_SPACEFLOW_ASSET_ROOT:-$SQ_SPACEFLOW_STORAGE_ROOT/sq_ui_assets}"
export SQ_SPACEFLOW_RUN_ROOT="${SQ_SPACEFLOW_RUN_ROOT:-$SQ_SPACEFLOW_STORAGE_ROOT/sq_ui_runs}"
export SQ_SPACEFLOW_HOST="${SQ_SPACEFLOW_HOST:-127.0.0.1}"
export SQ_SPACEFLOW_PORT="${SQ_SPACEFLOW_PORT:-11438}"
export VITE_DEV_PROXY_SPACEFLOW="${VITE_DEV_PROXY_SPACEFLOW:-http://127.0.0.1:$SQ_SPACEFLOW_PORT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$SQ_SPACEFLOW_PYTHON" sq_ui/scripts/spaceflow_service.py &
SPACEFLOW_PID=$!

cleanup() {
    kill "$SPACEFLOW_PID" 2>/dev/null || true
    wait "$SPACEFLOW_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Catch an occupied port or failed service before opening an unusable editor.
service_ready=0
for attempt in {1..40}; do
    if ! kill -0 "$SPACEFLOW_PID" 2>/dev/null; then
        echo 'The editor service failed to start. Check the message above and SQ_SPACEFLOW_PORT.' >&2
        exit 1
    fi
    if "$SQ_SPACEFLOW_PYTHON" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=0.2).close()' \
        "http://127.0.0.1:$SQ_SPACEFLOW_PORT/spaceflow/health" 2>/dev/null; then
        service_ready=1
        break
    fi
    sleep 0.1
done
if [ "$service_ready" = 0 ]; then
    echo 'The editor service did not become ready; inspect its startup error above.' >&2
    exit 1
fi

cd "$REPO_ROOT/sq_ui/app"
npm run dev -- --host "${SQ_EDITOR_HOST:-127.0.0.1}"
