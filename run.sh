#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Activate the documented Python environment before starting this script.
export SQ_SPACEFLOW_PYTHON="${SQ_SPACEFLOW_PYTHON:-$(command -v python)}"
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

cd "$REPO_ROOT/sq_ui/app"
npm run dev -- --host "${SQ_EDITOR_HOST:-127.0.0.1}"
