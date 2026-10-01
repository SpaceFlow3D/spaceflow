#!/usr/bin/env bash
# Submit from the repository root with a valid GPU account/partition.
#SBATCH --job-name=spaceflow-release
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=spaceflow-release-%j.log
set -euo pipefail
REPO_ROOT="${SPACEFLOW_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$REPO_ROOT"
VENV="${SPACEFLOW_VENV:-$REPO_ROOT/.venv}"
source "$VENV/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/spaceflow_runtime/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/spaceflow_runtime/torch}"
export U2NET_HOME="${U2NET_HOME:-$REPO_ROOT/spaceflow_runtime/u2net}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/spaceflow_runtime/xdg_cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_ROOT/spaceflow_runtime/torch-extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
nvidia-smi
if [ "${SPACEFLOW_BUILD_EXTENSIONS:-0}" = 1 ]; then
    SPACEFLOW_SETUP_STAGE=extensions SPACEFLOW_VENV="$VENV" bash setup.sh
fi
if ! python -c 'import json, os, pathlib, sys; path=pathlib.Path(os.environ["HF_HOME"])/"spaceflow-models.json"; record=json.loads(path.read_text()) if path.is_file() else {}; pins=json.loads(pathlib.Path("requirements/model-revisions.json").read_text()); sys.exit(0 if record.get("image", {}).get("dinov2")==pins["dinov2"] else 1)'; then
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python tools/cache_models.py --cache-dir "$HF_HOME" --include-image
fi
OUTPUT="${SPACEFLOW_VERIFY_OUTPUT:-$REPO_ROOT/runs/release-$SLURM_JOB_ID}"
IMAGE="${SPACEFLOW_VERIFY_IMAGE:-$REPO_ROOT/docs/media/sailboat_spin_poster.png}"
python tools/verify_release.py --output-dir "$OUTPUT" --image-path "$IMAGE"
