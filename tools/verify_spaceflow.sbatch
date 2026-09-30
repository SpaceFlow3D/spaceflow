#!/usr/bin/env bash
# Submit from the repository root and supply your account/partition to sbatch.
#SBATCH --job-name=spaceflow-verify
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=spaceflow-verify-%j.log
set -euo pipefail
REPO_ROOT="${SPACEFLOW_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$REPO_ROOT"
VENV="${SPACEFLOW_VENV:-$REPO_ROOT/.venv}"
source "$VENV/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MAX_JOBS="$OMP_NUM_THREADS"
export HF_HOME="${HF_HOME:-$REPO_ROOT/spaceflow_runtime/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/spaceflow_runtime/xdg_cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_ROOT/spaceflow_runtime/torch-extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
nvidia-smi
python -c 'import torch; print("PyTorch",torch.__version__,"GPU",torch.cuda.get_device_name(),"capability",torch.cuda.get_device_capability()); assert torch.cuda.is_available()'

if [ "${SPACEFLOW_BUILD_EXTENSIONS:-0}" = 1 ]; then
  export SPACEFLOW_SETUP_STAGE=extensions SPACEFLOW_VENV="$VENV"
  bash setup.sh
fi
test -f "$HF_HOME/spaceflow-models.json" || python tools/cache_models.py --cache-dir "$HF_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUTPUT="${SPACEFLOW_VERIFY_OUTPUT:-$REPO_ROOT/runs/spaceflow-$SLURM_JOB_ID}"
EXAMPLE="${SPACEFLOW_VERIFY_EXAMPLE:-$REPO_ROOT/examples/blue_teacup_full_experiment}"
VARIANT="${SPACEFLOW_VERIFY_VARIANT:-01_local_tau3_tau10_polyak0p18}"
python tools/replay_example.py --example-dir "$EXAMPLE" --output-dir "$OUTPUT" --only "$VARIANT"
python tools/check_replay_outputs.py "$OUTPUT"
printf 'Verified SpaceFlow output: %s\n' "$OUTPUT"
