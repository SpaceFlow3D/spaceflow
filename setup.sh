#!/usr/bin/env bash
# Linux/CUDA installation. Build CUDA extensions on an allocated compute node.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
STAGE="${SPACEFLOW_SETUP_STAGE:-all}"
VENV="${SPACEFLOW_VENV:-$REPO_ROOT/.venv}"
PYTHON="${SPACEFLOW_SETUP_PYTHON:-python3}"
export MAX_JOBS="${MAX_JOBS:-4}"

case "$STAGE" in deps|extensions|all) ;; *) echo 'SPACEFLOW_SETUP_STAGE must be deps, extensions, or all.' >&2; exit 2;; esac
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for SpaceFlow."'
  "$PYTHON" -m venv "$VENV"
fi
PIP=("$VENV/bin/python" -m pip)
"$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for SpaceFlow."'

if [ "$STAGE" != extensions ]; then
  "${PIP[@]}" install 'pip==25.3' 'setuptools==80.9.0' 'wheel==0.45.1'
  "${PIP[@]}" install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  "${PIP[@]}" install -c requirements/constraints.txt -c requirements/runtime-resolved-constraints.txt -r requirements/runtime.txt
  "${PIP[@]}" install -c requirements/constraints.txt -c requirements/runtime-resolved-constraints.txt git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
  "${PIP[@]}" install -c requirements/constraints.txt -c requirements/runtime-resolved-constraints.txt kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
  "${PIP[@]}" install -c requirements/constraints.txt -c requirements/runtime-resolved-constraints.txt torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
fi

if [ "$STAGE" != deps ]; then
  command -v nvcc >/dev/null || { echo 'Load the CUDA 12.8 toolkit before building extensions.' >&2; exit 1; }
  "$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "Build and verify CUDA extensions on an allocated GPU node."'
  "${PIP[@]}" install -c requirements/constraints.txt -c requirements/runtime-resolved-constraints.txt flash-attn==2.8.3 --no-build-isolation
  "${PIP[@]}" install -c requirements/constraints.txt -c requirements/runtime-resolved-constraints.txt git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae --no-build-isolation
  EXT_ROOT="${SPACEFLOW_EXTENSION_BUILD_ROOT:-$REPO_ROOT/.extension-build}"
  MIP_COMMIT=dda02ab5ecf45d6edb8c540d9bb65c7e451345a9
  MIP_ROOT="$EXT_ROOT/mip-splatting-$MIP_COMMIT"
  if [ ! -d "$MIP_ROOT" ]; then
    mkdir -p "$EXT_ROOT"
    git clone --no-checkout https://github.com/autonomousvision/mip-splatting.git "$MIP_ROOT"
    git -C "$MIP_ROOT" checkout --detach "$MIP_COMMIT"
    git -C "$MIP_ROOT" submodule update --init --recursive
  fi
  test "$(git -C "$MIP_ROOT" rev-parse HEAD)" = "$MIP_COMMIT"
  "${PIP[@]}" install -c requirements/constraints.txt -c requirements/runtime-resolved-constraints.txt "$MIP_ROOT/submodules/diff-gaussian-rasterization" --no-build-isolation
  "${PIP[@]}" install -c requirements/constraints.txt -c requirements/runtime-resolved-constraints.txt third_party/TRELLIS/extensions/vox2seq --no-build-isolation
fi

"${PIP[@]}" check
"${PIP[@]}" freeze > "$VENV/spaceflow-installed-packages.txt"
echo "Activate the environment with: source $VENV/bin/activate"
echo 'Place the PartField checkpoint at third_party/PartField/models/model_objaverse.ckpt.'
echo 'Set SPACEFLOW_BLENDER_PATH to a working Blender executable.'
