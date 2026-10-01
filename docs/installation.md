# Installation and environment configuration

Run commands from the repository root unless a different directory is shown.
The [main README](../README.md) contains the editor and generation quickstarts.

## Two separate environments

- `.venv-editor`: Python 3.10–3.12; install `requirements/editor.txt`. Handles editing,
  NPZ downloads, save/history/reopen, and CPU comparison rendering. No PyTorch/CUDA.
- `.venv`: Python 3.10; created by `setup.sh`. Handles GPU generation and can also
  run the editor service. Activate this environment when launching generation.

The service uses its active interpreter. `SQ_SPACEFLOW_PYTHON` explicitly selects
another interpreter if needed; old developer environment directories are not
selected automatically.

## Generation installation in two stages

On a login node or another Linux machine without a GPU, download dependencies:

```bash
SPACEFLOW_SETUP_PYTHON=python3.10 SPACEFLOW_SETUP_STAGE=deps bash setup.sh
```

Then, **inside an allocated GPU job**, with the CUDA 12.8 toolkit loaded:

```bash
SPACEFLOW_SETUP_STAGE=extensions bash setup.sh
source .venv/bin/activate
```

The environment pins PyTorch 2.8.0/cu128, Kaolin 0.18.0, FlashAttention 2.8.3,
nvdiffrast, mip-splatting's Gaussian rasterizer, and TRELLIS vox2seq.
`requirements/native-revisions.json` records native source revisions;
`requirements/runtime-resolved-constraints.txt` records the resolved Python
packages. Both stages finish with `pip check`.

Override the environment directory with `SPACEFLOW_VENV`; both stages must use
the same directory. Native build intermediates go into the ignored
`.extension-build/` directory. Use `MAX_JOBS` to limit compiler parallelism.

## Models and offline text generation

```bash
source .venv/bin/activate
python tools/cache_models.py --cache-dir spaceflow_runtime/huggingface
export HF_HOME="$PWD/spaceflow_runtime/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

The staging tool downloads only the mixed TRELLIS components used by the bundled
text-conditioned runs, the CLIP text conditioner, and the PartField checkpoint.
It uses the revisions in `requirements/model-revisions.json` and validates
PartField's SHA256 before placing it at
`third_party/PartField/models/model_objaverse.ckpt`.

Keep `HF_HOME` consistent across model staging, environment checks, and generation.
The cache marker is `spaceflow-models.json`. Downloaded weights, Blender, CUDA
builds, and generated outputs can consume many GB; keep them on suitable work or
scratch storage rather than a small cluster home directory.

## Additional image-conditioning models

Image appearance conditioning uses the same recorded TRELLIS image revision,
a DINOv2 source commit and checkpoint SHA256, and the U2Net background-removal
checkpoint checked against rembg's recorded checksum. Stage these before the
first image run:

```bash
export HF_HOME="$PWD/spaceflow_runtime/huggingface"
export TORCH_HOME="$PWD/spaceflow_runtime/torch"
export U2NET_HOME="$PWD/spaceflow_runtime/u2net"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python tools/cache_models.py --cache-dir "$HF_HOME" --include-image
python tools/doctor.py --gpu --image
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Keep all three cache variables consistent during staging, checks, and generation.
`--torch-cache-dir` and `--u2net-cache-dir` can select other work-storage locations.
The original `JeffreyXiang/TRELLIS-image-large` name redirects upstream to
`microsoft/TRELLIS-image-large`; staging makes that original name resolve to the
same pinned snapshot offline. Model architectures, conditioning defaults, and
weights are preserved. DINOv2 now requests its full recorded source commit.

Staging loads DINOv2 on the CPU to confirm its source and weights are usable. It
does not run image-conditioned 3D generation. That fresh GPU run remains a release
acceptance check. Input-image paths and bytes must be available on the generation
machine.

## Blender

The original renderer defaults to Blender 3.0.1. The README shows how to install
that Linux x86-64 build into `spaceflow_runtime/tools/` and set
`SPACEFLOW_BLENDER_PATH`. An installed Blender 3.x can be selected explicitly.

An explicitly configured missing executable is an error. Blender process failures
include exit status and stderr instead of silently proceeding with missing meshes.
The Blender render script remains overridable through
`SPACEFLOW_BLENDER_RENDER_SCRIPT`.

## Preflight checks

```bash
python tools/doctor.py --editor
# Run this second check inside the actual GPU allocation:
python tools/doctor.py --gpu --json
python tools/doctor.py --gpu --image --json
```

The GPU check reports Python/PyTorch/CUDA, native imports, Blender, the checkpoint,
and the model cache. It does **not** perform generation or prove output quality.
The [verification record](verification.md) documents those separate checks.

## Useful service settings

| Variable | Purpose |
| --- | --- |
| `SQ_SPACEFLOW_STORAGE_ROOT` | Root for assets, runs, and service cache; defaults to `spaceflow_runtime`. |
| `SQ_SPACEFLOW_ASSET_ROOT` | Saved input bundles. |
| `SQ_SPACEFLOW_RUN_ROOT` | New generation runs. |
| `SQ_SPACEFLOW_PYTHON` | Explicit generation interpreter; otherwise the service's interpreter. |
| `SQ_SPACEFLOW_PORT` | Backend port; default `11438`. |
| `SQ_SPACEFLOW_HOST` | Backend bind address; the launcher defaults to localhost. |
| `SQ_EDITOR_HOST` | Editor bind address; defaults to localhost. |
| `VITE_DEV_PROXY_SPACEFLOW` | Editor proxy target; the launcher configures the backend address. |
| `SQ_SPACEFLOW_OFFLINE_CACHE` | Set `1` for a fully staged offline cache, or `0` for online downloads. |
| `TORCH_HOME` | DINOv2 source and checkpoint cache selected during image staging. |
| `U2NET_HOME` | Background-removal checkpoint cache selected during image staging. |
| `SPACEFLOW_BLENDER_PATH` | Blender executable used for mesh normalization. |

Keep model/download environments and runtime data outside version control.
