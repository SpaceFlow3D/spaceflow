# SpaceFlow

SpaceFlow: Part-Wise Spatial and Semantic Guidance for Controllable 3D
Generation

SpaceFlow explores a practical question for text-to-3D systems: how can a user keep coarse geometric intent while still benefiting from a generative model's learned shape and appearance priors? The system uses a superquadric scene as an explicit control signal, applies local diffusion-time control over selected parts, and then refines the generated mesh with part-aware similarity guidance.

<video src="docs/media/sailboat_spin.mp4" controls muted loop playsinline poster="docs/media/sailboat_spin_poster.png" width="520"></video>


https://github.com/user-attachments/assets/c4c95db5-bb5f-4100-bb1a-42addd9a8e40



## Overview

Given a text prompt and an editable set of superquadrics, SpaceFlow produces a textured 3D asset through three stages:

1. **Spatial scaffold.** The UI exports all superquadrics, a high-control subset, and a low-control mask or bounding box.
2. **Structure generation.** TRELLIS generates sparse structure under SpaceControl constraints. Local-tau mode can apply different diffusion-time strengths to different spatial regions.
3. **Part-aware refinement.** PartField features and similarity guidance refine the generated structure toward global and local text or image appearance conditions.

The repository is a minimal delivery version of the project. It keeps the runtime pipeline, the superquadric editor, experiment launchers, local TRELLIS pipeline configuration, and the vendored code needed to reproduce the course experiments.

## Method

Superquadrics are used as a compact, editable proxy for user intent. They are expressive enough to block out object parts, but simple enough to manipulate interactively. Each primitive can be tagged as high-control or low-control. In local-tau experiments, high-control regions preserve the scaffold more strongly while low-control regions allow the generative prior to move more freely.

The local control variant passes the following signals into structure generation:

- `spatial_control_mesh.ply`: full superquadric control mesh.
- `high_control_spatial_control_mesh.ply`: subset of primitives that should be preserved more strongly.
- `low_control_superquadric_mask.ply`: low-control region used for local-tau masking.
- `shape_tau`, `shape_tau_high_control`, and `polyak_update_tau`: diffusion-time and model-averaging parameters for the local control schedule.

After structure generation, the mesh is normalized through Blender/TRELLIS tooling, voxelized, embedded with PartField, and optimized with similarity losses. The current pipeline supports global appearance prompts plus per-superquadric local text or image prompts.

## Qualitative Example

The sailboat case study was generated from:

- Prompt: `Sailboat`
- Global appearance: `white`
- Local appearance: `yellow` on four low-control primitives
- Local-tau settings: low tau `3`, high tau `10`, Polyak tau `0.18`

The figure above shows the intended control layout, the intermediate TRELLIS structure, and the final similarity-refined asset. The rotating preview is rendered from the resulting `out_sim.glb`.

Regenerate the README media with:

```bash
python docs/render_readme_media.py /path/to/completed/sailboat/run
```

## Repository Layout

- `run_local_tau.py`: main SpaceFlow pipeline entrypoint.
- `config/default.yaml`: runtime configuration and local TRELLIS pipeline path.
- `config/trellis_pipeline/pipeline.json`: mixed TRELLIS image/text model configuration used by spatial-control runs.
- `sq_ui/app`: React/Vite superquadric editor.
- `sq_ui/scripts/spaceflow_service.py`: HTTP service that saves SQ assets and launches local or Slurm runs.
- `sq_ui/scripts/run_spaceflow_experiment.py`: multi-variant experiment runner.
- `sq_ui/scripts/render_spaceflow_experiment_comparison.py`: CPU rasterizer for shared-view experiment figures.
- `docs/render_readme_media.py`: headless renderer for README figures and rotating previews.
- `lib`, `third_party`, `utils.py`: optimization, rendering, geometry, PartField, and TRELLIS runtime code.

## Running

Clone the `MINIMAL` branch for this runtime and UI. A shallow clone avoids
downloading the older research history:

```bash
git clone --depth 1 --branch MINIMAL https://github.com/joanlafuente/spaceflow.git
cd spaceflow
```

The generation pipeline requires Linux, Python 3.10, an NVIDIA GPU, and the CUDA
12.8 toolkit. The editor and asset save/reopen service can also run without a GPU.
The installer creates a separate `.venv` and pins PyTorch 2.8.0 with CUDA 12.8,
Kaolin 0.18.0 from the matching wheel index, and the native renderer revisions.
On a cluster, build the CUDA extensions inside a GPU allocation.

```bash
bash setup.sh
source .venv/bin/activate
```

Installation can be split into dependency downloads and GPU extension builds:

```bash
SPACEFLOW_SETUP_STAGE=deps bash setup.sh
# Run this second stage on an allocated GPU node with the CUDA 12.8 toolkit loaded:
SPACEFLOW_SETUP_STAGE=extensions bash setup.sh
```

Build the editor with Node.js 22.12 or later:

```bash
cd sq_ui/app
npm ci --include=optional
npm run build
npm test
```

From the repository root, start both the backend and editor:

```bash
bash run.sh
```

Then open the local URL printed by Vite. For SSH use, forward the editor port
and backend port 11438 to your workstation. Keep the service on an allocated
GPU node when it is expected to launch generation locally.

For service-managed Slurm jobs launched from a login node, explicitly configure
your valid allocation and partition. These values vary by institution:

```bash
export SQ_SPACEFLOW_SLURM_ACCOUNT=your_account
export SQ_SPACEFLOW_SLURM_PARTITION=your_gpu_partition
# Optional: SQ_SPACEFLOW_SLURM_GPUS, SQ_SPACEFLOW_SLURM_CONSTRAINT,
# SQ_SPACEFLOW_SLURM_TIME, SQ_SPACEFLOW_SLURM_EXCLUDE, SQ_SPACEFLOW_SLURM_EXTRA_ARGS.
bash run.sh
```

The service writes assets and runs under `spaceflow_runtime/` by default. Override paths with `SQ_SPACEFLOW_STORAGE_ROOT`, `SQ_SPACEFLOW_ASSET_ROOT`, or `SQ_SPACEFLOW_RUN_ROOT`.

## Runtime Notes

The pipeline expects TRELLIS and PartField checkpoints to be available through the configured Hugging Face cache or online download. The PartField checkpoint is expected at:

```text
third_party/PartField/models/model_objaverse.ckpt
```

For online Hugging Face access, keep `SQ_SPACEFLOW_OFFLINE_CACHE=0` or unset it. Offline cache mode can be enabled explicitly with:

```bash
SQ_SPACEFLOW_OFFLINE_CACHE=1 python sq_ui/scripts/spaceflow_service.py
```

The renderer uses Blender for pipeline mesh normalization. Set `SPACEFLOW_BLENDER_PATH=/path/to/blender` to use a specific Blender install.

Stage the recorded TRELLIS/CLIP revisions and the official PartField checkpoint
before running the pipeline:

```bash
python tools/cache_models.py --cache-dir spaceflow_runtime/huggingface
export HF_HOME="$PWD/spaceflow_runtime/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

The cache tool downloads the revisions in `requirements/model-revisions.json`
and checks that PartField matches the SHA256 of the recovered checkpoint.
This stages the mixed text/structure pipeline used by the bundled examples.
Image appearance conditioning requires additional image-model downloads and has
not been checked in the fresh verification environment.

## Replay an Example End to End

The `examples/` directory contains 83 recovered primitive bundles and their
saved prompts, conditions, and experiment parameters. Model checkpoints and
previously generated results are separate data artifacts.

After installing the runtime and supplying the PartField checkpoint and Blender,
run the SpaceFlow local control variant for the teacup example:

```bash
python tools/replay_example.py \
  --example-dir examples/blue_teacup_full_experiment \
  --output-dir runs/blue_teacup \
  --only 01_local_tau3_tau10_polyak0p18
python tools/check_replay_outputs.py runs/blue_teacup
```

Choose a new output directory for each replay. The tool copies the inputs,
relocates paths, and preserves the saved prompts, seeds, and optimization settings
(including the 300-step refinement). Omitting `--only` runs all saved experiment
variants; selecting a dependent variant also includes its required source variant.
Use `--prepare-only` to inspect the relocated configuration before generation.

The expected final asset is `output/01_local_tau3_tau10_polyak0p18/out_sim.glb`.
The output checker requires a successful run marker and finite, nonempty mesh
geometry with a baked texture and valid UV coordinates. It does not establish
visual quality or reproduction of all 83 cases.

For Slurm verification, submit from the repository root after loading your
cluster's compiler, CUDA 12.8, Python, and Blender modules:

```bash
sbatch --account=your_account --partition=your_gpu_partition tools/verify_spaceflow.sbatch
```

Set `SPACEFLOW_BUILD_EXTENSIONS=1` to build the native extensions at the beginning
of this job. `SPACEFLOW_VENV`, `SPACEFLOW_VERIFY_EXAMPLE`, and
`SPACEFLOW_VERIFY_VARIANT` select a separate environment or another saved case.
The default job requests one GPU, four CPUs, 64 GB RAM, and at most four hours.

## Verification

CPU workflow checks cover asset save/history/reopen and path-safe example replay:

```bash
python -m unittest discover -s tests -v
```

The recovered release has passed these checks, Python/shell syntax checks, and a
clean editor build. The pinned Python dependencies install in a fresh Linux
environment and pass `pip check`. CUDA extension installation and a fresh GPU
pipeline run have not yet been completed. Treat this as a release candidate
until both pass. See [the verification record](docs/verification.md) for scope.

## Attribution

SpaceFlow uses [GuideFlow3D](https://github.com/GradientSpaces/GuideFlow3D),
[TRELLIS](https://github.com/microsoft/TRELLIS), and
[PartField](https://github.com/nv-tlabs/PartField), with project-specific runtime
changes. Vendored components retain their own license files and source notices;
the root license does not replace their terms. The native dependencies and their
selected revisions are listed in `requirements/native-revisions.json`.

## Status

This is a research release candidate. Generated experiment outputs, Slurm logs,
downloaded model weights, and large checkpoints are excluded from version control.
The bundled examples contain inputs and saved configurations. The media in
`docs/media/` is a qualitative snapshot from the original experiments.
