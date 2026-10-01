# SpaceFlow

**Locally Controllable 3D Generation**

SpaceFlow is a training-free pipeline for controlling the geometry and appearance
of individual parts of a generated 3D asset. An editable superquadric scaffold
expresses the intended shape; high/low control labels adjust geometric adherence,
and global or per-part text/image conditions guide appearance.

[Project page](https://neilus03.github.io/spaceflow/) ·
[Paper and supplement](https://neilus03.github.io/spaceflow/assets/SpaceFlow-paper.pdf) ·
[Examples](examples/README.md) · [Verification](docs/verification.md)

[![Recorded sailboat example](docs/media/sailboat_spin_poster.png)](docs/media/sailboat_spin.mp4)

*Recorded result from the original experiments. Click the image for the rotating preview.*

> **Release candidate.** The fresh GPU build and end-to-end generation checks are
> pending. See the [verification record](docs/verification.md) for completed checks
> and limits. `v0.1.0` will be tagged after the GPU release checks pass.

## What is included

- The SpaceFlow generation pipeline, using TRELLIS and PartField.
- A React/Three.js editor with primitive editing, high/low control labels,
  global/local prompts, NPZ import/export, saved inputs, and generation results.
- **83 small example bundles:** primitives, prompts, metadata, and recorded parameters.
- Optional baseline runners, experiment replay, comparison rendering, and metrics.
- Pinned Python dependencies, model revisions, native extension revisions, and tests.

Model weights and generated experiment results are separate artifacts. Running an
example writes new results into the ignored `runs/` directory. The source package
contains no SuperDec runs or large checkpoints.

## Requirements

| Component | Editor and save/reopen service | Full generation |
| --- | --- | --- |
| Operating system | macOS or Linux | Linux x86-64 with an NVIDIA GPU |
| Python | 3.10–3.12 | **3.10** |
| Node.js | **24 recommended**, or 22.12+; npm included | Required only when running the editor |
| GPU/toolkit | Not required | CUDA **12.8** toolkit and a compatible NVIDIA driver |
| PyTorch | Not required | **2.8.0/cu128**, installed by `setup.sh` |
| Blender | Not required | Blender **3.x**; original experiments used **3.0.1** |
| Models | Not required | Pinned TRELLIS/CLIP models and the PartField checkpoint |

GPU memory requirements and runtime on the fresh release are not yet measured.
Build CUDA extensions inside a GPU allocation on Slurm. Detailed installation
and model-cache notes are in [Installation](docs/installation.md).

## 1. Clone the release

```bash
git clone --depth 1 --branch RELEASE https://github.com/joanlafuente/spaceflow.git
cd spaceflow
```

The shallow clone retrieves the current source without downloading the large
older research history. Existing research branches remain available.

## 2. Open the editor without a GPU

From the repository root, with Python 3.10–3.12 and Node.js available:

```bash
python3 -m venv .venv-editor
source .venv-editor/bin/activate
python -m pip install -r requirements/editor.txt

(cd sq_ui/app && npm ci --include=optional)
python tools/doctor.py --editor
bash run.sh
```

Open the URL printed by Vite, normally **http://127.0.0.1:5173**. The launcher
starts the asset service on **127.0.0.1:11438** and the editor on the printed port.

To open the supplied teacup directly, append this query to the printed editor URL:

```text
http://127.0.0.1:5173/?npz=examples/blue_teacup_full_experiment/inputs/all.npz
```

The example restores primitive names, control labels, global/local text prompts,
and saved run settings. You can edit primitives, download NPZs, or save and reopen
inputs using the service. **Generation needs the full GPU environment below.**
Stop the editor and service with `Ctrl+C`.

NPZs preserve image-path metadata, **not uploaded image bytes**. Supply the image
files again after reopening an image-conditioned scene. See [UI guide](sq_ui/README.md).

## 3. Generate a textured teacup

Run these steps from the repository root on a Linux GPU machine, with Python
3.10 and the CUDA 12.8 toolkit available. On Slurm, obtain a GPU allocation first.

```bash
SPACEFLOW_SETUP_PYTHON=python3.10 bash setup.sh
source .venv/bin/activate
python tools/cache_models.py --cache-dir spaceflow_runtime/huggingface
export HF_HOME="$PWD/spaceflow_runtime/huggingface"
```

Install Blender if it is not already available:

```bash
mkdir -p spaceflow_runtime/tools
curl --fail --location \
  https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz \
  --output spaceflow_runtime/tools/blender-3.0.1-linux-x64.tar.xz
tar -xf spaceflow_runtime/tools/blender-3.0.1-linux-x64.tar.xz -C spaceflow_runtime/tools
export SPACEFLOW_BLENDER_PATH="$PWD/spaceflow_runtime/tools/blender-3.0.1-linux-x64/blender"
```

If you already have Blender 3.x, set `SPACEFLOW_BLENDER_PATH` to that executable
instead. Then check the environment and replay the supplied SpaceFlow variant:

```bash
python tools/doctor.py --gpu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python tools/replay_example.py \
  --example-dir examples/blue_teacup_full_experiment \
  --output-dir runs/blue_teacup \
  --only 01_local_tau3_tau10_polyak0p18
python tools/check_replay_outputs.py runs/blue_teacup
```

The expected textured asset is:

```text
runs/blue_teacup/output/01_local_tau3_tau10_polyak0p18/out_sim.glb
```

Use a **new output directory** for each run. This replay preserves the saved
prompts and settings, including 300 refinement steps. The checker requires
successful completion, nonempty finite geometry, a baked texture, and valid UVs.
Visual quality must also be inspected; this check does not guarantee perceptual
reproduction of every research example.

To prepare the configuration without generation, add `--prepare-only` to the
replay command. Three useful starting examples are the teacup,
`01_a_chair_full_experiment`, and `13_sailboat_full_experiment`.

## 4. Generate through the UI

On the allocated GPU machine, with the generation environment activated and the
model/Blender variables configured:

```bash
source .venv/bin/activate
python tools/doctor.py --gpu
bash run.sh
```

1. Open the editor and import a supplied NPZ or select a preset.
2. Edit primitives and choose high/low geometry control for each part.
3. Open the **SpaceFlow** panel, set the shape prompt and global appearance,
   and optionally set per-part appearance conditions.
4. Start a SpaceFlow run, follow its status/log, and inspect the final result.
5. Download the final GLB or download the edited input bundle for reuse.

For SSH access, forward the **editor's printed port** and the service port:

```bash
ssh -L 5173:127.0.0.1:5173 -L 11438:127.0.0.1:11438 user@gpu-host
```

Use the full runtime for generation; the CPU editor environment only handles
editing and saved inputs. For a service on a cluster login node, configure your
valid Slurm account/partition as described in [Research workflows](docs/research.md).

Text-conditioned examples use the pinned cache above. **Image appearance
conditioning needs additional TRELLIS image/DINOv2 downloads and a network-enabled
first run**; its fresh verification is pending. See [Installation](docs/installation.md).

## Optional research workflows

The example replay command runs only SpaceFlow when `--only` selects the variant
above. Omitting `--only` runs the recorded comparisons as well and uses more GPU
time. See [Research workflows](docs/research.md) for baselines, Slurm jobs, metrics,
and CPU comparison rendering.

## Checks and troubleshooting

With the CPU editor environment activated, from the repository root:

```bash
python -m unittest discover -s tests -v
python tools/validate_examples.py
(cd sq_ui/app && npm test && npm run build)
```

The [prepared CI workflow](ci/release-checks.yml) repeats the CPU workflows and
editor build. Publishing it to GitHub Actions currently awaits workflow upload
permission; local checks have passed. GPU tests require an allocated GPU and are
recorded separately. The [verification record](docs/verification.md)
distinguishes input/configuration checks from completed generation.

| Symptom | Action |
| --- | --- |
| `cgi` missing / Python 3.13+ | Use Python 3.10–3.12 for the editor service, or 3.10 for generation. |
| Editor dependencies or native binding missing | Run `npm ci --include=optional` in `sq_ui/app` with a supported Node.js version. |
| `nvcc` missing | Load/install the CUDA 12.8 toolkit before extension compilation. |
| CUDA unavailable | Check the driver and GPU allocation; run `python tools/doctor.py --gpu` on the GPU node. |
| Missing CUDA extension | Run `SPACEFLOW_SETUP_STAGE=extensions bash setup.sh` inside the GPU allocation. |
| Missing models | Run `tools/cache_models.py` and use the same directory for `HF_HOME`. |
| Missing Blender or renderer failure | Set `SPACEFLOW_BLENDER_PATH`; the error includes Blender's stderr. |
| Output directory already exists | Choose a new replay output directory so previous results are preserved. |
| Images unavailable offline | Allow the first image-conditioned run to download the additional models. |

## Citation and attribution

The preprint citation follows the [project page](https://neilus03.github.io/spaceflow/):

```bibtex
@misc{delafuente2026spaceflow,
  title  = {SpaceFlow: Locally Controllable 3D Generation},
  author = {De La Fuente, Neil and Lafuente Baeza, Joan and
            Sayfiddinov, Mukhammadali and Scharitzer, Felicia and
            Pollefeys, Marc and Çelen, Ata and
            Deb Sarkar, Sayan and Fedele, Elisabetta},
  year   = {2026},
  note   = {Preprint}
}
```

Please also record the software commit used in your experiments.

SpaceFlow builds on [GuideFlow3D](https://github.com/GradientSpaces/GuideFlow3D),
[TRELLIS](https://github.com/microsoft/TRELLIS), and
[PartField](https://github.com/nv-tlabs/PartField).
The repository's [Apache-2.0 license](LICENSE) does not replace upstream terms.
**PartField is restricted to non-commercial research and educational use.**
See [third-party notices](THIRD_PARTY_NOTICES.md) and the component license files.
