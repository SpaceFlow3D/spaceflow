<h1 align="center">
  <a href="https://spaceflow3d.github.io/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/media/readme/spaceflow-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="docs/media/readme/spaceflow-light.svg">
      <img src="docs/media/readme/spaceflow-light.svg" width="680" alt="SpaceFlow: Locally Controllable 3D Generation">
    </picture>
  </a>
</h1>

<p align="center">
  <a href="https://neilus03.github.io/">Neil De La Fuente</a><sup>1*</sup> &nbsp;
  <a href="https://github.com/joanlafuente">Joan Lafuente</a><sup>1*</sup> &nbsp;
  <a href="https://www.linkedin.com/in/mukali/">Mukhammadali Sayfiddinov</a><sup>1*</sup> &nbsp;
  <a href="https://ch.linkedin.com/in/felicia-scharitzer/de">Felicia Scharitzer</a><sup>1*</sup>
  <br>
  <a href="https://people.inf.ethz.ch/pomarc/">Marc Pollefeys</a><sup>1,3</sup> &nbsp;
  <a href="https://github.com/atcelen">Ata Çelen</a><sup>1</sup> &nbsp;
  <a href="https://sayands.github.io/">Sayan Deb Sarkar</a><sup>2†</sup> &nbsp;
  <a href="https://elisabettafedele.github.io/">Elisabetta Fedele</a><sup>1†</sup>
</p>

<p align="center">
  <sup>1</sup> ETH Zürich &nbsp; · &nbsp; <sup>2</sup> Stanford University &nbsp; · &nbsp; <sup>3</sup> Microsoft<br>
  <sub>* Equal contribution (ordered alphabetically) &nbsp; · &nbsp; † Equal supervision</sub>
</p>

<p align="center">
  <a href="https://ethz.ch/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/media/readme/ethz-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="docs/media/readme/ethz-light.svg">
      <img src="docs/media/readme/ethz-light.svg" width="160" align="middle" alt="ETH Zürich">
    </picture>
  </a>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <a href="https://www.stanford.edu/">
    <img src="docs/media/readme/stanford-s.png" width="96" height="96" align="middle" alt="Stanford University">
  </a>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <a href="https://www.microsoft.com/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/media/readme/microsoft-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="docs/media/readme/microsoft-light.png">
      <img src="docs/media/readme/microsoft-light.png" width="230" align="middle" alt="Microsoft">
    </picture>
  </a>
</p>

<p align="center">
  <strong>TL;DR</strong> · Training-free 3D generation with local geometry and appearance control.<br>
  Edit primitives, choose which parts to preserve or complete,<br>
  and guide their appearance with text or reference images.
</p>

<p align="center">
  <a href="https://spaceflow3d.github.io/"><img src="docs/media/readme/link-project.svg" width="144" height="36" alt="Project Page"></a>
  <a href="https://spaceflow3d.github.io/assets/SpaceFlow-paper.pdf"><img src="docs/media/readme/link-paper.svg" width="194" height="36" alt="Paper &amp; Supplement"></a>
  <a href="https://spaceflow3d.github.io/#explore"><img src="docs/media/readme/link-examples.svg" width="197" height="36" alt="Interactive Examples"></a>
  <a href="https://github.com/joanlafuente/spaceflow/releases"><img src="docs/media/readme/link-downloads.svg" width="134" height="36" alt="Downloads"></a>
</p>

https://github.com/user-attachments/assets/dc72c902-e415-4aea-abb6-e576b5b5705e

<p align="center">
  <a href="#installation"><strong>Installation</strong></a> &nbsp; · &nbsp;
  <a href="#generate-an-example">Generate an example</a> &nbsp; · &nbsp;
  <a href="#interactive-editor">Interactive editor</a> &nbsp; · &nbsp;
  <a href="docs/README.md">More documentation</a>
</p>

## Installation

The generation steps below run on a **Linux GPU machine**. To author primitives
on your laptop, use the [editor without a GPU](#editor-without-a-gpu).

### Requirements

| Component | Generation |
| --- | --- |
| System | Linux x86-64; use a distribution supported by CUDA 12.8 |
| Python | **3.10**, with `venv` support |
| GPU | NVIDIA GPU; **24 GB VRAM recommended** |
| CUDA | **12.8 toolkit**, including `nvcc`, and a compatible NVIDIA driver |
| Blender | **3.0.1** in the instructions below, or an installed Blender 3.x |
| Storage | About **60 GB free** for dependencies, models, tools, and initial outputs |
| Node.js | **24 with npm** for the interactive editor; not needed for command-line generation |

Install Python and the CUDA toolkit before running the setup script. On Slurm,
obtain a GPU allocation before compiling extensions. See the
[installation guide](docs/installation.md) for cluster setup, compiler/platform
notes, and alternative tool locations.

### 1. Clone and install

```bash
git clone --depth 1 --branch RELEASE https://github.com/joanlafuente/spaceflow.git
cd spaceflow

SPACEFLOW_SETUP_PYTHON=python3.10 bash setup.sh
source .venv/bin/activate
```

The setup script creates `.venv`, installs the pinned Python dependencies, and
builds the CUDA extensions. **Run all following commands from this repository
root**, with `.venv` activated.

### 2. Download the models

```bash
export HF_HOME="$PWD/spaceflow_runtime/huggingface"
python tools/cache_models.py --cache-dir "$HF_HOME"
```

This downloads the required TRELLIS, CLIP, and PartField weights and places the
PartField checkpoint in its expected location. The first download needs internet
access. Image appearance conditioning needs the
[additional models described below](#use-reference-images).

### 3. Configure Blender

Install Blender 3.0.1 inside the project:

```bash
mkdir -p spaceflow_runtime/tools
curl --fail --location \
  https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz \
  --output spaceflow_runtime/tools/blender-3.0.1-linux-x64.tar.xz
tar -xf spaceflow_runtime/tools/blender-3.0.1-linux-x64.tar.xz -C spaceflow_runtime/tools
export SPACEFLOW_BLENDER_PATH="$PWD/spaceflow_runtime/tools/blender-3.0.1-linux-x64/blender"
```

If Blender 3.x is already installed, skip the download and set
`SPACEFLOW_BLENDER_PATH` to its executable. Check that the environment is ready:

```bash
python tools/doctor.py --gpu
```

## Generate an example

Start with the supplied **toy elephant**. Its saved configuration uses **blue
painted wood** globally and **pink painted wood** on the ears, illustrating local
appearance control. This command uses its saved primitives, prompts, and
generation settings:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python tools/replay_example.py \
  --example-dir examples/25_20260606T002730Z_toy_elephant_full_experiment \
  --output-dir runs/toy_elephant \
  --only 01_spaceflow_local_texture_routing
```

The final textured mesh is saved at:

```text
runs/toy_elephant/output/01_spaceflow_local_texture_routing/out_sim.glb
```

Open the GLB in Blender or another glTF viewer. Inputs, configuration, and logs
are saved alongside the output. **Choose a new `--output-dir` for each run.**
To try another asset, replace `--example-dir` with one of the
[83 supplied examples](examples/README.md).

## Interactive editor

The editor lets you build or import a set of primitives, adjust local controls,
set appearance prompts, and generate/download the result.



https://github.com/user-attachments/assets/7ffc4252-549d-4472-96b6-7dff11152a75



### Start the editor on your GPU machine

After completing installation, run from the repository root:

```bash
source .venv/bin/activate
export HF_HOME="$PWD/spaceflow_runtime/huggingface"
export SPACEFLOW_BLENDER_PATH="$PWD/spaceflow_runtime/tools/blender-3.0.1-linux-x64/blender"
export SQ_SPACEFLOW_OFFLINE_CACHE=1

(cd sq_ui/app && npm ci --include=optional)
bash run.sh
```

If you use another Blender installation, set its path instead. Open the URL
printed by the launcher, normally **http://127.0.0.1:5173**. Keep this terminal
running; `Ctrl+C` stops the editor and its service.

Load the toy elephant directly by opening:

```text
http://127.0.0.1:5173/?npz=examples/25_20260606T002730Z_toy_elephant_full_experiment/inputs/all.npz
```

If the launcher prints another port, use that port in the URL.

### Edit, generate, and download

1. **Edit the shape.** Select a primitive in **Scene** and change its scale,
   position, rotation, or deformation in **Properties**.
2. **Choose geometry control.** **High** encourages closer adherence to the
   primitive; **Low** gives the generator more freedom. Keep at least one visible
   high-control and one low-control primitive.
3. **Set appearance.** Open **SpaceFlow**. Enter the **Shape prompt** (for example,
   `toy elephant`) and choose the global text under **Texture guidance** (for
   example, `blue painted wood`). Select an ear primitive and use **Local texture
   → Text override** for its appearance (for example, `pink painted wood`). An
   empty override uses the global condition.
4. **Generate.** Leave **Dry run** unchecked and click **Run**. Follow progress
   and the run log in the SpaceFlow panel.
5. **Inspect and download.** When the run succeeds, click **Inspect mesh**, then
   **Export → Download GLB mesh**.

**Save inputs** stores the scene for reopening through **Show saved**. Use the
**Export** menu to download primitive NPZ files for later editing or replay.
Generated UI runs are stored in
`spaceflow_runtime/sq_ui_runs/<run-id>/output/`; the final asset is `out_sim.glb`.

### Connect to a remote GPU

Run the editor on the GPU machine. In a separate terminal on your laptop,
replace `user@gpu-host` with your SSH login and forward both ports:

```bash
ssh -N -L 5173:127.0.0.1:5173 -L 11438:127.0.0.1:11438 user@gpu-host
```

Keep the SSH terminal open and visit **http://127.0.0.1:5173** on your laptop.
Use the editor's printed port if it differs from 5173. See the [UI guide](sq_ui/README.md) for custom ports and
[cluster instructions](docs/research.md) for Slurm.

### Use reference images

Before your first image-conditioned run, stop the editor with `Ctrl+C` and stage
the additional models from the repository root, with `.venv` activated:

```bash
export HF_HOME="$PWD/spaceflow_runtime/huggingface"
export TORCH_HOME="$PWD/spaceflow_runtime/torch"
export U2NET_HOME="$PWD/spaceflow_runtime/u2net"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python tools/cache_models.py --cache-dir "$HF_HOME" --include-image
python tools/doctor.py --gpu --image
bash run.sh
```

Choose **Image** under **Texture guidance**, upload a global reference, and
optionally assign reference images to selected primitives in **Local texture**.
Keep the cache variables above set when restarting the editor.

NPZ files store image paths, **not the image bytes**. Reattach image files when
reopening an image-conditioned scene.

## Editor without a GPU

For primitive editing and input preparation on macOS or Linux, use Python
**3.10–3.12** and Node.js **24**. From the repository root:

```bash
python3 -m venv .venv-editor
source .venv-editor/bin/activate
python -m pip install -r requirements/editor.txt
(cd sq_ui/app && npm ci --include=optional)
bash run.sh
```

Open the printed URL and import an example NPZ. You can edit, save/reopen, and
export inputs. To generate an asset, run the editor with the GPU environment
above or transfer the inputs to your GPU machine.

## Examples and documentation

Good starting points in [`examples/`](examples/README.md):

| Asset | Example directory |
| --- | --- |
| Toy elephant (quickstart) | `examples/25_20260606T002730Z_toy_elephant_full_experiment` |
| Chair | `examples/01_a_chair_full_experiment` |
| Sailboat | `examples/13_sailboat_full_experiment` |

Each bundle includes primitives, prompts, metadata, and a replay configuration.
Use `inputs/all.npz` to open it in the editor, or its directory with
`tools/replay_example.py`.

The [documentation index](docs/README.md) covers installation troubleshooting,
UI settings, Slurm, baselines, comparisons, and the release verification record.
For questions or bugs, [open an issue](https://github.com/joanlafuente/spaceflow/issues).

## Citation

If you use SpaceFlow in your research, please cite:

```bibtex
@misc{delafuente2026spaceflow,
  title  = {SpaceFlow: Locally Controllable 3D Generation},
  author = {De La Fuente, Neil and Lafuente, Joan and
            Sayfiddinov, Mukhammadali and Scharitzer, Felicia and
            Pollefeys, Marc and Çelen, Ata and
            Deb Sarkar, Sayan and Fedele, Elisabetta},
  year   = {2026},
  note   = {Preprint}
}
```

## Acknowledgements and license

SpaceFlow builds on [GuideFlow3D](https://github.com/GradientSpaces/GuideFlow3D),
[TRELLIS](https://github.com/microsoft/TRELLIS), and
[PartField](https://github.com/nv-tlabs/PartField).

SpaceFlow is released under [Apache-2.0](LICENSE). **PartField is restricted to
non-commercial research and educational use.** See
[third-party notices](THIRD_PARTY_NOTICES.md) for component licenses and attribution.
