# SpaceFlow superquadric editor

React, TypeScript, and Three.js editor for SpaceFlow's geometric scaffold and
appearance prompts. It includes primitive editing, high/low control labels,
global and local prompts, NPZ import/export, saved asset history, and run results.

## Start

Use Node.js 22.12 or later. From this directory:

```bash
npm ci --include=optional
npm run dev -- --host 127.0.0.1
```

Open the URL printed by Vite, usually `http://127.0.0.1:5173`. Load a built-in
preset or import an NPZ, then edit its primitives and prompts. Editing and NPZ
downloads do not require a GPU.

To save and reopen assets, also start the Python service from the repository
root. The [UI instructions](../README.md) describe the service and Slurm options.
The development server proxies `/spaceflow` to `http://127.0.0.1:11438` by
default; set `VITE_DEV_PROXY_SPACEFLOW` for another backend address.

`bash run.sh` from the repository root starts both services using the active
Python environment. Keep runtime data outside the source tree or in the ignored
`spaceflow_runtime/` directory.

## Build

```bash
npm run build
npm test
```

The build type-checks the UI and writes the production bundle to `dist/`.
The tests check NPZ geometry and metadata round trips with Node's test runner.
Saved and downloaded NPZs retain primitive names, control labels, global/local
text prompts, and run settings. Uploaded image files need to be supplied again;
NPZ metadata stores image paths, not the image bytes. The
[controlled demo helper](../scripts/run_public_demo.sh) serves this bundle through
an authenticated gateway. Generation additionally requires the GPU runtime and
model checkpoints described in the [project README](../../README.md).

## Open a recovered example

With Vite running, append an NPZ path to the editor URL:

```text
http://127.0.0.1:5173/?npz=examples/blue_teacup_full_experiment/inputs/all.npz
```

The paths are resolved from the repository root. `SQ_UI_NPZ_ROOTS` configures
additional directories. For reproducible generation using the original
settings, use `tools/replay_example.py` rather than re-entering the parameters.
