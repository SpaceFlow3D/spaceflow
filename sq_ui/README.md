# SpaceFlow editor and service

The editor provides primitive manipulation, high/low control labels, global and
per-part appearance prompts, NPZ import/export, saved inputs, and run results.

## Start locally

From the repository root, follow the [CPU editor quickstart](../README.md#editor-without-a-gpu).
It installs `requirements/editor.txt` and the frontend dependencies, then starts
both services with `bash run.sh`.

- Editor: the URL printed by Vite, normally `http://127.0.0.1:5173`.
- Backend: `http://127.0.0.1:11438`.
- Health endpoint: `/spaceflow/health`.

The editor proxies `/spaceflow` requests to the backend, keeping requests on the
same origin. `VITE_DEV_PROXY_SPACEFLOW` selects another backend address.

## Open and save examples

Import an NPZ or append a repository-relative path to the printed URL:

```text
http://127.0.0.1:5173/?npz=examples/25_20260606T002730Z_toy_elephant_full_experiment/inputs/all.npz
```

Opening the all-primitives file restores part names, geometry-control labels,
global/local text prompts, and saved settings. The asset service supports
save/history/reopen. NPZ downloads work directly in the browser.

Image path metadata is preserved, but uploaded image bytes are not embedded in
the NPZ. Reattach the images when reopening an image-conditioned scene. Local
image conditions use the same appearance mode as the global condition.

`SQ_UI_NPZ_ROOTS` configures additional folders allowed by the development server.
Generation inputs and result storage are controlled by the variables in
[Installation](../docs/installation.md).

## Generate a result

Use the [full GPU environment](../README.md#installation), activate
`.venv`, and set the model-cache and Blender variables before `bash run.sh`. The
CPU editor environment does not include generation dependencies.

In the SpaceFlow panel, enter the shape prompt, choose global appearance, and
add local conditions for selected parts. Start a run, follow its status/log,
inspect the output, and download the final GLB. Retained experiment buttons
launch recorded comparisons and require additional GPU time.

For a remote GPU, forward the printed editor port and backend port over SSH.
Slurm account/partition settings are covered in [Research workflows](../docs/research.md).

## Frontend development

```bash
cd sq_ui/app
npm ci --include=optional
npm test
npm run dev -- --host 127.0.0.1
```

Start the backend separately from the repository root using the activated editor
or generation environment:

```bash
python sq_ui/scripts/spaceflow_service.py
```

`npm run build` type-checks the app and produces `dist/`. Frontend tests cover
geometry and metadata round trips, including legacy metadata and zero settings.
The [controlled demo helper](../docs/controlled-demo.md) is an optional advanced
workflow. Normal editor usage requires neither a public tunnel nor a shared password.
