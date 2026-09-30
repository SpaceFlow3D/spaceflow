# Verification record

Checked on 2026-09-30. This is a research release candidate; a fresh GPU run is
still required before claiming full end-to-end reproducibility.

## Recovered source

The source came from the `MINIMAL` working tree based on commit
`1d845ecdbcc4409b4ac82ffa6de8ccdf710626c5`. All 213 tracked files were restored
and checked against the preserved source hashes. The 14 uncommitted edits are
retained in the separate recovery commit `96b9cea` before the portability changes.
The original dirty worktree, inputs, metadata, and historical outputs were backed
up independently.

This checkout contains runtime source, the editor/service, dependency and model
revision records, tests, licenses, and 83 small input bundles. Checkpoints,
downloaded environments, generated meshes, and old run directories are excluded.
The repository's older Git history remains available; use the documented shallow
`MINIMAL` clone to download only the current code.

## Passed checks

| Check | Result and scope |
| --- | --- |
| Backend CPU workflow | Four Python tests passed, including real HTTP save/history/reopen and replay path preservation. |
| UI regression tests | Four tests passed for NPZ geometry, prompt/name/settings preservation, zero values, empty prompts, geometry-only inputs, and legacy NumPy-wrapped byte-string metadata. |
| UI installation/build | Clean `npm ci --include=optional`, TypeScript check, and production Vite build passed with Node.js 24.19.0. Vite reports a large JavaScript chunk warning. |
| Browser workflow | Imported real teacup primitives, edited a local text condition, saved inputs, started a fresh browser session, and reopened the geometry, names, control levels, global/local prompts, and non-default run settings. No browser console errors were observed. |
| Input bundles | All 249 NPZ inputs have finite numeric values and expected primitive array shapes. |
| Replay preparation | All 83 saved experiment configurations prepare in new directories with relocated paths. This does not execute generation. |
| Source syntax | All 151 Python files and four shell/Slurm scripts passed syntax checks. |
| Fresh Linux dependencies | Python 3.10.13 environment with pinned PyTorch 2.8.0/CUDA 12.8 dependencies installed and passed `pip check`. |
| Model staging | Recorded mixed TRELLIS/CLIP revisions downloaded; PartField checkpoint matches the recorded SHA256. |

## Pending checks and limits

- Build and import the CUDA extensions on an allocated GPU with the CUDA 12.8
  toolkit, then replay the saved 300-step SpaceFlow teacup variant using
  `tools/verify_spaceflow.sbatch`.
- Check the fresh final mesh, baked texture, and UV coordinates with
  `tools/check_replay_outputs.py`, then inspect visual quality separately.
- The staged models cover the mixed text/structure pipeline used by the bundled
  examples. Image appearance conditioning needs additional model downloads and
  has not been verified in this fresh environment.
- NPZ saves preserve text conditions and image path metadata. Uploaded image
  bytes must be supplied again when reopening; they are not embedded in the NPZ.
- Valid historical outputs and a passing UI build do not establish fresh GPU
  reproduction of every example or baseline.

## Repeat the CPU and UI checks

From the repository root, with Python 3.10 (or Python 3.12 for CPU checks):

```bash
python -m unittest discover -s tests -v
cd sq_ui/app
npm ci --include=optional
npm test
npm run build
```

The generation runtime uses Python 3.10. The asset service currently relies on
the standard-library `cgi` module and does not support Python 3.13 or later.
