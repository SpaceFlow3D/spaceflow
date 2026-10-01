# Verification record

Checked on **2026-10-01**. `RELEASE` is a release candidate derived from the
published `MINIMAL` commit `01accc583f20166cb12f4f370cb99c8c6cd8ee24`.
Fresh GPU acceptance is pending; no stable `v0.1.0` is claimed.

## Completed release checks

| Check | Result and scope |
| --- | --- |
| CPU tests | **21 passed** under Python 3.12.14, including real HTTP save/history/reopen, replay preservation, invalid geometry rejection, Blender diagnostics, appearance pipeline selection, immutable DINOv2 revision checks, runtime dependency diagnostics, clean preflight JSON, scoped PartField checkpoint compatibility, and verification-matrix preparation. |
| Fresh published clone | Independently cloned `RELEASE` from GitHub at `d94e400a3d48ccb8d2f294acebcb93e689dc182e`, created new Python and Node dependency environments, then repeated the 15 CPU tests, 5 UI tests, all 83 input/replay checks, GPU-matrix preparation, and production build successfully. |
| Input validation | All **249 NPZ files in 83 cases** have finite numeric arrays with the expected primitive shapes. All 83 replay configurations prepare in new directories. Original input files are unchanged. |
| UI regression tests | **5 passed**: NPZ round trips, primitive names, global/local prompts, run settings, empty values, geometry-only/legacy inputs, and delayed download URL cleanup. |
| Editor installation/build | Fresh `npm ci --include=optional`, TypeScript check, and Vite production build passed with Node.js **24.19.0** and npm **11.6.4**. The production build has a large JavaScript chunk warning. |
| Dependency audit | `npm audit` reports **zero known vulnerabilities** in this lockfile on the check date. This is a registry advisory check, not a guarantee against all vulnerabilities. |
| Browser editing | Imported the real teacup; edited its name, geometry, local/global appearance, and control settings; saved inputs; reopened in a fresh session with geometry, names, prompts, and settings restored. Tested in the Codex in-app browser and Chrome. |
| Browser downloads/import | Chrome downloaded NPZ and primitive PNG files. The downloaded NPZ's geometry and edited metadata were checked with NumPy and imported again through the browser file chooser. Download-event capture in the in-app browser timed out, so download support there is not claimed. |
| UI-to-service submission | A preparation-only UI submission produced the expected command and input bundle. **No GPU generation or final GLB download occurred in this check.** |
| GPU matrix preparation | Prepared three 300-step text cases, a seven-variant teacup comparison, and one 300-step image-conditioned sailboat case. Preparation creates configurations, not generated results. |
| Fresh pf-pc69 runtime | New Python **3.10.19** environment, PyTorch **2.8.0/cu128**, CUDA **12.8.1**, Blender **3.0.1**, and pinned dependencies; `pip check` passed. |
| Native extensions | Installed FlashAttention 2.8.3 and built nvdiffrast, mip-splatting's Gaussian rasterizer, and vox2seq in the fresh environment. All seven native/runtime extension imports passed. The Debian/CUDA compatibility adjustment below was required. |
| Image staging and CPU preprocessing | Pinned TRELLIS image components, DINOv2 source and checkpoint SHA256, and U2Net's upstream checksum passed. DINOv2 loaded on CPU; the original image-model alias resolved offline; the included RGB sailboat reference processed to 518 by 518 pixels. No image-conditioned GPU generation occurred. |
| GitHub CI | All three jobs passed independently at commit `3d8a38197e173fb440371349aa1a38aaa478acb8`: Linux/Python 3.10 and macOS/Python 3.12 CPU/example checks, plus editor tests/production build. [Recorded run](https://github.com/joanlafuente/spaceflow/actions/runs/36827659676). |

No browser console errors were observed in the checked workflows. Three.js emits
a deprecation warning for its Clock helper. Visual inspection confirmed the
primitive scaffold rendered; fresh generated asset quality remains unchecked.

The CPU/editor requirements are separate from the GPU runtime. See the root
[README](../README.md) for the exact clone/install/check commands. The published
[Release checks workflow](../.github/workflows/ci.yml) targets Linux (Python 3.10),
macOS (Python 3.12), and the editor tests/build on Linux. The initial workflow
upload was blocked by OAuth scope and integration permissions; it was published
through the existing signed-in GitHub web editor. No authorization change was
required. [Current CI runs](https://github.com/joanlafuente/spaceflow/actions?query=branch%3ARELEASE)
record the independent runner results.

## GPU blocker and remaining acceptance

The authorized Student Cluster course tags `dslab` and `dslab_jobs` were checked
with bounded scheduler validation on 2026-10-01. Both were rejected as **course
tag not configured / invalid account or account-partition combination**. No GPU
verification job was submitted or executed. Account/course membership does not
by itself prove the scheduler account is ready.

The user subsequently authorized **pf-pc69.ethz.ch** for verification. SSH works;
the machine has two RTX 4090 GPUs (24 GB each). On the initial 2026-10-01 check,
both were running the user's Delimit3D training and each held about 21 GB. A
separate fresh checkout, environment, and caches have been prepared on its work
storage. The user chose to wait. The one-off verification queue checks every
30 seconds and starts after a GPU has no compute processes, at least 22 GiB free,
and 120 seconds of sustained idle time. It does not stop existing jobs.

The GPUs became available on 2026-10-01. The first actual matrix at commit
`3db81d04556279cf88e4c6358a3997be7ceca318` ran from 07:36 to 07:42 UTC. The
raw TRELLIS comparison produced a valid textured GLB, but SpaceFlow stopped at
PartField checkpoint loading. This attempt is preserved as failed evidence.
The recorded checkpoint contains `yacs.config.CfgNode`, which PyTorch 2.8's
weights-only loader does not allow by default. A scoped compatibility helper now
allows that single configuration class while Lightning loads PartField; it keeps
weights-only loading enabled and does not change model weights or generation
arithmetic. Direct loading of the recorded checkpoint passed with this scope;
full generation is being repeated. See the
[PyTorch serialization documentation](https://docs.pytorch.org/docs/2.8/notes/serialization.html#torch-load-with-weights-only-true).

### Debian 13 / CUDA 12.8 compatibility

pf-pc69 runs Debian 13, glibc 2.41, GCC 14.2.0, and NVIDIA driver 595.71.05.
CUDA 12.8's unmodified pi-math declarations conflict with the newer glibc
exception specifications. The first nvdiffrast build failed with that exact
error. The task-owned CUDA toolkit received the four-declaration `noexcept(true)`
adjustment described in the [NVIDIA compiler issue](https://forums.developer.nvidia.com/t/error-exception-specification-is-incompatible-for-cospi-sinpi-cospif-sinpif-with-glibc-2-41/323591).
The original header was backed up, and native builds then passed. No kernel
implementations or arithmetic were changed. This is an explicitly recorded
compatibility adjustment for this host, not a claim of upstream Debian 13 support.

The changed `crt/math_functions.h` hashes are:

- Original: `2f2189d1752d862e96122f985484c89e16bd03a485a01e1f114514f738a2ed4f`
- Adjusted: `024ff8406766f26573a1fe842cfb8e66f2ae35652fc0adc5e38b68a758276de9`

The task's private audit keeps the installer checksum, exact patch, before/after
hashes, compiler commands, installed-package list, cache marker, and environment
checks. For an unmodified toolkit, use a distribution supported by CUDA 12.8,
such as Ubuntu 22.04, following NVIDIA's installation guide.

The actual pipeline entrypoint check also exposed the missing `scikit-learn`
dependency. Its 1.7.2 pin and joblib/threadpoolctl pins were restored from the
recovered environment inventory. Preflight now imports that dependency and the
generation entrypoint as well as the native extensions.

The stable release requires:

1. Fresh CUDA 12.8 extension builds/imports: **completed on pf-pc69**, with the
   recorded host compatibility adjustment above.
2. Successful full 300-step teacup, chair, and sailboat SpaceFlow generations.
3. Output validation: successful status, nonempty finite geometry, a baked
   texture, and finite UV coordinates with the correct shape.
4. One image-conditioned generation and one retained baseline/comparison case.
5. One browser-initiated GPU generation, result inspection, and final GLB download.
6. Visual quality review, plus the commit, hardware, environment, model revisions,
   commands, and results recorded with the verification artifacts.
7. Verify the additional pinned image/DINOv2/U2Net cache with `doctor.py --gpu --image`
   and record the staging evidence alongside the generated result.

Once these checks pass, `RELEASE` can become the default branch, `v0.1.0` can be
tagged with a thin source archive, and the superseded promotion draft can be
retired. Changing the repository default requires repository administrator access.
Hugging Face deployment is deferred.

## Repeat the release matrix

With the CPU editor environment activated, from the repository root:

```bash
python tools/verify_release.py --prepare-only --output-dir runs/release-prepared
```

This prepares five cases without a GPU. The teacup uses its saved 300-step
setting. The saved chair/sailboat examples use shorter refinement runs; the
acceptance copies set 300 steps and record that change in `replay_provenance.json`.
The original examples and generation defaults are preserved. The image case uses
the included historical sailboat preview as its appearance reference; override it
with `--image-path /path/to/reference.png` if needed.

On the allocated GPU, with a clean committed checkout, the pinned runtime/models,
CUDA 12.8, and Blender configured:

```bash
python tools/cache_models.py --cache-dir "$HF_HOME" --include-image
python tools/doctor.py --gpu --image
python tools/verify_release.py --output-dir runs/release-gpu
```

For a fresh extension build under Slurm, follow [Research workflows](research.md)
and submit `tools/verify_release.sbatch` with a valid account/partition.
The suite writes `release-verification.json`, environment/hardware/package records,
per-case commands/logs, hashes, and geometry/texture validation reports. Text runs
use the offline pinned Hugging Face cache. Image conditioning also uses the
recorded DINOv2 source/weight and U2Net caches staged before the run.
The suite does not substitute for browser generation or visual review, and its
report keeps the stable-release milestone pending until those are complete.

## Recovery lineage and earlier evidence

The original `MINIMAL` working tree was based on
`1d845ecdbcc4409b4ac82ffa6de8ccdf710626c5`. All 213 original tracked files were
restored and checked against preserved source hashes. Fourteen uncommitted edits
were retained in recovery commit `96b9cea` before the portability changes. The
original dirty worktree, inputs, metadata, and historical outputs were backed up
independently. Older Git history and `MINIMAL` remain recovery references; the
root README uses a shallow `RELEASE` clone to avoid downloading that large history.

Earlier checks on 2026-09-30 installed the pinned generation dependencies in a
fresh Linux Python 3.10.13 environment, passed `pip check`, and staged the recorded
TRELLIS/CLIP models and hash-matching PartField checkpoint. CUDA extensions and
fresh generation were not executed. Those earlier dependency/cache checks do not
verify the current release's GPU behavior.

The image pipeline selection fix has CPU regression coverage. Its actual GPU
conditioning, model loading, and numerical output remain pending. Text-guided
calculation defaults, CLI flags, service endpoints, saved NPZ inputs, and model
configuration are preserved by this cleanup.

NPZ saves preserve image path metadata, not uploaded image bytes. Supply images
again when reopening. Historical generated outputs are separate data artifacts;
valid historical GLBs do not prove a fresh run succeeds.
