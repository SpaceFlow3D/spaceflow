# Changelog

## RELEASE candidate — 2026-10-01

- Start the public release branch from the recovered `MINIMAL` source, including
  its preserved uncommitted edits.
- Provide a researcher-oriented README and separate installation/research guides.
- Add a small CPU-only editor environment and an actionable environment checker.
- Use the active Python interpreter for the service instead of old environment paths.
- Report Blender executable/process failures explicitly while keeping the original
  mesh normalization and generation calculations.
- Add numeric input/replay validation, regression tests, and CPU/editor GitHub CI.
- Repair the editor dependency lock for clean installation across platforms.
- Update compatible editor dependencies to resolve the reported npm advisories.
- Select an image-capable pipeline for image appearance guidance; text-guided
  numerical behavior and parameters are unchanged. GPU verification is pending.
- Prepare a five-case GPU verification matrix with commit/environment reports.
- Defer blob URL cleanup so browsers can complete NPZ/PNG/GLB downloads.
- Retire the unused launcher for the old 20-file example layout and the duplicated
  standalone NPZ checker. The current replay/input validators replace them;
  baseline and comparison runners remain available.
- Keep the 83 input bundles and optional comparison tools; exclude large runtime data.

Fresh CUDA builds, GPU generation, image conditioning, and UI generation remain
release acceptance checks. No stable `v0.1.0` tag is issued before those pass.
