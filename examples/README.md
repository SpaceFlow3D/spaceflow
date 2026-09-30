# Recovered SpaceFlow inputs

These 83 small example bundles contain the superquadric primitives, global and
local prompts, scene metadata, and saved experiment parameters. Each example
has three NPZ inputs (`all`, `high_control`, and `low_control_bbox`), a manifest,
scene JSON, runner configuration, and run metadata.

Paths in the published metadata use a portable example-root placeholder. The
replay tool resolves them into a new output directory without changing prompts,
seeds, or optimization settings. The original metadata and generated SpaceFlow
and baseline outputs are preserved separately in the recovered data archive.

To run the SpaceFlow variant after installing the GPU runtime and model cache:

```bash
python tools/replay_example.py \
  --example-dir examples/blue_teacup_full_experiment \
  --output-dir runs/blue_teacup \
  --only 01_local_tau3_tau10_polyak0p18
python tools/check_replay_outputs.py runs/blue_teacup
```

Use a new output directory for each replay. Add `--prepare-only` to inspect the
relocated configuration without running the GPU pipeline. Omitting `--only`
runs all saved variants, including baselines, and needs more GPU time.

The bundles contain inputs and configurations, not generated meshes or model
weights. A fresh end-to-end GPU verification remains pending; valid historical
outputs alone do not establish a fresh installation's reproducibility.
