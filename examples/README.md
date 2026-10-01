# SpaceFlow example inputs

These 83 small example bundles contain the superquadric primitives, global and
local prompts, scene metadata, and saved experiment parameters. Each example
has three NPZ inputs (`all`, `high_control`, and `low_control_bbox`), a manifest,
shape prompt, runner configuration, and run metadata.

Paths in the published metadata use a portable example-root placeholder. The
replay tool resolves them into a new output directory without changing prompts,
seeds, or optimization settings. The original metadata and generated SpaceFlow
and baseline outputs are preserved separately in the recovered data archive.
The all-primitives NPZ also contains editor metadata so that opening it restores
the saved part names and text conditions. Adding this metadata preserves the
original numeric primitive array bytes.

To run the SpaceFlow variant after installing the GPU runtime and model cache:

```bash
python tools/replay_example.py \
  --example-dir examples/25_20260606T002730Z_toy_elephant_full_experiment \
  --output-dir runs/toy_elephant \
  --only 01_spaceflow_local_texture_routing
python tools/check_replay_outputs.py runs/toy_elephant
```

Use a new output directory for each replay. Add `--prepare-only` to inspect the
relocated configuration without running the GPU pipeline. Omitting `--only`
runs all saved variants, including baselines, and needs more GPU time.

The bundles contain inputs and configurations, not generated meshes or model
weights. See the [main README](../README.md) for installation and the
[verification record](../docs/verification.md) for the cases generated freshly.
