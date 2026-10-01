# Replaying experiments and comparisons

The default quickstart selects one SpaceFlow variant. Research tools retain the
recorded baseline configurations without adding them to every normal generation.

## Examples

Each of the 83 directories in `examples/` includes:

- `inputs/all.npz`, `high_control.npz`, and `low_control_bbox.npz`;
- editor metadata, shape/global/local prompts, and primitive names;
- a runner configuration and historical run metadata.

Useful starters: `blue_teacup_full_experiment`, `01_a_chair_full_experiment`, and
`13_sailboat_full_experiment`. Saved generated meshes are separate data artifacts;
the source bundles do not contain old outputs.

## Prepare without a GPU

```bash
python tools/replay_example.py \
  --example-dir examples/blue_teacup_full_experiment \
  --output-dir runs/teacup-prepared \
  --only 01_local_tau3_tau10_polyak0p18 --prepare-only
python tools/validate_examples.py
```

The preparation step relocates paths into a new run directory and preserves the
recorded prompts, parameters, and variant dependencies. It creates no generated
result. `replay_provenance.json` records the source configuration hash.

## Run recorded comparisons

With the GPU runtime activated and model/Blender variables set:

```bash
python tools/replay_example.py \
  --example-dir examples/blue_teacup_full_experiment \
  --output-dir runs/teacup-comparisons
python tools/check_replay_outputs.py runs/teacup-comparisons
```

Omitting `--only` runs every variant in that example's configuration. Typical
bundles compare local/global geometric control, raw TRELLIS, and fixed-structure
appearance baselines. Selecting a dependent variant also includes its source.
These comparisons take more GPU time than one SpaceFlow run.

The retained in-process runner is `sq_ui/scripts/run_spaceflow_experiment.py`;
`trellis_texture_variants.py` contains the associated baseline implementations.
All seven variants in the retained teacup comparison passed in the fresh release
environment. See [verification](verification.md) for the checked configurations
and output hashes. Other baseline cases have not all been generated freshly.

## Render comparison figures on CPU

With the CPU editor environment activated, install the optional plotting dependency.
The full GPU environment already includes it. Run from the repository root:

```bash
python -m pip install -r requirements/visualization.txt
python sq_ui/scripts/render_spaceflow_experiment_comparison.py \
  runs/teacup-comparisons --output-name variant_comparison.png
```

The renderer uses a shared view for the recorded variants. For a single result
plus its primitive controls, add `--single`. `--azim` and `--elev` select the view.
Metrics in `metrics/partfield_baseline_distance.py` and the latent-control figure
script in `docs/` are optional analysis tools; inspect their `--help` output for
inputs rather than treating them as generation requirements.

## Slurm verification

Stage dependencies/models and load your institution's Python, compiler, CUDA
12.8, and Blender environment. Submit from the repository root:

```bash
export HF_HOME="$PWD/spaceflow_runtime/huggingface"
export SPACEFLOW_BLENDER_PATH=/path/to/installed/blender
export SPACEFLOW_BUILD_EXTENSIONS=1
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION tools/verify_spaceflow.sbatch
```

This single-example job requests one GPU, four CPUs, 64 GB host memory, and four
hours. It replays the teacup with 300 refinement steps and validates the result.
The resource request is a verification allocation, not a measured minimum.

For the complete release matrix, use `tools/verify_release.sbatch` instead.
It requests 12 hours with the same GPU/CPU/memory allocation. The recorded RTX 4090
suite completed in about 18 minutes; Slurm hardware/runtime may differ. It prepares
teacup/chair/sailboat at 300 refinement steps, a
full teacup comparison, and sailboat image conditioning. The default image is the
included historical sailboat preview; set `SPACEFLOW_VERIFY_IMAGE` to another
reference if desired. Stage the additional image models with `cache_models.py --include-image`
while network access is available. Both text and image generation then use the
pinned offline caches. Changes to the saved example
step counts are recorded only in the new verification directory's provenance.
Its preparation-only mode is described in the [verification record](verification.md).

For a UI service on a login node, set a valid account and partition before launch:

```bash
export SQ_SPACEFLOW_SLURM_ACCOUNT=YOUR_ACCOUNT
export SQ_SPACEFLOW_SLURM_PARTITION=YOUR_GPU_PARTITION
bash run.sh
```

The service uses Slurm when available and launches locally on an allocated node.
Optional overrides include `SQ_SPACEFLOW_SLURM_GPUS`, `SQ_SPACEFLOW_SLURM_TIME`,
`SQ_SPACEFLOW_SLURM_CONSTRAINT`, and `SQ_SPACEFLOW_SLURM_EXTRA_ARGS`.
A scheduler rejection is an allocation failure; it does not count as a started
or completed SpaceFlow generation.
