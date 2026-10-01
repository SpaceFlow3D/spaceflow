# SpaceFlow documentation

Start with the [main README](../README.md) to install SpaceFlow, generate an
example, or use the interactive editor. This directory contains the detailed
configuration, research workflows, and release records.

## Guides

| Guide | Contents |
| --- | --- |
| [Installation and troubleshooting](installation.md) | Python/CUDA environments, model caches, Blender, platform notes, and service configuration. |
| [Editor and service](../sq_ui/README.md) | Import/export, saved scenes, text/image conditions, backend settings, and frontend development. |
| [Example inputs](../examples/README.md) | The 83 input bundles and recorded replay settings. |
| [Research workflows](research.md) | Slurm, experiment replay, retained baselines, comparison rendering, and metrics. |
| [Release verification](verification.md) | Completed CPU/UI/GPU checks, hardware, commands, quality limits, and recovery lineage. |
| [Verification results](verification-results.json) | Recorded model revisions and accepted output hashes. |
| [Third-party notices](../THIRD_PARTY_NOTICES.md) | Upstream attribution and license restrictions. |

## Development checks

Run from the repository root with the CPU editor environment activated:

```bash
python -m unittest discover -s tests -v
python tools/validate_examples.py
(cd sq_ui/app && npm test && npm run build)
```

The [GitHub workflow](../.github/workflows/ci.yml) runs CPU/example checks on
Linux and macOS and builds the editor. GPU verification is a separate workflow
on a machine with the full runtime; see the verification guide above.
