#!/usr/bin/env python3
"""Validate numeric primitive bundles and prepare portable replays without a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from replay_example import prepare

REPO_ROOT = Path(__file__).resolve().parents[1]
ARRAY_SHAPES = {
    "scales": (3,), "shapes": (2,), "translations": (3,), "rotations": (3, 3),
    "tapering": (2,), "bending": (6,), "control_levels": (),
}
REQUIRED_ARRAYS = {"scales", "shapes", "translations", "rotations"}


def validate_npz(path: Path) -> int:
    with np.load(path, allow_pickle=False) as arrays:
        missing = REQUIRED_ARRAYS - set(arrays.files)
        if missing:
            raise ValueError(f"Missing primitive arrays: {sorted(missing)}")
        count = arrays["scales"].shape[0]
        if path.name == "all.npz" and count == 0:
            raise ValueError("The all-primitives bundle is empty")
        for key, tail in ARRAY_SHAPES.items():
            if key not in arrays.files:
                continue
            value = arrays[key]
            if value.shape != (count, *tail):
                raise ValueError(f"{key}: expected {(count, *tail)}, found {value.shape}")
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError(f"{key}: expected finite numeric values")
        for key in ("scales", "shapes"):
            if np.any(arrays[key] <= 0):
                raise ValueError(f"{key}: values must be positive")
        return count


def validate_examples(examples_dir: Path) -> dict:
    cases = []
    input_count = 0
    with tempfile.TemporaryDirectory(prefix="spaceflow-replay-check-") as temporary:
        for config in sorted(examples_dir.glob("*/experiment_runner_config.json")):
            case = config.parent
            errors = []
            counts = {}
            for filename in ("all.npz", "high_control.npz", "low_control_bbox.npz"):
                try:
                    counts[filename] = validate_npz(case / "inputs" / filename)
                    input_count += 1
                except (OSError, ValueError, KeyError, IndexError) as exc:
                    errors.append(f"{filename}: {exc}")
            try:
                prepare(case.resolve(), Path(temporary) / case.name, [])
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"replay preparation: {exc}")
            cases.append({"name": case.name, "primitives": counts.get("all.npz"), "errors": errors})
    return {
        "passed": bool(cases) and all(not item["errors"] for item in cases),
        "case_count": len(cases), "input_npz_count": input_count, "cases": cases,
        "scope": "Numeric inputs and replay preparation only; generation was not executed.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-dir", type=Path, default=REPO_ROOT / "examples")
    parser.add_argument("--report", type=Path, help="Optional JSON report destination.")
    args = parser.parse_args(argv)
    report = validate_examples(args.examples_dir)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['case_count']} cases; {report['input_npz_count']} numeric input bundles")
    for item in report["cases"]:
        for error in item["errors"]:
            print(f"  {item['name']}: {error}")
    print(report["scope"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
