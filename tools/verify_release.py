#!/usr/bin/env python3
"""Prepare or run the fresh GPU release matrix; browser verification is separate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from replay_example import prepare

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACEFLOW_VARIANT = "01_local_tau3_tau10_polyak0p18"
TEXT_CASES = (
    ("text-teacup", "blue_teacup_full_experiment"),
    ("text-chair", "01_a_chair_full_experiment"),
    ("text-sailboat", "13_sailboat_full_experiment"),
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def require_full_refinement(config_path: Path) -> None:
    """The acceptance matrix requires 300 steps, independent of the saved run."""
    config = json.loads(config_path.read_text())
    changed = False
    for variant in config["variants"]:
        argv = variant.get("argv", [])
        if "--texture_optim_steps" in argv:
            index = argv.index("--texture_optim_steps") + 1
            changed |= argv[index] != "300"
            argv[index] = "300"
            variant["texture_optim_steps"] = 300
    config["texture_optim_steps"] = 300
    write_json(config_path, config)
    if changed:
        provenance_path = config_path.parent / "replay_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["parameters_changed"].append("texture_optim_steps set to 300 for release acceptance")
        write_json(provenance_path, provenance)


def prepare_matrix(destination: Path, image_path: Path) -> list[dict]:
    """Prepare 300-step acceptance cases without altering the saved examples."""
    if destination.exists():
        raise FileExistsError(f"Choose a new verification directory: {destination}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Conditioning image not found: {image_path}")
    destination.mkdir(parents=True)
    cases = []
    for name, example in TEXT_CASES:
        config = prepare(REPO_ROOT / "examples" / example, destination / name, [SPACEFLOW_VARIANT])
        require_full_refinement(config)
        cases.append({"name": name, "config": str(config), "status": "prepared"})
    comparison = prepare(
        REPO_ROOT / "examples/blue_teacup_full_experiment", destination / "comparisons-teacup", [],
    )
    cases.append({"name": "comparisons-teacup", "config": str(comparison), "status": "prepared"})

    config_path = prepare(
        REPO_ROOT / "examples/13_sailboat_full_experiment", destination / "image-sailboat", [SPACEFLOW_VARIANT],
    )
    require_full_refinement(config_path)
    target_image = config_path.parent / "inputs" / ("appearance" + image_path.suffix.lower())
    shutil.copy2(image_path, target_image)
    config = json.loads(config_path.read_text())
    variant = config["variants"][0]
    argv = variant["argv"]
    for flag in ("--appearance_text", "--local_text_prompts"):
        if flag in argv:
            index = argv.index(flag)
            del argv[index:index + 2]
    argv.extend(["--appearance_image", str(target_image)])
    write_json(config_path, config)
    provenance_path = config_path.parent / "replay_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["parameters_changed"].extend(["global appearance image", "local text overrides removed"])
    provenance["conditioning_image_sha256"] = file_hash(target_image)
    write_json(provenance_path, provenance)
    cases.append({"name": "image-sailboat", "config": str(config_path), "status": "prepared"})
    return cases


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    except OSError as exc:
        return str(exc)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def run_recorded(command: list[str], log_path: Path, environment: dict) -> int:
    print("Running:", " ".join(command), flush=True)
    with log_path.open("w") as stream:
        result = subprocess.run(command, cwd=REPO_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory; previous results are preserved.")
    parser.add_argument("--prepare-only", action="store_true", help="Write configs without executing GPU generation.")
    parser.add_argument("--image-path", type=Path, default=REPO_ROOT / "docs/media/sailboat_spin_poster.png",
                        help="Sailboat appearance reference; defaults to the included historical preview.")
    args = parser.parse_args(argv)
    dirty = command_output(["git", "status", "--porcelain", "--untracked-files=normal"])
    if dirty and not args.prepare_only:
        parser.error("Commit the source before fresh GPU verification so the report identifies reproducible code.")
    destination = args.output_dir.expanduser().resolve()
    cases = prepare_matrix(destination, args.image_path.expanduser().resolve())
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "release_commit": command_output(["git", "rev-parse", "HEAD"]),
        "source_dirty": bool(dirty), "python": sys.version, "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model_revisions": json.loads((REPO_ROOT / "requirements/model-revisions.json").read_text()),
        "cases": cases, "generation_executed": False, "status": "prepared",
        "stable_release_ready": False,
        "remaining_acceptance": ["Browser editor-to-backend generation and GLB download",
                                 "Visual quality review"],
    }
    report_path = destination / "release-verification.json"
    write_json(report_path, report)
    if args.prepare_only:
        print(f"Prepared {len(cases)} cases; no generation executed. Report: {report_path}")
        return 0

    environment = os.environ.copy()
    preflight = [sys.executable, str(REPO_ROOT / "tools/doctor.py"), "--gpu", "--image", "--json"]
    report["preflight_command"] = preflight
    preflight_status = run_recorded(preflight, destination / "environment-check.json", environment)
    report["gpu_hardware"] = command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"])
    report["installed_packages"] = command_output([sys.executable, "-m", "pip", "freeze"])
    cache_marker = Path(environment.get("HF_HOME", str(REPO_ROOT / "spaceflow_runtime/huggingface"))) / "spaceflow-models.json"
    if cache_marker.is_file():
        report["staged_models"] = json.loads(cache_marker.read_text())
    if preflight_status:
        report["status"] = "preflight_failed"
        write_json(report_path, report)
        print(f"Environment check failed; inspect {destination / 'environment-check.json'}", file=sys.stderr)
        return preflight_status

    for case in cases:
        run_dir = Path(case["config"]).parent
        # Both HF pipelines use explicitly staged revisions. DINOv2 source and
        # its checkpoint are pinned and staged in the selected Torch cache.
        case_environment = environment.copy()
        case_environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        command = [sys.executable, str(REPO_ROOT / "sq_ui/scripts/run_spaceflow_experiment.py"), "--config", case["config"]]
        case["command"] = command
        case["status"] = "running"
        report["generation_executed"] = True
        write_json(report_path, report)
        exit_code = run_recorded(command, run_dir / "verification-run.log", case_environment)
        check = [sys.executable, str(REPO_ROOT / "tools/check_replay_outputs.py"), str(run_dir)]
        validation_code = run_recorded(check, run_dir / "verification-outputs.log", case_environment)
        case.update(exit_code=exit_code, validation_exit_code=validation_code,
                    validation_command=check, status="passed" if exit_code == validation_code == 0 else "failed")
        write_json(report_path, report)
    report["status"] = "passed" if all(case["status"] == "passed" for case in cases) else "failed"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(report_path, report)
    print(f"GPU matrix {report['status']}. Browser generation and visual review are separate. Report: {report_path}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
