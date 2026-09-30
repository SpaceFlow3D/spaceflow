#!/usr/bin/env python3
"""HTTP service for saving SQ editor assets and launching SpaceFlow runs."""

from __future__ import annotations

import http.server
import json
import mimetypes
import os
import re
import signal
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="'cgi' is deprecated.*", category=DeprecationWarning)
    import cgi


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
HOST = os.environ.get("SQ_SPACEFLOW_HOST", "0.0.0.0").strip() or "0.0.0.0"
PORT = int(os.environ.get("SQ_SPACEFLOW_PORT", "11438"))
CORS_ORIGIN = os.environ.get("SQ_SPACEFLOW_CORS_ORIGIN", "*").strip()
PUBLIC_DEMO = os.environ.get("SQ_SPACEFLOW_PUBLIC_DEMO", "").strip().lower() in {"1", "true", "yes", "on"}
USER = os.environ.get("USER", "user")
DEFAULT_STORAGE_ROOT = REPO_ROOT / "spaceflow_runtime"
TEAM_STORAGE_ROOT = Path(
    os.environ.get("SQ_SPACEFLOW_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT))
).expanduser()
SAVE_ROOT = Path(
    os.environ.get("SQ_SPACEFLOW_ASSET_ROOT", str(TEAM_STORAGE_ROOT / "sq_ui_assets"))
).expanduser()
RUN_ROOT = Path(
    os.environ.get("SQ_SPACEFLOW_RUN_ROOT", str(TEAM_STORAGE_ROOT / "sq_ui_runs"))
).expanduser()
MAX_ACTIVE_RUNS = max(0, int(os.environ.get("SQ_SPACEFLOW_MAX_ACTIVE_RUNS", "0") or "0"))
RETENTION_HOURS = max(
    0.0,
    float(os.environ.get("SQ_SPACEFLOW_RETENTION_HOURS", "48" if PUBLIC_DEMO else "0") or "0"),
)
MAX_STORAGE_GB = max(
    0.0,
    float(os.environ.get("SQ_SPACEFLOW_MAX_STORAGE_GB", "40" if PUBLIC_DEMO else "0") or "0"),
)
RUN_TIMEOUT = int(os.environ.get("SQ_SPACEFLOW_TIMEOUT_SEC", "7200"))
STOP_GRACE_SEC = float(os.environ.get("SQ_SPACEFLOW_STOP_GRACE_SEC", "8"))
FORCE_LOCAL = os.environ.get("SQ_SPACEFLOW_FORCE_LOCAL", "").strip() == "1"
FULL_PIPELINE = os.environ.get("SQ_SPACEFLOW_FULL_PIPELINE", "1").strip().lower() not in {"0", "false", "no", "off"}
PARTITION = os.environ.get("SQ_SPACEFLOW_SLURM_PARTITION", "").strip()
ACCOUNT = os.environ.get("SQ_SPACEFLOW_SLURM_ACCOUNT", "").strip()
GPUS = os.environ.get("SQ_SPACEFLOW_SLURM_GPUS", "1").strip()
CONSTRAINT = os.environ.get("SQ_SPACEFLOW_SLURM_CONSTRAINT", "").strip()
EXCLUDE_NODES = os.environ.get("SQ_SPACEFLOW_SLURM_EXCLUDE", "").strip()
TIME_LIMIT = os.environ.get("SQ_SPACEFLOW_SLURM_TIME", "02:00:00")
EXTRA_ARGS = os.environ.get("SQ_SPACEFLOW_SLURM_EXTRA_ARGS", "").strip()
GPU_PREFLIGHT_MODE = os.environ.get("SQ_SPACEFLOW_GPU_PREFLIGHT", "fast").strip().lower() or "fast"
RUN_SCRIPT = Path(os.environ.get("SQ_SPACEFLOW_RUN_SCRIPT", str(REPO_ROOT / "run_local_tau.py"))).expanduser()
EXPERIMENT_RUNNER_SCRIPT = Path(
    os.environ.get(
        "SQ_SPACEFLOW_EXPERIMENT_RUN_SCRIPT",
        str(REPO_ROOT / "sq_ui" / "scripts" / "run_spaceflow_experiment.py"),
    )
).expanduser()
OFFLINE_CACHE_MODE = os.environ.get("SQ_SPACEFLOW_OFFLINE_CACHE", "auto").strip().lower() or "auto"
HF_OFFLINE_ENV_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
CACHE_ROOT = Path(
    os.environ.get("SQ_SPACEFLOW_CACHE_ROOT", str(TEAM_STORAGE_ROOT / "huggingface"))
).expanduser()
XDG_CACHE_ROOT = Path(
    os.environ.get("SQ_SPACEFLOW_XDG_CACHE_ROOT", str(TEAM_STORAGE_ROOT / "xdg_cache"))
).expanduser()


def _default_python_bin() -> str:
    env_override = os.environ.get("SQ_SPACEFLOW_PYTHON", "").strip()
    if env_override:
        return env_override
    candidates = [
        REPO_ROOT / "envs" / "guideflow3d" / "bin" / "python",
        REPO_ROOT.parent / "guideflow3d" / "envs" / "guideflow3d" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


PYTHON_BIN = _default_python_bin()
RUNS: dict[str, subprocess.Popen[bytes]] = {}
RUN_LOCK = threading.Lock()
ACTIVE_RUN_STATUSES = {"running", "cancelling"}
CLEANUP_RUN_STATUSES = {"succeeded", "failed", "cancelled", "dry_run"}
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
KNOWN_OUTPUTS = [
    "input_superquadrics_all.npz",
    "input_superquadrics_high_control.npz",
    "input_superquadrics_low_control_bbox.npz",
    "input_superquadrics_colored.glb",
    "routing_warnings.json",
    "variant_comparison_lower_camera.png",
    "structure_variant_comparison_lower_camera.png",
    "texture_variant_comparison_lower_camera.png",
    "out_sim.glb",
    "out_sim_geometry.glb",
    "out_gaussian_sim.mp4",
    "sample.glb",
    "struct_mesh_zup.glb",
    "struct_mesh.glb",
    "denoising_evolution.mp4",
    "spatial_control_mesh.ply",
    "high_control_spatial_control_mesh.ply",
    "low_control_superquadric_mask.ply",
    "struct_renders/000.png",
    "struct_renders/mesh.ply",
    "voxels/struct_voxels.ply",
    "app_image.png",
]
PUBLIC_DEMO_FINAL_OUTPUT = os.environ.get("SQ_SPACEFLOW_PUBLIC_FINAL_OUTPUT", "out_sim.glb").strip() or "out_sim.glb"


def _send_file(res: http.server.BaseHTTPRequestHandler, file_path: Path) -> None:
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    res.send_response(200)
    res._cors()
    res.send_header("Content-Type", content_type)
    res.send_header("Content-Length", str(file_path.stat().st_size))
    res.send_header("Content-Disposition", f'inline; filename="{file_path.name}"')
    res.end_headers()
    with file_path.open("rb") as fh:
        shutil.copyfileobj(fh, res.wfile)


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    safe = re.sub(r"_+", "_", safe)
    return safe or "superquadrics"


def _utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _local_log_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _split_args(s: str) -> list[str]:
    return [tok for tok in s.split() if tok]


def _drop_gres_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--gres="):
            i += 1
            continue
        if t == "--gres":
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def _current_hostnames() -> set[str]:
    names = {
        socket.gethostname(),
        socket.getfqdn(),
    }
    return {name for name in names if name} | {name.split(".")[0] for name in names if name}


def _allocated_hostnames() -> set[str]:
    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST") or ""
    if not nodelist:
        return set()
    if shutil.which("scontrol"):
        try:
            result = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            hosts = {line.strip() for line in result.stdout.splitlines() if line.strip()}
            return hosts | {host.split(".")[0] for host in hosts}
        except subprocess.SubprocessError:
            pass
    if "[" in nodelist or "," in nodelist:
        return {nodelist}
    return {nodelist, nodelist.split(".")[0]}


def _running_on_allocated_node() -> bool:
    allocated = _allocated_hostnames()
    if not allocated:
        return bool(os.environ.get("SLURM_JOB_ID"))
    return bool(_current_hostnames() & allocated)


def _should_use_srun() -> bool:
    if FORCE_LOCAL and _running_on_allocated_node():
        return False
    if os.environ.get("SLURM_JOB_ID"):
        return not _running_on_allocated_node() and shutil.which("srun") is not None
    return shutil.which("srun") is not None


def _gpu_preflight_mode() -> str:
    if GPU_PREFLIGHT_MODE in {"fast", "full", "skip"}:
        return GPU_PREFLIGHT_MODE
    return "fast"


def _srun_bash_wrapper(safe_cmd: str, python_bin: str) -> str:
    mode = _gpu_preflight_mode()
    if mode == "skip":
        preflight = """
    HOSTNAME=$(hostname -f 2>/dev/null || hostname)
    sf_log "Compute host: $HOSTNAME"
    sf_log "GPU preflight: skipped"
    export SPCONV_ALGO="${SPCONV_ALGO:-native}"
    sf_log "SPCONV_ALGO: $SPCONV_ALGO"
"""
    elif mode == "full":
        preflight = f"""
    HOSTNAME=$(hostname -f 2>/dev/null || hostname)
    sf_log "Compute host: $HOSTNAME"
    REQ_CUDA=$({python_bin} -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "Unknown")
    if ! NVIDIA_SMI_OUTPUT=$(nvidia-smi 2>&1); then
        echo "$NVIDIA_SMI_OUTPUT"
        sf_log "ERROR: nvidia-smi failed on $HOSTNAME; the GPU driver on this node is unhealthy."
        sf_log "Hint: restart the service with SQ_SPACEFLOW_SLURM_EXCLUDE=$HOSTNAME, or ask cluster support to fix the node."
        exit 88
    fi
    NODE_CUDA=$(printf "%s\\n" "$NVIDIA_SMI_OUTPUT" | sed -n -E 's/.*CUDA Version: ([0-9]+\\.[0-9]+).*/\\1/p' | head -n 1)
    if [ -z "$NODE_CUDA" ]; then
        NODE_CUDA="Unknown"
    fi
    sf_log "PyTorch requires CUDA: $REQ_CUDA"
    sf_log "Node supports max CUDA: $NODE_CUDA"
    source /etc/profile.d/modules.sh 2>/dev/null || true
    if command -v module &> /dev/null; then
        sf_log "Attempting to load module: cuda/$REQ_CUDA..."
        module load cuda/$REQ_CUDA 2>/dev/null || sf_log "Warning: module load cuda/$REQ_CUDA failed. Proceeding anyway..."
    fi
    export SPCONV_ALGO="${{SPCONV_ALGO:-native}}"
    sf_log "SPCONV_ALGO: $SPCONV_ALGO"
    if ! {python_bin} -c "import sys, torch; available = torch.cuda.is_available(); print('[sq-spaceflow] PyTorch CUDA available:', available); print('[sq-spaceflow] PyTorch CUDA devices:', torch.cuda.device_count()); sys.exit(0 if available else 88)"; then
        sf_log "ERROR: PyTorch cannot use CUDA on $HOSTNAME."
        exit 88
    fi
"""
    else:
        preflight = """
    HOSTNAME=$(hostname -f 2>/dev/null || hostname)
    sf_log "Compute host: $HOSTNAME"
    sf_log "GPU preflight: fast"
    if ! NVIDIA_SMI_OUTPUT=$(nvidia-smi -L 2>&1); then
        echo "$NVIDIA_SMI_OUTPUT"
        sf_log "ERROR: nvidia-smi failed on $HOSTNAME; the GPU driver on this node is unhealthy."
        sf_log "Hint: set SQ_SPACEFLOW_GPU_PREFLIGHT=full for deeper diagnostics or exclude this node."
        exit 88
    fi
    printf "%s\\n" "$NVIDIA_SMI_OUTPUT" | sed -n '1,4p'
    export SPCONV_ALGO="${SPCONV_ALGO:-native}"
    sf_log "SPCONV_ALGO: $SPCONV_ALGO"
"""

    return f"""
    set -o pipefail
    sf_log() {{ printf '%s [sq-spaceflow] %s\\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }}
    echo "========================================"
{preflight.rstrip()}
    echo "========================================"
    sf_log "Starting run command."
    exec {safe_cmd}
    """


def _wrap_with_srun(cmd: list[str]) -> list[str]:
    if not ACCOUNT:
        raise ValueError("Set SQ_SPACEFLOW_SLURM_ACCOUNT for jobs launched outside an allocation.")
    srun_cmd = [
        "srun",
        f"--account={ACCOUNT}",
        f"--time={TIME_LIMIT}",
        "--job-name=sq_spaceflow",
        "--ntasks=1",
        "--export=ALL",
        f"--gpus={GPUS or '1'}",
    ]
    if PARTITION:
        srun_cmd.append(f"--partition={PARTITION}")
    if CONSTRAINT:
        srun_cmd.append(f"--constraint={CONSTRAINT}")
    if EXCLUDE_NODES:
        srun_cmd.append(f"--exclude={EXCLUDE_NODES}")
        
    srun_cmd.extend(_drop_gres_tokens(_split_args(EXTRA_ARGS)))
    
    safe_cmd = shlex.join(cmd)
    python_bin = shlex.quote(cmd[0])
    srun_cmd.extend(["bash", "-c", _srun_bash_wrapper(safe_cmd, python_bin)])
    return srun_cmd


def _wrap_shell_with_srun(script_path: Path, job_name: str = "sq_spaceflow_exp") -> list[str]:
    if not ACCOUNT:
        raise ValueError("Set SQ_SPACEFLOW_SLURM_ACCOUNT for jobs launched outside an allocation.")
    srun_cmd = [
        "srun",
        f"--account={ACCOUNT}",
        f"--time={TIME_LIMIT}",
        f"--job-name={job_name}",
        "--ntasks=1",
        "--export=ALL",
        f"--gpus={GPUS or '1'}",
    ]
    if PARTITION:
        srun_cmd.append(f"--partition={PARTITION}")
    if CONSTRAINT:
        srun_cmd.append(f"--constraint={CONSTRAINT}")
    if EXCLUDE_NODES:
        srun_cmd.append(f"--exclude={EXCLUDE_NODES}")
    srun_cmd.extend(_drop_gres_tokens(_split_args(EXTRA_ARGS)))

    python_bin = shlex.quote(PYTHON_BIN)
    safe_cmd = shlex.join(["bash", str(script_path)])
    srun_cmd.extend(["bash", "-c", _srun_bash_wrapper(safe_cmd, python_bin)])
    return srun_cmd


def _num_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _local_tau_variant_name(index: int, low_tau: float, high_tau: float, polyak_tau: float) -> str:
    return (
        f"{index:02d}_local_tau{_num_tag(low_tau)}"
        f"_tau{_num_tag(high_tau)}_polyak{_num_tag(polyak_tau)}"
    )


def _global_tau_variant_name(index: int, tau: float, polyak_tau: float = 0.0) -> str:
    return f"{index:02d}_global_tau{_num_tag(tau)}_polyak{_num_tag(polyak_tau)}"


def _spaceflow_cmd(
    asset_paths: dict[str, object],
    output_dir: Path,
    *,
    text_prompt: str,
    appearance_args: list[str],
    local_texture_args: list[str],
    low_tau: float,
    high_tau: float | None,
    polyak_tau: float,
    n_repaint_steps: int,
    texture_optim_steps: int,
    convert_yup_to_zup: bool,
) -> list[str]:
    cmd = [
        PYTHON_BIN,
        str(RUN_SCRIPT),
        "--guidance_mode",
        "similarity",
        "--output_dir",
        str(output_dir),
        "--shape_superquadric_path",
        str(asset_paths["all"]),
        "--shape_tau",
        str(low_tau),
        "--polyak_update_tau",
        str(polyak_tau),
        "--n_repaint_steps",
        str(n_repaint_steps),
        "--texture_optim_steps",
        str(texture_optim_steps),
        "--text_prompt",
        text_prompt,
    ]
    if high_tau is not None:
        if high_tau < low_tau:
            raise ValueError("High tau must be greater than or equal to low tau")
        cmd.extend([
            "--shape_superquadric_high_control_path",
            str(asset_paths["high_control"]),
            "--shape_tau_high_control",
            str(high_tau),
            "--low_control_superquadric_mask_path",
            str(asset_paths["low_control_bbox"]),
            "--local_tau_mode",
            "low_control_mask",
        ])
    if convert_yup_to_zup:
        cmd.append("--convert_yup_to_zup")
    if FULL_PIPELINE:
        cmd.append("--full_pipeline")
    cmd.extend(appearance_args)
    cmd.extend(local_texture_args)
    return cmd


def _write_experiment_script(script_path: Path, variants: list[dict[str, object]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "sf_log() { printf '%s [experiment] %s\\n' \"$(date '+%Y-%m-%d %H:%M:%S')\" \"$*\"; }",
        "experiment_status=0",
        "experiment_start=$SECONDS",
        'sf_log "starting SpaceFlow experiment"',
    ]
    for index, variant in enumerate(variants, start=1):
        name = str(variant["name"])
        cmd = variant["command"]
        output_dir = Path(str(variant["output_dir"]))
        log_path = output_dir / "spaceflow.log"
        assert isinstance(cmd, list)
        lines.extend([
            "variant_start=$SECONDS",
            f'sf_log "variant {index}/{len(variants)} started: {name}"',
            f"mkdir -p {shlex.quote(str(output_dir))}",
            f"if {shlex.join([str(part) for part in cmd])} 2>&1 | tee {shlex.quote(str(log_path))}; then",
            f"  echo succeeded > {shlex.quote(str(output_dir / 'status.txt'))}",
            "  variant_elapsed=$((SECONDS - variant_start))",
            f'  sf_log "variant {index}/{len(variants)} succeeded: {name} in ${{variant_elapsed}}s"',
            "else",
            "  code=$?",
            f"  echo failed:$code > {shlex.quote(str(output_dir / 'status.txt'))}",
            "  variant_elapsed=$((SECONDS - variant_start))",
            f'  sf_log "variant {index}/{len(variants)} failed with code $code: {name} after ${{variant_elapsed}}s"',
            "  if [ \"$experiment_status\" -eq 0 ]; then experiment_status=$code; fi",
            "fi",
        ])
    lines.extend([
        'sf_log "completed SpaceFlow experiment in $((SECONDS - experiment_start))s"',
        "exit \"$experiment_status\"",
    ])
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)


def _write_command_script(script_path: Path, cmd: list[str]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        f"exec {shlex.join([str(part) for part in cmd])}",
    ]
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)


def _write_single_run_script(script_path: Path, cmd: list[str], run_dir: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "sf_log() { printf '%s [sq-spaceflow] %s\\n' \"$(date '+%Y-%m-%d %H:%M:%S')\" \"$*\"; }",
        "spaceflow_start=$SECONDS",
        shlex.join([str(part) for part in cmd]),
        "spaceflow_status=$?",
        'sf_log "SpaceFlow command exited with status $spaceflow_status in $((SECONDS - spaceflow_start))s"',
        'exit "$spaceflow_status"',
    ]
    if not PUBLIC_DEMO:
        render_cmd = [
            PYTHON_BIN,
            str(REPO_ROOT / "sq_ui" / "scripts" / "render_spaceflow_experiment_comparison.py"),
            "--single",
            str(run_dir),
            "--output-name",
            "output/variant_comparison_lower_camera.png",
            "--azim",
            "0.0",
            "--elev",
            "55.0",
        ]
        lines[-1:-1] = [
            'if [ "$spaceflow_status" -eq 0 ]; then',
            "  render_start=$SECONDS",
            '  sf_log "Rendering tau-by-parts summary figure..."',
            f"  if {shlex.join([str(part) for part in render_cmd])}; then",
            '    sf_log "Rendered tau-by-parts summary figure in $((SECONDS - render_start))s."',
            "  else",
            '    sf_log "Warning: tau-by-parts summary render failed after $((SECONDS - render_start))s."',
            "  fi",
            "fi",
        ]
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)


def _write_experiment_runner_config(
    config_path: Path,
    variants: list[dict[str, object]],
    *,
    experiment_type: str | None = None,
    texture_flattened_prompt: str | None = None,
    texture_optim_steps: int | None = None,
) -> None:
    runner_variants = []
    for variant in variants:
        runner_variants.append({
            key: value
            for key, value in variant.items()
            if key != "command"
        })
    payload: dict[str, object] = {
        "spaceflow_config": "config/default.yaml",
        "run_dir": str(config_path.parent),
        "variants": runner_variants,
        "comparison": {
            "enabled": not PUBLIC_DEMO,
            "output_name": "output/variant_comparison_lower_camera.png",
            "azim": 0.0,
            "elev": 55.0,
        },
    }
    if experiment_type:
        payload["experiment_type"] = experiment_type
    if texture_flattened_prompt:
        payload["texture_flattened_prompt"] = texture_flattened_prompt
    if texture_optim_steps is not None:
        payload["texture_optim_steps"] = texture_optim_steps
    config_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _assert_experiment_variant_layout(experiment_type: str, variants: list[dict[str, object]]) -> None:
    actual = [str(variant.get("name") or "") for variant in variants]
    if experiment_type == "full":
        expected_name_prefixes = ["01_local_tau", "02_global_tau", "03_global_tau"]
        actual_modes = [str(variant.get("mode") or "") for variant in variants]
        expected_modes = [
            "local_tau",
            "global_tau",
            "global_tau",
            "spaceflow_local_texture_routing",
            "trellis_raw_text",
            "fixed_structure_appearance_fm",
            "fixed_structure_guideflow_appearance_fm",
        ]
        expected_tail_names = [
            "01_spaceflow_local_texture_routing",
            "02_trellis_raw_flat_prompt",
            "03_fixed_structure_appearance_fm",
            "04_fixed_structure_guideflow_appearance_fm",
        ]
        if (
            actual_modes != expected_modes
            or len(actual) != len(expected_modes)
            or any(not name.startswith(prefix) for name, prefix in zip(actual[:3], expected_name_prefixes))
            or actual[3:] != expected_tail_names
            or len(set(actual)) != len(actual)
        ):
            raise ValueError(
                f"{experiment_type} experiment variant layout mismatch: "
                f"expected modes {expected_modes}, prefixes {expected_name_prefixes}, "
                f"and tail names {expected_tail_names}; got modes {actual_modes} and names {actual}"
            )
        return

    if experiment_type == "texture":
        expected = [
            "01_spaceflow_local_texture_routing",
            "02_trellis_raw_flat_prompt",
            "03_fixed_structure_appearance_fm",
            "04_fixed_structure_guideflow_appearance_fm",
        ]
        if actual != expected:
            raise ValueError(
                f"{experiment_type} experiment variant layout mismatch: "
                f"expected {expected}, got {actual}"
            )
        return

    expected_modes = ["local_tau", "global_tau", "global_tau"]
    actual_modes = [str(variant.get("mode") or "") for variant in variants]
    expected_name_prefixes = ["01_local_tau", "02_global_tau", "03_global_tau"]
    if (
        actual_modes != expected_modes
        or len(actual) != len(expected_name_prefixes)
        or any(not name.startswith(prefix) for name, prefix in zip(actual, expected_name_prefixes))
        or len(set(actual)) != len(actual)
    ):
        raise ValueError(
            f"{experiment_type} experiment variant layout mismatch: "
            f"expected modes {expected_modes} with prefixes {expected_name_prefixes}, "
            f"got modes {actual_modes} and names {actual}"
        )


def _write_experiment_manifest(output_dir: Path, variants: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_variants = [
        {key: value for key, value in variant.items() if key not in {"command", "argv"}}
        for variant in variants
    ]
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps({"variants": manifest_variants}, indent=2),
        encoding="utf-8",
    )
    for variant in manifest_variants:
        variant_output_dir = Path(str(variant["output_dir"]))
        variant_output_dir.mkdir(parents=True, exist_ok=True)
        (variant_output_dir / "run_config.json").write_text(
            json.dumps(variant, indent=2),
            encoding="utf-8",
        )


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_nonnegative_int(value: object, default: int, name: str, *, min_value: int = 0) -> int:
    if value is None:
        return default
    requirement = "a non-negative integer" if min_value == 0 else f"an integer at least {min_value}"
    if isinstance(value, bool):
        raise ValueError(f"{name} must be {requirement}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        parsed = int(value.strip())
    else:
        raise ValueError(f"{name} must be {requirement}")
    if parsed < min_value:
        raise ValueError(f"{name} must be {requirement}")
    return parsed


def _offline_cache_enabled() -> bool:
    mode = OFFLINE_CACHE_MODE
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    required_cache_dirs = [
        CACHE_ROOT / "hub" / "models--microsoft--TRELLIS-text-xlarge",
        CACHE_ROOT / "hub" / "models--microsoft--TRELLIS-image-large",
        CACHE_ROOT / "hub" / "models--openai--clip-vit-large-patch14",
    ]
    return all(path.is_dir() for path in required_cache_dirs)


def _build_run_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(CACHE_ROOT))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_ROOT / "hub"))
    env.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "transformers"))
    env.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_ROOT))
    env.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
    env.setdefault("TMPDIR", str(RUN_ROOT / "tmp"))
    env.setdefault("SPCONV_ALGO", "native")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if _offline_cache_enabled():
        for key in HF_OFFLINE_ENV_KEYS:
            env.setdefault(key, "1")
    else:
        for key in HF_OFFLINE_ENV_KEYS:
            env.pop(key, None)
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TORCH_HOME", "TMPDIR"):
        Path(env[key]).expanduser().mkdir(parents=True, exist_ok=True)
    return env


def _read_json_field(form: cgi.FieldStorage, name: str, default: object) -> object:
    raw = form.getfirst(name)
    if not raw:
        return default
    return json.loads(raw)


def _ensure_inside_root(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    root_resolved = root.expanduser().resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Path is outside allowed root: {resolved}")
    return resolved


def _copy_uploaded(form: cgi.FieldStorage, field: str, target: Path) -> None:
    item = form[field] if field in form else None
    if item is None or not getattr(item, "filename", ""):
        raise ValueError(f"Missing uploaded file field: {field}")
    with target.open("wb") as fh:
        shutil.copyfileobj(item.file, fh)


def _uploaded_item(form: cgi.FieldStorage, field: str) -> cgi.FieldStorage | None:
    item = form[field] if field in form else None
    if item is None or not getattr(item, "filename", ""):
        return None
    return item


def _image_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return suffix
    return ".png"


def _copy_texture_upload(
    form: cgi.FieldStorage,
    field: str,
    target_dir: Path,
    stem: str,
) -> str | None:
    item = _uploaded_item(form, field)
    if item is None:
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{_sanitize_name(stem)}{_image_suffix(str(item.filename))}"
    with target.open("wb") as fh:
        shutil.copyfileobj(item.file, fh)
    return str(target)


def _primitive_count_from_asset(asset_entry: dict[str, object]) -> int:
    counts = asset_entry.get("counts", {})
    if isinstance(counts, dict):
        try:
            return int(counts.get("all") or 0)
        except (TypeError, ValueError):
            pass
    return 0


def _primitive_name_tags(asset_entry: dict[str, object], count: int) -> list[str]:
    names = [f"primitive_{i}" for i in range(count)]
    display_names = _primitive_display_names(asset_entry, count)
    return [_sanitize_name(name) for name in display_names] or names


def _primitive_display_names(asset_entry: dict[str, object], count: int) -> list[str]:
    names = [f"SQ {i + 1}" for i in range(count)]
    manifest_path = str(asset_entry.get("manifest_path") or "")
    if not manifest_path:
        return names
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        primitives = manifest.get("primitives", []) if isinstance(manifest, dict) else []
        if not isinstance(primitives, list):
            return names
        for item in primitives:
            if not isinstance(item, dict):
                continue
            index = int(item.get("index"))
            if 0 <= index < count:
                names[index] = str(item.get("name") or names[index]).strip() or names[index]
    except Exception:
        return names
    return names


def _normalize_local_values(raw: object, count: int, field_name: str) -> list[str]:
    if raw is None:
        values: list[object] = []
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError(f"{field_name} must be a JSON array")
    if count < 1:
        raise ValueError("Could not determine primitive count for local texture guidance")
    if len(values) > count:
        raise ValueError(f"{field_name} has {len(values)} entries but only {count} primitives")
    normalized = [str(value or "").strip() for value in values]
    normalized.extend([""] * (count - len(normalized)))
    return normalized


def _flatten_texture_prompt(
    *,
    text_prompt: str,
    global_texture_text: str,
    local_text_prompts: list[str],
    primitive_names: list[str],
) -> str:
    shape = text_prompt.strip()
    global_texture = (global_texture_text.strip() or shape).strip()
    parts = [
        f"{shape}.",
        f"Overall appearance and texture: {global_texture}.",
    ]
    local_parts = []
    for index, prompt in enumerate(local_text_prompts):
        prompt = prompt.strip()
        if not prompt:
            continue
        name = primitive_names[index] if index < len(primitive_names) else f"SQ {index + 1}"
        local_parts.append(f"part {index + 1} ({name}): {prompt}")
    if local_parts:
        parts.append("Local texture overrides: " + "; ".join(local_parts) + ".")
        parts.append("All unspecified parts should use the overall appearance and texture.")
    else:
        parts.append("Apply the overall appearance and texture consistently to every part.")
    return " ".join(parts)


def _local_texture_upload_indices(form: cgi.FieldStorage) -> list[int]:
    indices: list[int] = []
    for key in form.keys():
        match = re.fullmatch(r"local_texture_image_(\d+)", str(key))
        if match and _uploaded_item(form, str(key)) is not None:
            indices.append(int(match.group(1)))
    return sorted(indices)


def _history_path() -> Path:
    return SAVE_ROOT / "index.json"


def _read_history() -> dict[str, object]:
    path = _history_path()
    if not path.is_file():
        return {"entries": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data
    except json.JSONDecodeError:
        pass
    return {"entries": []}


def _write_history(data: dict[str, object]) -> None:
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    path = _history_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_history(entry: dict[str, object]) -> None:
    data = _read_history()
    entries = data.setdefault("entries", [])
    if not isinstance(entries, list):
        entries = []
        data["entries"] = entries
    entries.insert(0, entry)
    del entries[250:]
    _write_history(data)


def _save_bundle_from_form(form: cgi.FieldStorage, run_id: str | None = None) -> dict[str, object]:
    project_name = _sanitize_name(form.getfirst("projectName", "superquadrics") or "superquadrics")
    timestamp = _utc_timestamp()
    save_id = run_id or f"{timestamp}_{project_name}"
    asset_dir = SAVE_ROOT / project_name / save_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "all": asset_dir / "all.npz",
        "high_control": asset_dir / "high_control.npz",
        "low_control_bbox": asset_dir / "low_control_bbox.npz",
    }
    _copy_uploaded(form, "all", paths["all"])
    _copy_uploaded(form, "high_control", paths["high_control"])
    _copy_uploaded(form, "low_control_bbox", paths["low_control_bbox"])

    manifest = _read_json_field(form, "manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    manifest = {
        **manifest,
        "project_name": project_name,
        "saved_at": timestamp,
        "save_id": save_id,
        "paths": {key: str(path) for key, path in paths.items()},
    }
    manifest_path = asset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    counts = manifest.get("counts", {})
    entry = {
        "id": save_id,
        "project_name": project_name,
        "saved_at": timestamp,
        "asset_dir": str(asset_dir),
        "manifest_path": str(manifest_path),
        "paths": {key: str(path) for key, path in paths.items()},
        "counts": counts,
    }
    _append_history(entry)
    return entry


def _run_meta_path(run_id: str) -> Path:
    return RUN_ROOT / run_id / "run_meta.json"


def _write_run_meta(run_id: str, meta: dict[str, object]) -> None:
    path = _run_meta_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _read_run_meta(run_id: str) -> dict[str, object] | None:
    path = _run_meta_path(run_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _signal_run_process(proc: subprocess.Popen[bytes], sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            return


def _log_tail(log_path: str | Path, n: int = 6000) -> str:
    path = Path(log_path)
    if not path.is_file():
        return ""
    return _clean_log_text(path.read_text(encoding="utf-8", errors="replace")[-n:])


def _clean_log_text(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "\n")
    lines: list[str] = []
    previous_blank = False
    for line in text.splitlines():
        clean = line.rstrip()
        stripped = clean.strip()
        if "%|" in stripped and ("it/s" in stripped or "B/s" in stripped):
            continue
        if stripped.startswith(("Sampling:", "Rendering:")) and "it/s" in stripped:
            continue
        is_blank = not clean
        if is_blank and previous_blank:
            continue
        lines.append(clean)
        previous_blank = is_blank
    return "\n".join(lines)


def _file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".glb", ".gltf", ".obj"}:
        return "mesh"
    if suffix in {".ply", ".npz", ".npy"}:
        return "data"
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix in {".mp4", ".webm", ".mov"}:
        return "video"
    if suffix in {".log", ".txt"}:
        return "log"
    return "file"


def _run_file_url(run_id: str, relative_path: str) -> str:
    return f"/spaceflow/runs/file?run_id={quote(run_id)}&path={quote(relative_path)}"


def _public_demo_variant_priority(variant: object, index: int) -> tuple[int, int]:
    if not isinstance(variant, dict):
        return (100, index)
    mode = str(variant.get("mode") or "").strip().lower()
    name = str(variant.get("name") or "").strip().lower()
    if mode == "spaceflow_local_texture_routing" or name.startswith("01_spaceflow_local_texture_routing"):
        return (0, index)
    if mode == "local_tau" or name.startswith("01_local"):
        return (1, index)
    if name.startswith("01_"):
        return (2, index)
    return (50, index)


def _public_demo_final_output_paths(meta: dict[str, object], output_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    add(output_dir / PUBLIC_DEMO_FINAL_OUTPUT)
    variants = meta.get("experiment_variants")
    if isinstance(variants, list):
        ordered = sorted(enumerate(variants), key=lambda item: _public_demo_variant_priority(item[1], item[0]))
        for _, variant in ordered:
            if not isinstance(variant, dict):
                continue
            raw = str(variant.get("output_dir") or "").strip()
            if not raw:
                continue
            variant_output_dir = Path(raw)
            if not variant_output_dir.is_absolute():
                variant_output_dir = output_dir / variant_output_dir
            add(variant_output_dir / PUBLIC_DEMO_FINAL_OUTPUT)

    add(output_dir / "01_spaceflow_local_texture_routing" / PUBLIC_DEMO_FINAL_OUTPUT)
    return candidates


def _public_demo_allowed_output_rel_paths(meta: dict[str, object], output_dir: Path) -> set[str]:
    output_root = output_dir.resolve()
    allowed: set[str] = set()
    for path in _public_demo_final_output_paths(meta, output_dir):
        try:
            allowed.add(path.resolve().relative_to(output_root).as_posix())
        except ValueError:
            continue
    return allowed


def _list_output_files(meta: dict[str, object]) -> list[dict[str, object]]:
    output_dir_raw = str(meta.get("output_dir") or "")
    if not output_dir_raw:
        return []
    output_dir = Path(output_dir_raw)
    if not output_dir.is_dir():
        return []

    run_id = str(meta.get("run_id") or "")
    seen: set[str] = set()
    files: list[dict[str, object]] = []

    def add(path: Path) -> None:
        if not path.is_file():
            return
        try:
            rel = path.relative_to(output_dir).as_posix()
        except ValueError:
            return
        if rel in seen:
            return
        seen.add(rel)
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "relative_path": rel,
                "kind": _file_kind(path),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "url": _run_file_url(run_id, rel) if run_id else "",
            }
        )

    if PUBLIC_DEMO:
        for path in _public_demo_final_output_paths(meta, output_dir):
            add(path)
        return files

    for rel in KNOWN_OUTPUTS:
        add(output_dir / rel)

    output_file_limit = 220 if meta.get("experiment_mode") else 80
    allowed_suffixes = {".glb", ".ply", ".mp4", ".png", ".jpg", ".jpeg", ".webp", ".npz", ".json", ".log", ".txt"}
    for path in sorted(output_dir.rglob("*")):
        if len(files) >= output_file_limit:
            break
        if path.suffix.lower() in allowed_suffixes:
            add(path)

    return files


def _warning_key(warning: dict[str, object]) -> tuple[str, str]:
    return (
        str(warning.get("kind") or "warning"),
        str(warning.get("message") or ""),
    )


def _normalize_warning(raw: object, *, source: str) -> dict[str, object] | None:
    if isinstance(raw, str):
        message = raw.strip()
        if not message:
            return None
        return {
            "kind": "warning",
            "severity": "warning",
            "message": message,
            "source": source,
        }
    if not isinstance(raw, dict):
        return None
    message = str(raw.get("message") or "").strip()
    if not message:
        return None
    warning = dict(raw)
    warning.setdefault("kind", "warning")
    warning.setdefault("severity", "warning")
    warning.setdefault("source", source)
    return warning


def _warnings_from_file(path: Path, output_dir: Path) -> list[dict[str, object]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = raw.get("warnings") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    try:
        source = path.relative_to(output_dir).as_posix()
    except ValueError:
        source = str(path)
    warnings = []
    for item in items:
        warning = _normalize_warning(item, source=source)
        if warning is not None:
            warnings.append(warning)
    return warnings


def _routing_warnings_from_log(log_path: str) -> list[dict[str, object]]:
    if not log_path:
        return []
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    warnings = []
    pattern = re.compile(
        r"Local (?P<kind>text|image) condition\(s\) for SQ\(s\) "
        r"\[(?P<indices>[^\]]+)\] have zero routed cells"
    )
    for match in pattern.finditer(text):
        indices: list[int] = []
        for token in match.group("indices").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                indices.append(int(token))
            except ValueError:
                pass
        if not indices:
            continue
        sq_numbers = [index + 1 for index in indices]
        sq_label = ", ".join(str(number) for number in sq_numbers)
        local_prompt_type = match.group("kind")
        warnings.append({
            "kind": "local_routing_zero_cells",
            "severity": "warning",
            "message": (
                f"Local {local_prompt_type} prompt(s) for SQs {sq_label} "
                "received zero routed cells; those local overrides had no effect."
            ),
            "local_prompt_type": local_prompt_type,
            "sq_indices": indices,
            "sq_numbers": sq_numbers,
            "source": "spaceflow.log",
        })
    return warnings


def _collect_run_warnings(meta: dict[str, object]) -> list[dict[str, object]]:
    output_dir_raw = str(meta.get("output_dir") or "")
    warnings: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    if output_dir_raw and not PUBLIC_DEMO:
        output_dir = Path(output_dir_raw)
        if output_dir.is_dir():
            for warning_path in sorted(output_dir.rglob("routing_warnings.json")):
                for warning in _warnings_from_file(warning_path, output_dir):
                    key = _warning_key(warning)
                    if key not in seen:
                        warnings.append(warning)
                        seen.add(key)

    for warning in _routing_warnings_from_log(str(meta.get("log_path") or "")):
        key = _warning_key(warning)
        if key not in seen:
            warnings.append(warning)
            seen.add(key)

    return warnings


def _run_with_outputs(meta: dict[str, object]) -> dict[str, object]:
    run = dict(meta)
    run["output_files"] = _list_output_files(run)
    run["warnings"] = _collect_run_warnings(run)
    return run


def _reconcile_untracked_run(meta: dict[str, object]) -> tuple[dict[str, object], bool]:
    if not _is_active_status(meta.get("status")):
        return meta, False
    log_text = _log_tail(str(meta.get("log_path", "")), 12000)
    lower = log_text.lower()
    failure_markers = [
        "force terminated",
        "allocation",
        "revoked",
        "traceback",
        "runtimeerror",
        "error:",
        "batch job submission failed",
        "exited with exit code",
        "cuda out of memory",
        "failed with code",
        "cancelled at",
        "job step aborted",
        "task 0: terminated",
        "exited with status 1",
        "exited with status 2",
        "exited with status 88",
    ]
    if any(marker in lower for marker in failure_markers):
        meta = dict(meta)
        meta["status"] = "failed"
        meta.setdefault("returncode", -1)
        return meta, True
    success_markers = [
        "structure-only mode complete",
        "[experiment] completed spaceflow experiment",
        "completed spaceflow run",
        "spaceflow command exited with status 0",
        "rendered tau-by-parts summary figure",
    ]
    if any(marker in lower for marker in success_markers) and _list_output_files(meta):
        meta = dict(meta)
        meta["status"] = "succeeded"
        meta.setdefault("returncode", 0)
        return meta, True
    last_activity = _run_last_activity_time(meta)
    if last_activity and time.time() - last_activity > RUN_TIMEOUT:
        meta = dict(meta)
        meta["status"] = "succeeded" if _list_output_files(meta) else "failed"
        meta.setdefault("returncode", 0 if meta["status"] == "succeeded" else -1)
        return meta, True
    return meta, False


def _run_last_activity_time(meta: dict[str, object]) -> float:
    candidates: list[float] = []
    for key in ("log_path", "output_dir"):
        raw = str(meta.get(key) or "")
        if not raw:
            continue
        path = Path(raw)
        try:
            candidates.append(path.stat().st_mtime)
        except OSError:
            pass
    run_id = str(meta.get("run_id") or "")
    if run_id:
        try:
            candidates.append(_run_meta_path(run_id).stat().st_mtime)
        except OSError:
            pass
    return max(candidates) if candidates else 0.0


def _is_active_status(status: object) -> bool:
    return str(status or "").strip().lower() in ACTIVE_RUN_STATUSES


def _reconcile_run_state(run_id: str, meta: dict[str, object]) -> dict[str, object]:
    meta = dict(meta)
    changed = False
    proc = RUNS.get(run_id)
    if proc is not None:
        code = proc.poll()
        if code is None:
            if meta.get("cancel_requested"):
                stop_requested_at = float(meta.get("stop_requested_at") or 0)
                if stop_requested_at and time.time() - stop_requested_at > STOP_GRACE_SEC:
                    _signal_run_process(proc, signal.SIGKILL)
                next_status = "cancelling"
            else:
                next_status = "running"
            if meta.get("status") != next_status:
                meta["status"] = next_status
                changed = True
        else:
            next_status = "cancelled" if meta.get("cancel_requested") else ("succeeded" if code == 0 else "failed")
            if meta.get("status") != next_status or meta.get("returncode") != code:
                meta["status"] = next_status
                meta["returncode"] = code
                changed = True
            RUNS.pop(run_id, None)
    else:
        meta, changed = _reconcile_untracked_run(meta)
    if changed:
        _write_run_meta(run_id, meta)
    return meta


def _active_run_ids() -> list[str]:
    active: set[str] = set()
    for run_id in list(RUNS.keys()):
        meta = _read_run_meta(run_id) or {"run_id": run_id, "status": "running"}
        meta = _reconcile_run_state(run_id, meta)
        if _is_active_status(meta.get("status")):
            active.add(run_id)

    if RUN_ROOT.is_dir():
        for meta_path in RUN_ROOT.glob("*/run_meta.json"):
            run_id = meta_path.parent.name
            if run_id in active:
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(meta, dict):
                continue
            meta = _reconcile_run_state(run_id, meta)
            if _is_active_status(meta.get("status")):
                active.add(run_id)
    return sorted(active)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file() or item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _cleanup_run_records(active_run_ids: set[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not RUN_ROOT.is_dir():
        return records
    for meta_path in RUN_ROOT.glob("*/run_meta.json"):
        run_dir = meta_path.parent
        run_id = run_dir.name
        if run_id in active_run_ids:
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            stat = run_dir.stat()
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        status = str(meta.get("status") or "").strip().lower()
        if status in ACTIVE_RUN_STATUSES or status not in CLEANUP_RUN_STATUSES:
            continue
        records.append({
            "run_id": run_id,
            "path": run_dir,
            "status": status,
            "mtime": stat.st_mtime,
            "size": _dir_size_bytes(run_dir),
        })
    records.sort(key=lambda item: float(item["mtime"]))
    return records


def _delete_run_dir(record: dict[str, object], reason: str) -> bool:
    run_dir = record["path"]
    if not isinstance(run_dir, Path):
        return False
    try:
        shutil.rmtree(run_dir)
        print(
            f"[sq-spaceflow] Cleanup removed {record.get('run_id')} "
            f"({record.get('status')}, {reason})",
            flush=True,
        )
        return True
    except OSError as exc:
        print(f"[sq-spaceflow] Cleanup could not remove {run_dir}: {exc}", flush=True)
        return False


def _cleanup_old_runs() -> None:
    if RETENTION_HOURS <= 0 and MAX_STORAGE_GB <= 0:
        return
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    active = set(_active_run_ids())
    records = _cleanup_run_records(active)

    if RETENTION_HOURS > 0:
        cutoff = time.time() - RETENTION_HOURS * 3600
        kept: list[dict[str, object]] = []
        for record in records:
            if float(record["mtime"]) < cutoff:
                _delete_run_dir(record, f"older than {RETENTION_HOURS:g}h")
            else:
                kept.append(record)
        records = kept

    if MAX_STORAGE_GB > 0:
        limit_bytes = int(MAX_STORAGE_GB * 1024 * 1024 * 1024)
        total_bytes = _dir_size_bytes(RUN_ROOT)
        for record in records:
            if total_bytes <= limit_bytes:
                break
            if _delete_run_dir(record, f"storage over {MAX_STORAGE_GB:g}GB"):
                total_bytes -= int(record["size"])


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "SpaceFlowUIService/0.1"

    def _cors(self) -> None:
        if CORS_ORIGIN:
            self.send_header("Access-Control-Allow-Origin", CORS_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[sq-spaceflow] {fmt % args}", flush=True)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"", "/", "/spaceflow/health"}:
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "spaceflow",
                    "asset_root": str(SAVE_ROOT),
                    "run_root": str(RUN_ROOT),
                    "cache_root": str(CACHE_ROOT),
                    "python": PYTHON_BIN,
                    "run_script": str(RUN_SCRIPT),
                    "uses_srun": _should_use_srun(),
                    "gpu_preflight": _gpu_preflight_mode(),
                    "offline_cache": _offline_cache_enabled(),
                    "slurm_constraint": CONSTRAINT,
                    "slurm_exclude": EXCLUDE_NODES,
                },
            )
            return
        if parsed.path == "/spaceflow/assets/history":
            self._handle_history(parsed.query)
            return
        if parsed.path == "/spaceflow/assets/open":
            self._handle_open_asset(parsed.query)
            return
        if parsed.path == "/spaceflow/runs/status":
            self._handle_run_status(parsed.query)
            return
        if parsed.path == "/spaceflow/runs/log":
            self._handle_run_log(parsed.query)
            return
        if parsed.path == "/spaceflow/runs/file":
            self._handle_run_file(parsed.query)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/spaceflow/assets/save":
            self._handle_save()
            return
        if self.path == "/spaceflow/runs/start":
            self._handle_run_start()
            return
        if self.path == "/spaceflow/runs/stop":
            self._handle_run_stop()
            return
        self.send_error(404)

    def _multipart_form(self) -> cgi.FieldStorage:
        ctype, pdict = cgi.parse_header(self.headers.get("Content-Type", ""))
        if ctype != "multipart/form-data" or "boundary" not in pdict:
            raise ValueError("Expected multipart/form-data")
        return cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})

    def _handle_save(self) -> None:
        try:
            form = self._multipart_form()
            entry = _save_bundle_from_form(form)
            self._send_json(200, {"status": "ok", "entry": entry})
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": {"message": str(exc)}})

    def _handle_history(self, query: str) -> None:
        params = parse_qs(query)
        limit = max(1, min(250, int((params.get("limit") or ["50"])[0] or "50")))
        data = _read_history()
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            entries = []
        self._send_json(200, {"status": "ok", "entries": entries[:limit]})

    def _handle_open_asset(self, query: str) -> None:
        params = parse_qs(query)
        raw_path = (params.get("path") or [""])[0]
        if not raw_path:
            self._send_json(400, {"error": {"message": "Missing path"}})
            return
        try:
            file_path = _ensure_inside_root(Path(raw_path), SAVE_ROOT)
        except ValueError as exc:
            self._send_json(403, {"error": {"message": str(exc)}})
            return
        if file_path.suffix.lower() != ".npz" or not file_path.is_file():
            self._send_json(404, {"error": {"message": f"NPZ file not found: {file_path}"}})
            return
        _send_file(self, file_path)

    def _handle_run_start(self) -> None:
        with RUN_LOCK:
            self._handle_run_start_locked()

    def _handle_run_start_locked(self) -> None:
        try:
            _cleanup_old_runs()
            if MAX_ACTIVE_RUNS > 0:
                active_run_ids = _active_run_ids()
                if len(active_run_ids) >= MAX_ACTIVE_RUNS:
                    self._send_json(
                        409,
                        {
                            "error": {
                                "message": "GPU busy: another SpaceFlow run is active. Try again when it finishes.",
                                "active_run_ids": active_run_ids,
                            }
                        },
                    )
                    return
            form = self._multipart_form()
            run_config = _read_json_field(form, "runConfig", {})
            if not isinstance(run_config, dict):
                raise ValueError("runConfig must be a JSON object")

            project_name = _sanitize_name(form.getfirst("projectName", "superquadrics") or "superquadrics")
            experiment_mode = _parse_bool(str(run_config.get("experimentMode", False)), False)
            text_prompt = str(run_config.get("textPrompt", "")).strip()
            if not text_prompt:
                raise ValueError("Missing text prompt")
            output_name_raw = str(run_config.get("outputName") or text_prompt)
            experiment_type_raw = str(run_config.get("experimentType") or "").strip().lower()
            if not experiment_type_raw and experiment_mode:
                if output_name_raw.endswith("_full_experiment"):
                    experiment_type_raw = "full"
                elif output_name_raw.endswith("_texture_experiment"):
                    experiment_type_raw = "texture"
                else:
                    experiment_type_raw = "geometry"
            experiment_type = experiment_type_raw or "single"
            if experiment_mode and experiment_type not in {"geometry", "texture", "full"}:
                raise ValueError(f"Unknown experiment type: {experiment_type}")
            if experiment_mode:
                experiment_suffix = (
                    "_texture_experiment"
                    if experiment_type == "texture"
                    else "_full_experiment" if experiment_type == "full" else "_experiment"
                )
                if not output_name_raw.endswith(experiment_suffix):
                    output_name_raw = f"{output_name_raw}{experiment_suffix}"
            output_name = _sanitize_name(output_name_raw)
            run_id = f"{_utc_timestamp()}_{output_name}"
            asset_entry = _save_bundle_from_form(form, run_id=run_id)
            run_dir = RUN_ROOT / run_id
            output_dir = run_dir / "output"
            log_path = run_dir / "spaceflow.log"
            run_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            low_tau = float(run_config.get("lowTau", 3.0))
            high_tau = float(run_config.get("highTau", 10.0))
            polyak = float(run_config.get("polyakTau", 0.18))
            n_repaint_steps = _parse_nonnegative_int(
                run_config.get(
                    "repaintSteps",
                    run_config.get("nRepaintSteps", run_config.get("n_repaint_steps", 10)),
                ),
                10,
                "Repaint steps",
            )
            texture_optim_steps = _parse_nonnegative_int(
                run_config.get(
                    "textureOptimSteps",
                    run_config.get("texture_optim_steps", 300),
                ),
                300,
                "Texture optimization steps",
                min_value=2,
            )
            texture_mode = str(run_config.get("textureMode") or run_config.get("appearanceMode", "text")).strip().lower()
            if texture_mode not in {"text", "image"}:
                raise ValueError(f"Unknown texture mode: {texture_mode}")
            if experiment_mode and experiment_type in {"texture", "full"} and texture_mode != "text":
                raise ValueError("Texture and full experiments support text texture guidance only. Switch Texture guidance to Text.")
            convert_yup_to_zup = _parse_bool(str(run_config.get("convertYupToZup", True)), True)
            dry_run = _parse_bool(str(run_config.get("dryRun", False)), False)

            asset_paths = asset_entry["paths"]
            assert isinstance(asset_paths, dict)
            primitive_count = _primitive_count_from_asset(asset_entry)
            primitive_tags = _primitive_name_tags(asset_entry, primitive_count)
            texture_dir = run_dir / "inputs" / "texture"

            appearance_args: list[str] = []
            local_texture_args: list[str] = []
            appearance_text = ""
            local_text_prompts = [""] * primitive_count
            texture_meta: dict[str, object] = {
                "mode": texture_mode,
                "saved_uploads": {},
            }
            saved_uploads: dict[str, str] = {}
            if texture_mode == "image":
                appearance_image_path = str(
                    run_config.get("globalTextureImagePath")
                    or run_config.get("appearanceImagePath")
                    or ""
                ).strip()
                uploaded_global = (
                    _copy_texture_upload(form, "texture_image", texture_dir, "global_texture")
                    or _copy_texture_upload(form, "appearance_image", texture_dir, "global_texture")
                )
                if uploaded_global:
                    appearance_image_path = uploaded_global
                    saved_uploads["global"] = uploaded_global
                if not appearance_image_path:
                    raise ValueError("Image texture mode requires a global image path or upload")
                appearance_args = ["--appearance_image", appearance_image_path]

                local_image_paths = _normalize_local_values(
                    run_config.get("localTextureImagePaths"),
                    primitive_count,
                    "localTextureImagePaths",
                )
                for index in _local_texture_upload_indices(form):
                    if index < 0 or index >= primitive_count:
                        raise ValueError(f"Local texture image upload index {index} is outside 0..{primitive_count - 1}")
                    stem = f"local_texture_{index}_{primitive_tags[index]}"
                    uploaded_local = _copy_texture_upload(form, f"local_texture_image_{index}", texture_dir, stem)
                    if uploaded_local:
                        local_image_paths[index] = uploaded_local
                        saved_uploads[f"local_{index}"] = uploaded_local
                if any(path for path in local_image_paths):
                    local_texture_args = ["--local_image_paths", json.dumps(local_image_paths)]
                texture_meta.update({
                    "global_image_path": appearance_image_path,
                    "local_override_count": sum(1 for path in local_image_paths if path),
                    "local_image_paths": local_image_paths,
                })
            else:
                appearance_text = str(
                    run_config.get("globalTextureText")
                    or run_config.get("appearanceText")
                    or text_prompt
                ).strip() or text_prompt
                appearance_args = ["--appearance_text", appearance_text]
                local_text_prompts = _normalize_local_values(
                    run_config.get("localTextureTexts"),
                    primitive_count,
                    "localTextureTexts",
                )
                if any(prompt.strip() for prompt in local_text_prompts):
                    local_texture_args = ["--local_text_prompts", json.dumps(local_text_prompts)]
                texture_meta.update({
                    "global_text": appearance_text,
                    "local_override_count": sum(1 for prompt in local_text_prompts if prompt.strip()),
                    "local_text_prompts": local_text_prompts,
                })
            texture_meta["saved_uploads"] = saved_uploads

            experiment_variants: list[dict[str, object]] = []
            experiment_script: Path | None = None
            run_script: Path | None = None

            texture_flattened_prompt: str | None = None

            if experiment_mode:
                if experiment_type in {"geometry", "full"}:
                    experiment_specs = [
                        {
                            "name": _local_tau_variant_name(1, low_tau, high_tau, polyak),
                            "mode": "local_tau",
                            "low_tau": low_tau,
                            "high_tau": high_tau,
                            "polyak_tau": polyak,
                        },
                        {
                            "name": _global_tau_variant_name(2, low_tau),
                            "mode": "global_tau",
                            "low_tau": low_tau,
                            "high_tau": None,
                            "polyak_tau": 0.0,
                        },
                        {
                            "name": _global_tau_variant_name(3, high_tau),
                            "mode": "global_tau",
                            "low_tau": high_tau,
                            "high_tau": None,
                            "polyak_tau": 0.0,
                        },
                    ]
                    for spec in experiment_specs:
                        variant_name = _sanitize_name(str(spec["name"]))
                        variant_output_dir = output_dir / variant_name
                        variant_cmd = _spaceflow_cmd(
                            asset_paths,
                            variant_output_dir,
                            text_prompt=text_prompt,
                            appearance_args=appearance_args,
                            local_texture_args=local_texture_args,
                            low_tau=float(spec["low_tau"]),
                            high_tau=None if spec["high_tau"] is None else float(spec["high_tau"]),
                            polyak_tau=float(spec["polyak_tau"]),
                            n_repaint_steps=n_repaint_steps,
                            texture_optim_steps=texture_optim_steps,
                            convert_yup_to_zup=convert_yup_to_zup,
                        )
                        variant_argv = [str(part) for part in variant_cmd[2:]]
                        experiment_variants.append({
                            "name": variant_name,
                            "output_dir": str(variant_output_dir),
                            "command": variant_cmd,
                            "argv": variant_argv,
                            "mode": spec["mode"],
                            "low_tau": spec["low_tau"],
                            "high_tau": spec["high_tau"],
                            "polyak_tau": spec["polyak_tau"],
                            "n_repaint_steps": n_repaint_steps,
                            "texture_optim_steps": texture_optim_steps,
                        })

                if experiment_type in {"texture", "full"}:
                    primitive_display_names = _primitive_display_names(asset_entry, primitive_count)
                    auto_texture_flattened_prompt = _flatten_texture_prompt(
                        text_prompt=text_prompt,
                        global_texture_text=appearance_text,
                        local_text_prompts=local_text_prompts,
                        primitive_names=primitive_display_names,
                    )
                    ui_texture_prompt = str(run_config.get("textureExperimentPrompt") or "").strip()
                    texture_flattened_prompt = ui_texture_prompt or auto_texture_flattened_prompt
                    texture_meta["flattened_text_prompt"] = texture_flattened_prompt
                    texture_meta["auto_flattened_text_prompt"] = auto_texture_flattened_prompt
                    texture_meta["trellis_experiment_prompt"] = texture_flattened_prompt
                    texture_meta["trellis_experiment_prompt_source"] = "ui" if ui_texture_prompt else "auto"

                    local_variant_name = "01_spaceflow_local_texture_routing"
                    local_variant_output_dir = output_dir / local_variant_name
                    if experiment_type == "full":
                        source_variant_name = _sanitize_name(_local_tau_variant_name(1, low_tau, high_tau, polyak))
                        source_variant_output_dir = output_dir / source_variant_name
                        experiment_variants.append({
                            "name": local_variant_name,
                            "output_dir": str(local_variant_output_dir),
                            "runner": "copy_variant",
                            "mode": "spaceflow_local_texture_routing",
                            "source_variant": source_variant_name,
                            "source_output_dir": str(source_variant_output_dir),
                            "low_tau": low_tau,
                            "high_tau": high_tau,
                            "polyak_tau": polyak,
                            "n_repaint_steps": n_repaint_steps,
                            "texture_optim_steps": texture_optim_steps,
                        })
                    else:
                        local_variant_cmd = _spaceflow_cmd(
                            asset_paths,
                            local_variant_output_dir,
                            text_prompt=text_prompt,
                            appearance_args=appearance_args,
                            local_texture_args=local_texture_args,
                            low_tau=low_tau,
                            high_tau=high_tau,
                            polyak_tau=polyak,
                            n_repaint_steps=n_repaint_steps,
                            texture_optim_steps=texture_optim_steps,
                            convert_yup_to_zup=convert_yup_to_zup,
                        )
                        experiment_variants.append({
                            "name": local_variant_name,
                            "output_dir": str(local_variant_output_dir),
                            "command": local_variant_cmd,
                            "argv": [str(part) for part in local_variant_cmd[2:]],
                            "mode": "spaceflow_local_texture_routing",
                            "low_tau": low_tau,
                            "high_tau": high_tau,
                            "polyak_tau": polyak,
                            "n_repaint_steps": n_repaint_steps,
                            "texture_optim_steps": texture_optim_steps,
                        })
                    experiment_variants.append({
                        "name": "02_trellis_raw_flat_prompt",
                        "output_dir": str(output_dir / "02_trellis_raw_flat_prompt"),
                        "runner": "trellis_raw_text",
                        "mode": "trellis_raw_text",
                        "prompt": texture_flattened_prompt,
                        "flattened_prompt": texture_flattened_prompt,
                        "seed": 1,
                        "low_tau": None,
                        "high_tau": None,
                        "polyak_tau": None,
                        "n_repaint_steps": n_repaint_steps,
                        "texture_optim_steps": texture_optim_steps,
                        "input_superquadrics_glb_path": str(local_variant_output_dir / "input_superquadrics_colored.glb"),
                    })
                    experiment_variants.append({
                        "name": "03_fixed_structure_appearance_fm",
                        "output_dir": str(output_dir / "03_fixed_structure_appearance_fm"),
                        "runner": "fixed_structure_appearance_fm",
                        "mode": "fixed_structure_appearance_fm",
                        "prompt": texture_flattened_prompt,
                        "flattened_prompt": texture_flattened_prompt,
                        "structure_voxels_path": str(local_variant_output_dir / "voxels" / "struct_voxels.ply"),
                        "source_variant": local_variant_name,
                        "seed": 1,
                        "low_tau": None,
                        "high_tau": None,
                        "polyak_tau": None,
                        "n_repaint_steps": n_repaint_steps,
                        "texture_optim_steps": texture_optim_steps,
                        "input_superquadrics_glb_path": str(local_variant_output_dir / "input_superquadrics_colored.glb"),
                    })
                    experiment_variants.append({
                        "name": "04_fixed_structure_guideflow_appearance_fm",
                        "output_dir": str(output_dir / "04_fixed_structure_guideflow_appearance_fm"),
                        "runner": "fixed_structure_guideflow_appearance_fm",
                        "mode": "fixed_structure_guideflow_appearance_fm",
                        "prompt": texture_flattened_prompt,
                        "flattened_prompt": texture_flattened_prompt,
                        "structure_voxels_path": str(local_variant_output_dir / "voxels" / "struct_voxels.ply"),
                        "source_variant": local_variant_name,
                        "seed": 1,
                        "low_tau": None,
                        "high_tau": None,
                        "polyak_tau": None,
                        "n_repaint_steps": n_repaint_steps,
                        "texture_optim_steps": texture_optim_steps,
                        "input_superquadrics_glb_path": str(local_variant_output_dir / "input_superquadrics_colored.glb"),
                    })
                _assert_experiment_variant_layout(experiment_type, experiment_variants)
                _write_experiment_manifest(output_dir, experiment_variants)
                experiment_runner_config = run_dir / "experiment_runner_config.json"
                _write_experiment_runner_config(
                    experiment_runner_config,
                    experiment_variants,
                    experiment_type=experiment_type if experiment_type in {"texture", "full"} else None,
                    texture_flattened_prompt=texture_flattened_prompt,
                    texture_optim_steps=texture_optim_steps,
                )
                experiment_script = run_dir / "run_experiment.sh"
                experiment_cmd = [
                    PYTHON_BIN,
                    str(EXPERIMENT_RUNNER_SCRIPT),
                    "--config",
                    str(experiment_runner_config),
                ]
                _write_command_script(experiment_script, experiment_cmd)
                final_cmd = _wrap_shell_with_srun(experiment_script) if _should_use_srun() else ["bash", str(experiment_script)]
            else:
                cmd = _spaceflow_cmd(
                    asset_paths,
                    output_dir,
                    text_prompt=text_prompt,
                    appearance_args=appearance_args,
                    local_texture_args=local_texture_args,
                    low_tau=low_tau,
                    high_tau=high_tau,
                    polyak_tau=polyak,
                    n_repaint_steps=n_repaint_steps,
                    texture_optim_steps=texture_optim_steps,
                    convert_yup_to_zup=convert_yup_to_zup,
                )
                run_script = run_dir / "run_spaceflow.sh"
                _write_single_run_script(run_script, cmd, run_dir)
                final_cmd = (
                    _wrap_shell_with_srun(run_script, job_name="sq_spaceflow")
                    if _should_use_srun()
                    else ["bash", str(run_script)]
                )

            meta = {
                "run_id": run_id,
                "status": "dry_run" if dry_run else "running",
                "project_name": project_name,
                "created_at": _utc_timestamp(),
                "asset_entry": asset_entry,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
                "command": final_cmd,
                "run_config": run_config,
                "pipeline_stage": "full_pipeline" if FULL_PIPELINE else "structure_only",
                "launch_mode": "srun" if _should_use_srun() else "subprocess",
                "experiment_mode": experiment_mode,
                "experiment_type": experiment_type if experiment_mode else None,
                "texture_guidance": texture_meta,
            }
            if experiment_mode:
                meta["experiment_script"] = str(experiment_script)
                meta["experiment_runner_config"] = str(experiment_runner_config)
                meta["experiment_variants"] = [
                    {key: value for key, value in variant.items() if key not in {"command", "argv"}}
                    for variant in experiment_variants
                ]
            elif run_script is not None:
                meta["run_script"] = str(run_script)
            _write_run_meta(run_id, meta)

            if dry_run:
                log_path.write_text("Dry run command:\n" + " ".join(final_cmd) + "\n", encoding="utf-8")
            else:
                log_file = log_path.open("wb")
                try:
                    log_file.write(
                        (
                            f"{_local_log_timestamp()} [sq-spaceflow] Launching run command "
                            f"via {meta['launch_mode']}.\n"
                        ).encode("utf-8")
                    )
                    log_file.flush()
                    proc = subprocess.Popen(
                        final_cmd,
                        cwd=REPO_ROOT,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        env=_build_run_env(),
                        start_new_session=True,
                    )
                finally:
                    log_file.close()
                RUNS[run_id] = proc

            self._send_json(200, {"status": "ok", "run_id": run_id, "run": _run_with_outputs(meta)})
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": {"message": str(exc)}})

    def _handle_run_status(self, query: str) -> None:
        params = parse_qs(query)
        run_id = (params.get("run_id") or [""])[0]
        if not run_id:
            self._send_json(400, {"error": {"message": "Missing run_id"}})
            return
        meta = _read_run_meta(run_id)
        if not meta:
            self._send_json(404, {"error": {"message": f"Unknown run_id: {run_id}"}})
            return
        meta = _reconcile_run_state(run_id, meta)
        log_tail = _log_tail(str(meta.get("log_path", "")))
        self._send_json(200, {"status": "ok", "run": _run_with_outputs(meta), "log_tail": log_tail})

    def _handle_run_stop(self) -> None:
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0") or "0")) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "Expected JSON body"}})
            return
        run_id = str(payload.get("run_id") or "")
        if not run_id:
            self._send_json(400, {"error": {"message": "Missing run_id"}})
            return
        meta = _read_run_meta(run_id)
        if not meta:
            self._send_json(404, {"error": {"message": f"Unknown run_id: {run_id}"}})
            return
        proc = RUNS.get(run_id)
        meta = dict(meta)
        if proc is None or proc.poll() is not None:
            if meta.get("status") == "running":
                meta["status"] = "failed"
                meta.setdefault("returncode", proc.returncode if proc is not None else -1)
                _write_run_meta(run_id, meta)
            self._send_json(200, {"status": "ok", "run": _run_with_outputs(meta)})
            return

        meta["status"] = "cancelling"
        meta["cancel_requested"] = True
        meta["stop_requested_at"] = time.time()
        _write_run_meta(run_id, meta)
        log_path_raw = str(meta.get("log_path") or "")
        if log_path_raw:
            log_path = Path(log_path_raw)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n{_local_log_timestamp()} [sq-spaceflow] Stop requested by UI.\n")
        _signal_run_process(proc, signal.SIGTERM)
        self._send_json(200, {"status": "ok", "run": _run_with_outputs(meta)})

    def _handle_run_log(self, query: str) -> None:
        params = parse_qs(query)
        run_id = (params.get("run_id") or [""])[0]
        meta = _read_run_meta(run_id) if run_id else None
        if not meta:
            self._send_json(404, {"error": {"message": f"Unknown run_id: {run_id}"}})
            return
        self._send_json(200, {"status": "ok", "log_tail": _log_tail(str(meta.get("log_path", "")), 20000)})

    def _handle_run_file(self, query: str) -> None:
        params = parse_qs(query)
        run_id = (params.get("run_id") or [""])[0]
        rel_path = (params.get("path") or [""])[0]
        meta = _read_run_meta(run_id) if run_id else None
        if not meta:
            self._send_json(404, {"error": {"message": f"Unknown run_id: {run_id}"}})
            return
        if not rel_path:
            self._send_json(400, {"error": {"message": "Missing path"}})
            return
        try:
            output_dir = Path(str(meta.get("output_dir") or "")).expanduser().resolve()
            file_path = (output_dir / rel_path).expanduser().resolve()
            if file_path != output_dir and output_dir not in file_path.parents:
                raise ValueError("Requested file is outside the run output directory")
        except ValueError as exc:
            self._send_json(403, {"error": {"message": str(exc)}})
            return
        if PUBLIC_DEMO:
            try:
                normalized_rel = file_path.relative_to(output_dir).as_posix()
            except ValueError:
                self._send_json(403, {"error": {"message": "Requested file is outside the run output directory"}})
                return
            if normalized_rel not in _public_demo_allowed_output_rel_paths(meta, output_dir):
                self._send_json(
                    403,
                    {"error": {"message": f"Public demo only exposes the final {PUBLIC_DEMO_FINAL_OUTPUT} output."}},
                )
                return
        if not file_path.is_file():
            self._send_json(404, {"error": {"message": f"Run output file not found: {rel_path}"}})
            return
        _send_file(self, file_path)

    def _send_json(self, code: int, obj: object) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    for path in (SAVE_ROOT, RUN_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    _cleanup_old_runs()
    server = http.server.ThreadingHTTPServer((HOST, PORT), Handler)
    print(
        f"[sq-spaceflow] Listening on {HOST}:{PORT}\n"
        f"[sq-spaceflow] Service host: {socket.getfqdn()}\n"
        f"[sq-spaceflow] Allocated hosts: {', '.join(sorted(_allocated_hostnames())) or 'none'}\n"
        f"[sq-spaceflow] Launch mode: {'srun' if _should_use_srun() else 'local'}\n"
        f"[sq-spaceflow] Public demo: {PUBLIC_DEMO} max_active_runs={MAX_ACTIVE_RUNS or 'unlimited'} cleanup_retention_h={RETENTION_HOURS:g} cleanup_max_gb={MAX_STORAGE_GB:g}\n"
        f"[sq-spaceflow] Asset root: {SAVE_ROOT}\n"
        f"[sq-spaceflow] Run root: {RUN_ROOT}\n"
        f"[sq-spaceflow] Cache root: {CACHE_ROOT}\n"
        f"[sq-spaceflow] Python: {PYTHON_BIN}\n"
        f"[sq-spaceflow] Run script: {RUN_SCRIPT}\n"
        f"[sq-spaceflow] Slurm: uses_srun={_should_use_srun()} gpu_flag=--gpus={GPUS or '1'} constraint={CONSTRAINT or 'none'} exclude={EXCLUDE_NODES or 'none'} preflight={_gpu_preflight_mode()}\n"
        f"[sq-spaceflow] Cache: offline={_offline_cache_enabled()} mode={OFFLINE_CACHE_MODE}\n",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
