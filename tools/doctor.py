#!/usr/bin/env python3
"""Check the editor or generation environment without starting a generation."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
EDITOR_MODULES = ("numpy", "scipy", "trimesh", "PIL")
GPU_MODULES = (
    "flash_attn", "kaolin", "nvdiffrast.torch", "torch_scatter",
    "spconv.pytorch", "diff_gaussian_rasterization", "vox2seq",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_environment(gpu: bool = False, cache_dir: Path | None = None, image: bool = False) -> dict:
    checks = []

    def record(name: str, passed: bool, detail: str, action: str = "") -> None:
        checks.append({"name": name, "passed": passed, "detail": detail, "action": action})

    version = sys.version_info[:2]
    record(
        "python", version == (3, 10) if gpu else (3, 10) <= version <= (3, 12),
        platform.python_version(),
        "Use Python 3.10 for generation; the editor service supports 3.10–3.12.",
    )
    for name in EDITOR_MODULES:
        try:
            module = importlib.import_module(name)
            record(name, True, str(getattr(module, "__version__", "installed")))
        except (ImportError, OSError) as exc:
            record(name, False, str(exc), "Install requirements/editor.txt in the active environment.")

    node = shutil.which("node")
    if node and not gpu:
        try:
            node_version = subprocess.check_output([node, "--version"], text=True, timeout=10).strip()
            major, minor = (int(part) for part in node_version.lstrip("v").split(".")[:2])
            supported = (major == 20 and minor >= 19) or (major == 22 and minor >= 12) or major >= 23
            record("node", supported, node_version, "Install Node.js 24 LTS, or 22.12+.")
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            record("node", False, str(exc), "Install Node.js 24 LTS, or 22.12+.")
    elif not gpu:
        record("node", False, "not found on PATH", "Install Node.js 24 LTS, or 22.12+.")
    if not gpu:
        record("npm", shutil.which("npm") is not None, shutil.which("npm") or "not found on PATH", "Install npm with Node.js.")

    if gpu:
        record("linux", platform.system() == "Linux", platform.system(), "Run generation on a Linux NVIDIA GPU machine.")
        nvcc = shutil.which("nvcc")
        if nvcc:
            try:
                output = subprocess.check_output([nvcc, "--version"], text=True, timeout=10)
                record("cuda_toolkit", "release 12.8" in output, output.strip().splitlines()[-1], "Load the CUDA 12.8 toolkit used by setup.sh.")
            except (OSError, subprocess.SubprocessError) as exc:
                record("cuda_toolkit", False, str(exc), "Load the CUDA 12.8 toolkit.")
        else:
            record("cuda_toolkit", False, "nvcc not found", "Load the CUDA 12.8 toolkit before building extensions.")
        try:
            torch = importlib.import_module("torch")
            record("pytorch", torch.__version__.split("+")[0] == "2.8.0", torch.__version__, "Run setup.sh to install pinned PyTorch 2.8.0/cu128.")
            record("pytorch_cuda", torch.version.cuda == "12.8", str(torch.version.cuda), "Install the CUDA 12.8 PyTorch wheel through setup.sh.")
            available = torch.cuda.is_available()
            record("gpu", available, torch.cuda.get_device_name(0) if available else "CUDA GPU unavailable", "On Slurm, run this check inside an allocated GPU job.")
        except (ImportError, OSError, RuntimeError) as exc:
            record("pytorch", False, str(exc), "Run SPACEFLOW_SETUP_STAGE=deps bash setup.sh.")
        for name in GPU_MODULES:
            try:
                importlib.import_module(name)
                record(name, True, "imported")
            except Exception as exc:
                record(name, False, str(exc), "Build the pinned CUDA extensions on the allocated GPU: SPACEFLOW_SETUP_STAGE=extensions bash setup.sh.")

        for name in ("sklearn", "run_local_tau"):
            try:
                # TRELLIS prints its backend on import. Keep --json output valid.
                with contextlib.redirect_stdout(io.StringIO()):
                    importlib.import_module(name)
                record(name, True, "imported")
            except Exception as exc:
                record(name, False, str(exc), "Install the pinned Python runtime with SPACEFLOW_SETUP_STAGE=deps bash setup.sh, then rebuild native extensions if the error names one.")

        blender = os.environ.get("SPACEFLOW_BLENDER_PATH") or str(REPO_ROOT / "blender-3.0.1-linux-x64/blender")
        resolved = shutil.which(blender)
        record("blender", resolved is not None, resolved or blender, "Set SPACEFLOW_BLENDER_PATH to an installed Blender 3.x executable (3.0.1 was used in the original runs).")
        checkpoint = REPO_ROOT / "third_party/PartField/models/model_objaverse.ckpt"
        pins = json.loads((REPO_ROOT / "requirements/model-revisions.json").read_text())
        if checkpoint.is_file():
            record("partfield_checkpoint", sha256(checkpoint) == pins["partfield_sha256"], str(checkpoint), "Run tools/cache_models.py; the checkpoint must match the recorded SHA256.")
        else:
            record("partfield_checkpoint", False, str(checkpoint), "Run python tools/cache_models.py --cache-dir spaceflow_runtime/huggingface.")
        cache = cache_dir or Path(os.environ.get("HF_HOME", str(REPO_ROOT / "spaceflow_runtime/huggingface")))
        marker = cache.expanduser() / "spaceflow-models.json"
        try:
            staged = json.loads(marker.read_text())
            recorded = staged.get("revisions", {})
            record("model_cache", all(recorded.get(key) == value for key, value in pins["huggingface"].items()), str(marker), "Stage the pinned models with tools/cache_models.py.")
            if image:
                image_cache = staged.get("image", {})
                record("image_cache", image_cache.get("dinov2") == pins["dinov2"] and image_cache.get("trellis_image_revision") == pins["huggingface"]["microsoft/TRELLIS-image-large"],
                       str(marker), "Stage the additional models with tools/cache_models.py --include-image.")
                torch_cache = Path(os.environ.get("TORCH_HOME", str(REPO_ROOT / "spaceflow_runtime/torch")))
                checkpoint = torch_cache / "hub/checkpoints" / pins["dinov2"]["checkpoint"]
                record("dinov2_checkpoint", checkpoint.is_file() and sha256(checkpoint) == pins["dinov2"]["sha256"],
                       str(checkpoint), "Use the same TORCH_HOME as the image staging command.")
                u2net_cache = Path(os.environ.get("U2NET_HOME", str(REPO_ROOT / "spaceflow_runtime/u2net")))
                background = u2net_cache / "u2net.onnx"
                record("u2net_checkpoint", background.is_file() and hashlib.md5(background.read_bytes()).hexdigest() == pins["u2net"]["md5"],
                       str(background), "Use the same U2NET_HOME as the image staging command.")
        except (OSError, ValueError) as exc:
            record("model_cache", False, str(exc), "Run tools/cache_models.py and set HF_HOME to the same cache directory.")

    return {
        "mode": "gpu" if gpu else "editor", "passed": all(item["passed"] for item in checks),
        "checks": checks, "scope": "Dependency preflight only; a completed GPU generation is a separate verification.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--editor", action="store_true", help="Check the CPU editor/service environment (default).")
    mode.add_argument("--gpu", action="store_true", help="Check the Linux generation environment on an allocated GPU.")
    parser.add_argument("--cache-dir", type=Path, help="The same cache directory used by tools/cache_models.py.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report.")
    parser.add_argument("--image", action="store_true", help="Also check the pinned image-conditioning cache (requires --gpu).")
    args = parser.parse_args(argv)
    if args.image and not args.gpu:
        parser.error("--image requires --gpu")
    report = inspect_environment(args.gpu, args.cache_dir, args.image)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for item in report["checks"]:
            print(f"{'OK' if item['passed'] else 'MISSING'} {item['name']}: {item['detail']}")
            if not item["passed"] and item["action"]:
                print(f"  {item['action']}")
        print(report["scope"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
