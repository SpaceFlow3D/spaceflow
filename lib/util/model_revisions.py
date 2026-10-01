"""Recorded source revision for TRELLIS's DINOv2 image conditioner."""

import json
from pathlib import Path
import re


MODEL_REVISIONS = Path(__file__).resolve().parents[2] / "requirements/model-revisions.json"


def dinov2_hub_repository(pins_path: Path = MODEL_REVISIONS) -> str:
    """Use an immutable upstream commit instead of PyTorch Hub's moving main."""
    record = json.loads(pins_path.read_text())["dinov2"]
    if record["repository"] != "facebookresearch/dinov2" or not re.fullmatch(r"[0-9a-f]{40}", record["revision"]):
        raise ValueError("DINOv2 requires the recorded upstream repository and a full commit SHA.")
    return f"{record['repository']}:{record['revision']}"
