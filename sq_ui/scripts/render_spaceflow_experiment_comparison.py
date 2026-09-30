#!/usr/bin/env python3
"""Render shared-view comparison plots for SpaceFlow experiment variants."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
from PIL import Image, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "spaceflow_runtime" / "sq_ui_runs"
SQ_HIGH_COLOR = np.array([0xf5, 0x9e, 0x0b], dtype=np.float64) / 255.0
SQ_LOW_COLOR = np.array([0xf8, 0xfa, 0xfc], dtype=np.float64) / 255.0
SQ_FALLBACK_COLOR = np.array([0.68, 0.73, 0.76], dtype=np.float64)
PANEL_RENDER_SIZE = 900
CONDITION_THUMB_SIZE = 256

STRUCTURE_VARIANTS = [
    (
        "tau-by-parts\nlow 3 / high 10",
        [
            "output/tau3_tau10_polyak0p18/out_sim_geometry.glb",
            "output/01_local_tau3_tau10_polyak0p18/out_sim_geometry.glb",
            "output/tau3_tau10_polyak0p18/out_sim_geometry.glb",
            "output/01_local_tau3_tau10_polyak0p18/out_sim_geometry.glb",
        ],
    ),
    (
        "global low tau\n3",
        [
            "output/tau3_polyak0/out_sim_geometry.glb",
            "output/02_global_tau3_polyak0/out_sim_geometry.glb",
            "output/tau3_polyak0/out_sim_geometry.glb",
            "output/02_global_tau3_polyak0/out_sim_geometry.glb",
        ],
    ),
    (
        "global high tau\n10",
        [
            "output/tau10_polyak0/out_sim_geometry.glb",
            "output/03_global_tau10_polyak0/out_sim_geometry.glb",
            "output/tau10_polyak0/out_sim_geometry.glb",
            "output/03_global_tau10_polyak0/out_sim_geometry.glb",
        ],
    ),
]

TEXTURE_VARIANTS = [
    (
        "TRELLIS raw\nflat prompt",
        [
            "output/02_trellis_raw_flat_prompt/out_sim.glb",
            "output/02_trellis_raw_flat_prompt/out_sim_geometry.glb",
        ],
        180.0,
    ),
    (
        "SpaceFlow structure\nTRELLIS appearance FM",
        [
            "output/03_fixed_structure_appearance_fm/out_sim.glb",
            "output/03_fixed_structure_appearance_fm/out_sim_geometry.glb",
        ],
    ),
    (
        "SpaceFlow structure\nGuideFlow appearance FM",
        [
            "output/04_fixed_structure_guideflow_appearance_fm/out_sim.glb",
            "output/04_fixed_structure_guideflow_appearance_fm/out_sim_geometry.glb",
        ],
    ),
    (
        "SpaceFlow\nlocal texture routing",
        [
            "output/01_spaceflow_local_texture_routing/out_sim.glb",
            "output/01_spaceflow_local_texture_routing/out_sim_geometry.glb",
        ],
    ),
]

SINGLE_RESULT_PATHS = [
    "output/out_sim.glb",
    "output/out_sim_geometry.glb",
]

SQ_RENDER_PATHS = [
    "output/spatial_control_mesh.ply",
    "output/01_spaceflow_local_texture_routing/spatial_control_mesh.ply",
    "output/01_local_tau3_tau10_polyak0p18/spatial_control_mesh.ply",
    "output/tau3_tau10_polyak0p18/spatial_control_mesh.ply",
    "output/02_global_tau3_polyak0/spatial_control_mesh.ply",
    "output/tau3_polyak0/spatial_control_mesh.ply",
    "output/03_global_tau10_polyak0/spatial_control_mesh.ply",
    "output/tau10_polyak0/spatial_control_mesh.ply",
]


def first_existing(run_dir: Path, rel_paths: list[str]) -> Path | None:
    for rel_path in rel_paths:
        path = run_dir / rel_path
        if path.is_file():
            return path
    return None


def read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run_meta(run_dir: Path) -> dict[str, object]:
    return read_json(run_dir / "run_meta.json")


def experiment_type_from_meta(meta: dict[str, object]) -> str:
    raw = str(meta.get("experiment_type") or "").strip().lower()
    if raw in {"geometry", "texture", "full"}:
        return raw
    run_config = meta.get("run_config")
    if isinstance(run_config, dict):
        raw = str(run_config.get("experimentType") or "").strip().lower()
        if raw in {"geometry", "texture", "full"}:
            return raw
    return "geometry"


def experiment_type_for_run(run_dir: Path, meta: dict[str, object] | None = None) -> str:
    meta = meta if meta is not None else run_meta(run_dir)
    kind = experiment_type_from_meta(meta)
    if kind != "geometry":
        return kind
    runner_config = read_json(run_dir / "experiment_runner_config.json")
    raw = str(runner_config.get("experiment_type") or "").strip().lower()
    return raw if raw in {"geometry", "texture", "full"} else kind


def asset_manifest(meta: dict[str, object]) -> dict[str, object]:
    asset_entry = meta.get("asset_entry")
    if not isinstance(asset_entry, dict):
        return {}
    manifest_path = asset_entry.get("manifest_path")
    if not manifest_path:
        return {}
    return read_json(Path(str(manifest_path)))


def primitive_rows(manifest: dict[str, object]) -> list[dict[str, object]]:
    primitives = manifest.get("primitives")
    if not isinstance(primitives, list):
        return []
    rows = []
    for fallback_index, primitive in enumerate(primitives):
        if not isinstance(primitive, dict):
            continue
        rows.append({
            "index": int(primitive.get("index", fallback_index)),
            "name": str(primitive.get("name") or f"SQ {fallback_index + 1}"),
            "controlLevel": "low" if primitive.get("controlLevel") == "low" else "high",
        })
    return rows


def texture_mode(meta: dict[str, object]) -> str:
    texture = meta.get("texture_guidance")
    if isinstance(texture, dict):
        mode = str(texture.get("mode") or "").strip().lower()
        if mode:
            return mode
    run_config = meta.get("run_config")
    if isinstance(run_config, dict):
        mode = str(run_config.get("textureMode") or run_config.get("appearanceMode") or "text").strip().lower()
        return "image" if mode == "image" else "text"
    return "text"


def short_condition(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "/" in text:
        return Path(text).name
    return text


def texture_conditions(meta: dict[str, object], primitive_count: int) -> tuple[str, list[str], str]:
    mode = texture_mode(meta)
    texture = meta.get("texture_guidance")
    run_config = meta.get("run_config")
    texture = texture if isinstance(texture, dict) else {}
    run_config = run_config if isinstance(run_config, dict) else {}

    if mode == "image":
        global_condition = short_condition(
            texture.get("global_image_path")
            or run_config.get("globalTextureImagePath")
            or run_config.get("appearanceImagePath")
        ) or "global image"
        local_values = texture.get("local_image_paths") or run_config.get("localTextureImagePaths") or []
    else:
        global_condition = short_condition(
            texture.get("global_text")
            or run_config.get("globalTextureText")
            or run_config.get("appearanceText")
            or run_config.get("textPrompt")
        ) or "global text"
        local_values = texture.get("local_text_prompts") or run_config.get("localTextureTexts") or []

    if not isinstance(local_values, list):
        local_values = []
    local_conditions = [short_condition(value) for value in local_values[:primitive_count]]
    if len(local_conditions) < primitive_count:
        local_conditions.extend([""] * (primitive_count - len(local_conditions)))
    return mode, local_conditions, global_condition


def resolve_condition_image_path(run_dir: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    candidates = [path] if path.is_absolute() else [run_dir / path, Path.cwd() / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _condition_image_values(meta: dict[str, object]) -> tuple[object, list[object]]:
    texture = meta.get("texture_guidance")
    run_config = meta.get("run_config")
    texture = texture if isinstance(texture, dict) else {}
    run_config = run_config if isinstance(run_config, dict) else {}

    global_value = (
        texture.get("global_image_path")
        or run_config.get("globalTextureImagePath")
        or run_config.get("appearanceImagePath")
    )
    local_values = texture.get("local_image_paths") or run_config.get("localTextureImagePaths") or []
    if not isinstance(local_values, list):
        local_values = []
    return global_value, local_values


def _condition_label(labels: list[str]) -> str:
    global_label = "Global" if "Global" in labels else ""
    sq_numbers = [label.removeprefix("SQ ") for label in labels if label.startswith("SQ ")]
    sq_label = f"SQ {', '.join(sq_numbers)}" if sq_numbers else ""
    if global_label and sq_label:
        return f"{global_label} + {sq_label}"
    return global_label or sq_label


def condition_image_tiles(run_dir: Path, meta: dict[str, object]) -> list[dict[str, object]]:
    if texture_mode(meta) != "image":
        return []

    global_value, local_values = _condition_image_values(meta)
    grouped: dict[str, dict[str, object]] = {}

    def add_tile(path: Path, label: str) -> None:
        key = str(path.resolve(strict=False))
        tile = grouped.setdefault(key, {"path": path, "labels": []})
        labels = tile["labels"]
        if isinstance(labels, list) and label not in labels:
            labels.append(label)

    global_path = resolve_condition_image_path(run_dir, global_value)
    if global_path is not None:
        add_tile(global_path, "Global")

    for index, value in enumerate(local_values):
        local_path = resolve_condition_image_path(run_dir, value)
        if local_path is not None:
            add_tile(local_path, f"SQ {index + 1}")

    tiles = []
    for tile in grouped.values():
        labels = tile.get("labels")
        path = tile.get("path")
        if not isinstance(labels, list) or not isinstance(path, Path):
            continue
        tiles.append({
            "path": path,
            "label": _condition_label([str(label) for label in labels]),
            "caption": path.name,
        })
    return tiles


def _warning_message(raw: object) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        return str(raw.get("message") or "").strip()
    return ""


def _routing_warnings_from_file(path: Path) -> list[str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = raw.get("warnings") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    return [message for message in (_warning_message(item) for item in items) if message]


def _routing_warnings_from_log(log_path: Path) -> list[str]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    pattern = re.compile(
        r"Local (?P<kind>text|image) condition\(s\) for SQ\(s\) "
        r"\[(?P<indices>[^\]]+)\] have zero routed cells"
    )
    warnings = []
    for match in pattern.finditer(text):
        indices = []
        for token in match.group("indices").split(","):
            try:
                indices.append(int(token.strip()))
            except ValueError:
                pass
        if not indices:
            continue
        sq_label = ", ".join(str(index + 1) for index in indices)
        warnings.append(
            f"Local {match.group('kind')} prompt(s) for SQs {sq_label} "
            "received zero routed cells; those local overrides had no effect."
        )
    return warnings


def routing_warning_messages(run_dir: Path) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for warning_path in sorted((run_dir / "output").rglob("routing_warnings.json")):
        for message in _routing_warnings_from_file(warning_path):
            if message not in seen:
                warnings.append(message)
                seen.add(message)
    for message in _routing_warnings_from_log(run_dir / "spaceflow.log"):
        if message not in seen:
            warnings.append(message)
            seen.add(message)
    return warnings


def _ellipsize(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _thumbnail(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    image = ImageOps.contain(image, (CONDITION_THUMB_SIZE, CONDITION_THUMB_SIZE), method=resampling)
    canvas = Image.new("RGBA", (CONDITION_THUMB_SIZE, CONDITION_THUMB_SIZE), (255, 255, 255, 255))
    offset = ((CONDITION_THUMB_SIZE - image.width) // 2, (CONDITION_THUMB_SIZE - image.height) // 2)
    canvas.alpha_composite(image, offset)
    return np.asarray(canvas.convert("RGB"), dtype=np.float64) / 255.0


def draw_condition_strip(fig: plt.Figure, tiles: list[dict[str, object]]) -> None:
    if not tiles:
        return

    fig_width, fig_height = fig.get_size_inches()
    available_width = 0.88
    gap = 0.014
    max_tile_height = 0.165
    tile_width = min(max_tile_height * fig_height / fig_width, (available_width - gap * (len(tiles) - 1)) / len(tiles))
    tile_height = tile_width * fig_width / fig_height
    total_width = tile_width * len(tiles) + gap * (len(tiles) - 1)
    x = 0.5 - total_width / 2.0
    y = 0.720

    fig.text(0.5, y + tile_height + 0.040, "Conditioning images", ha="center", va="bottom", fontsize=10.5, color="#20242a")

    for tile in tiles:
        path = tile.get("path")
        if not isinstance(path, Path):
            continue
        ax = fig.add_axes([x, y, tile_width, tile_height])
        ax.imshow(_thumbnail(path))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#20242a")
            spine.set_linewidth(0.8)
        ax.set_title(str(tile.get("label") or ""), fontsize=8.8, pad=3, color="#20242a")
        fig.text(
            x + tile_width / 2.0,
            y - 0.018,
            _ellipsize(str(tile.get("caption") or ""), 30),
            ha="center",
            va="top",
            fontsize=7.2,
            color="#4b5563",
        )
        x += tile_width + gap


def prompt_footer(run_dir: Path, meta: dict[str, object], manifest: dict[str, object]) -> str:
    primitives = primitive_rows(manifest)
    mode, local_conditions, global_condition = texture_conditions(meta, len(primitives))
    global_line = f"Global {mode} condition: {global_condition}"
    flat_prompt = ""
    texture = meta.get("texture_guidance")
    if isinstance(texture, dict):
        flat_prompt = str(texture.get("flattened_text_prompt") or "").strip()
    run_config = meta.get("run_config")
    if not flat_prompt and isinstance(run_config, dict):
        flat_prompt = str(run_config.get("flattenedTextPrompt") or "").strip()
    flat_line = (
        f"\nFlattened TRELLIS prompt: {flat_prompt}"
        if experiment_type_from_meta(meta) in {"texture", "full"} and flat_prompt
        else ""
    )
    if not primitives:
        footer = global_line + flat_line
        warning_lines = [f"Warning: {message}" for message in routing_warning_messages(run_dir)]
        return footer + ("\n" + "\n".join(warning_lines) if warning_lines else "")

    pieces = []
    for row in primitives:
        index = int(row["index"])
        local = local_conditions[index] if index < len(local_conditions) else ""
        condition = local or f"global ({global_condition})"
        pieces.append(f"{index + 1}. {row['name']} [{row['controlLevel']}]: {condition}")
    footer = global_line + "\nSQ conditions: " + "; ".join(pieces) + flat_line
    warning_lines = [f"Warning: {message}" for message in routing_warning_messages(run_dir)]
    return footer + ("\n" + "\n".join(warning_lines) if warning_lines else "")


def sq_mesh_path(run_dir: Path, meta: dict[str, object] | None = None) -> Path | None:
    for variant in experiment_variants(run_dir, meta):
        path = first_variant_file(run_dir, variant, ["spatial_control_mesh.ply"])
        if path is not None:
            return path
    return first_existing(run_dir, SQ_RENDER_PATHS)


def experiment_variants(run_dir: Path, meta: dict[str, object] | None = None) -> list[dict[str, object]]:
    meta = meta if meta is not None else run_meta(run_dir)
    raw = meta.get("experiment_variants")
    if not isinstance(raw, list):
        manifest = read_json(run_dir / "output" / "experiment_manifest.json")
        raw = manifest.get("variants")
    if not isinstance(raw, list):
        return []
    return [variant for variant in raw if isinstance(variant, dict)]


def variant_output_candidates(run_dir: Path, variant: dict[str, object]) -> list[Path]:
    candidates: list[Path] = []
    output_dir_raw = str(variant.get("output_dir") or "").strip()
    if output_dir_raw:
        output_dir = Path(output_dir_raw)
        candidates.append(output_dir if output_dir.is_absolute() else run_dir / output_dir)

    name = str(variant.get("name") or "").strip()
    if name:
        candidates.append(run_dir / "output" / name)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def first_variant_file(run_dir: Path, variant: dict[str, object], filenames: list[str]) -> Path | None:
    for output_dir in variant_output_candidates(run_dir, variant):
        for filename in filenames:
            path = output_dir / filename
            if path.is_file():
                return path
    return None


def _variant_number(variant: dict[str, object], fallback: int) -> int:
    name = str(variant.get("name") or "")
    prefix = name.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else fallback


def _geometry_variant_label(variant: dict[str, object], fallback_index: int) -> str:
    mode = str(variant.get("mode") or "")
    low_tau = variant.get("low_tau")
    high_tau = variant.get("high_tau")
    ordinal = _variant_number(variant, fallback_index)
    if mode == "local_tau":
        if high_tau is None:
            return f"tau-by-parts\nlow {_fmt_number(low_tau)}"
        try:
            if float(low_tau) == float(high_tau):
                return f"tau-by-parts\nall {_fmt_number(low_tau)}"
        except (TypeError, ValueError):
            pass
        return f"tau-by-parts\nlow {_fmt_number(low_tau)} / high {_fmt_number(high_tau)}"
    if ordinal == 2:
        return f"global low tau\n{_fmt_number(low_tau)}"
    if ordinal == 3:
        return f"global high tau\n{_fmt_number(low_tau)}"
    return f"global tau\n{_fmt_number(low_tau)}"


def _complete_paths_for_specs(run_dir: Path, specs: list[tuple]) -> list[tuple[str, Path, float]] | None:
    paths: list[tuple[str, Path, float]] = []
    for spec in specs:
        label = str(spec[0])
        rel_paths = spec[1]
        render_yaw_deg = float(spec[2]) if len(spec) > 2 else 0.0
        path = first_existing(run_dir, rel_paths)
        if path is None:
            return None
        paths.append((label, path, render_yaw_deg))
    return paths


def _complete_geometry_paths_from_meta(run_dir: Path, meta: dict[str, object]) -> list[tuple[str, Path, float]] | None:
    variants = [
        variant
        for variant in experiment_variants(run_dir, meta)
        if str(variant.get("mode") or "") in {"local_tau", "global_tau"}
    ]
    if len(variants) < 3:
        return None
    variants = variants[:3]
    paths: list[tuple[str, Path, float]] = []
    for index, variant in enumerate(variants, start=1):
        path = first_variant_file(run_dir, variant, ["out_sim_geometry.glb", "out_sim.glb"])
        if path is None:
            return None
        paths.append((_geometry_variant_label(variant, index), path, 0.0))
    return paths


def complete_experiment_paths(
    run_dir: Path,
    experiment_type: str | None = None,
) -> list[tuple[str, Path, float]] | None:
    meta = run_meta(run_dir)
    kind = (experiment_type or experiment_type_for_run(run_dir, meta)).strip().lower()
    if kind in {"texture", "full"}:
        specs = TEXTURE_VARIANTS
        return _complete_paths_for_specs(run_dir, specs)

    dynamic_paths = _complete_geometry_paths_from_meta(run_dir, meta)
    if dynamic_paths is not None:
        return dynamic_paths

    specs = STRUCTURE_VARIANTS
    return _complete_paths_for_specs(run_dir, specs)


def single_result_path(run_dir: Path) -> Path | None:
    return first_existing(run_dir, SINGLE_RESULT_PATHS)


def discover_experiments(root: Path) -> list[Path]:
    candidates = sorted(path for path in root.glob("*_experiment") if path.is_dir())
    return [path for path in candidates if complete_experiment_paths(path) is not None]


def title_for(run_dir: Path, experiment_type: str | None = None) -> str:
    meta = run_meta(run_dir)
    kind = (experiment_type or experiment_type_for_run(run_dir, meta)).strip().lower()
    run_kind = experiment_type_for_run(run_dir, meta)
    if run_kind == "full" and kind == "geometry":
        prefix = "Full structure comparison"
    elif run_kind == "full" and kind == "texture":
        prefix = "Full texture comparison"
    elif kind == "texture":
        prefix = "Texture experiment"
    elif kind == "full":
        prefix = "Full experiment"
    else:
        prefix = "Textured comparison"
    meta_path = run_dir / "run_meta.json"
    if meta_path.is_file():
        try:
            prompt = str(meta.get("run_config", {}).get("textPrompt") or "").strip()
            if prompt:
                return f"{prefix}: {prompt}"
        except json.JSONDecodeError:
            pass
    name = run_dir.name
    if "_" in name:
        name = name.split("_", 1)[1]
    return prefix + ": " + name.replace("_experiment", "").replace("_", " ")


def _fmt_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.3g}"


def single_title_for(run_dir: Path) -> str:
    meta = run_meta(run_dir)
    run_config = meta.get("run_config")
    if isinstance(run_config, dict):
        prompt = str(run_config.get("textPrompt") or "").strip()
        if prompt:
            return f"Tau-by-parts result: {prompt}"
    name = run_dir.name
    if "_" in name:
        name = name.split("_", 1)[1]
    return "Tau-by-parts result: " + name.replace("_", " ")


def tau_by_parts_label(meta: dict[str, object]) -> str:
    run_config = meta.get("run_config")
    run_config = run_config if isinstance(run_config, dict) else {}
    low_tau = run_config.get("lowTau", 3)
    high_tau = run_config.get("highTau")
    if high_tau is None:
        return f"tau-by-parts\nlow {_fmt_number(low_tau)}"
    return f"tau-by-parts\nlow {_fmt_number(low_tau)} / high {_fmt_number(high_tau)}"


def camera_rotation(azim_deg: float, elev_deg: float) -> np.ndarray:
    azim = np.deg2rad(azim_deg)
    elev = np.deg2rad(elev_deg)
    rz = np.array(
        [
            [np.cos(azim), -np.sin(azim), 0.0],
            [np.sin(azim), np.cos(azim), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elev), -np.sin(elev)],
            [0.0, np.sin(elev), np.cos(elev)],
        ]
    )
    return rx @ rz


def _material_image(mesh: trimesh.Trimesh):
    material = getattr(getattr(mesh, "visual", None), "material", None)
    if material is None:
        return None
    for attr in ("baseColorTexture", "image"):
        image = getattr(material, attr, None)
        if image is not None:
            return image
    return None


def _material_color(mesh: trimesh.Trimesh) -> np.ndarray:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    color = getattr(material, "main_color", None) if material is not None else None
    if color is None:
        return np.array([0.68, 0.73, 0.76], dtype=np.float64)
    color = np.asarray(color, dtype=np.float64)
    if color.max(initial=0.0) > 1.0:
        color = color / 255.0
    return color[:3]


def _material_texture(mesh: trimesh.Trimesh) -> np.ndarray | None:
    image = _material_image(mesh)
    if image is None:
        return None
    return np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0


def _sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width = texture.shape[:2]
    x = uv[:, 0] * (width - 1)
    y = (1.0 - uv[:, 1]) * (height - 1)

    x_floor = np.floor(x).astype(np.int64) % width
    y_floor = np.floor(y).astype(np.int64) % height
    x_ceil = np.ceil(x).astype(np.int64) % width
    y_ceil = np.ceil(y).astype(np.int64) % height

    dx = (x % width) - x_floor
    dy = (y % height) - y_floor
    dx = dx[:, None]
    dy = dy[:, None]

    colors00 = texture[y_floor, x_floor]
    colors01 = texture[y_floor, x_ceil]
    colors10 = texture[y_ceil, x_floor]
    colors11 = texture[y_ceil, x_ceil]
    return (
        colors00 * (1.0 - dx) * (1.0 - dy)
        + colors01 * dx * (1.0 - dy)
        + colors10 * (1.0 - dx) * dy
        + colors11 * dx * dy
    )


def _face_colors(mesh: trimesh.Trimesh, faces: np.ndarray) -> np.ndarray:
    visual = getattr(mesh, "visual", None)
    image = _material_texture(mesh)
    uv = getattr(visual, "uv", None)
    if image is not None and uv is not None:
        vertex_colors = _sample_texture(image, np.asarray(uv)[faces].reshape(-1, 2))
        colors = vertex_colors.reshape((-1, 3, 4)).mean(axis=1)[:, :3]
        return colors * _material_color(mesh)[None, :]

    if visual is not None and hasattr(visual, "face_colors"):
        colors = np.asarray(visual.face_colors, dtype=np.float64)
        if len(colors) == len(faces):
            if colors.max(initial=0.0) > 1.0:
                colors = colors / 255.0
            return colors[:, :3]

    if visual is not None and hasattr(visual, "vertex_colors"):
        colors = np.asarray(visual.vertex_colors, dtype=np.float64)
        if len(colors) == len(mesh.vertices):
            if colors.max(initial=0.0) > 1.0:
                colors = colors / 255.0
            return colors[faces, :3].mean(axis=1)

    return np.tile(_material_color(mesh), (len(faces), 1))


def _vertex_normals(mesh: trimesh.Trimesh, vertices: np.ndarray) -> np.ndarray:
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if normals.shape != vertices.shape:
        normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (len(vertices), 1))
    return normals


def _visual_uv(mesh: trimesh.Trimesh, vertices: np.ndarray) -> np.ndarray | None:
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if uv is None:
        return None
    uv = np.asarray(uv, dtype=np.float64)
    if len(uv) != len(vertices):
        return None
    return uv


def load_mesh(path: Path) -> dict[str, object]:
    mesh = trimesh.load(path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError(f"Empty mesh: {path}")
    texture = _material_texture(mesh)
    uv = _visual_uv(mesh, vertices) if texture is not None else None
    face_colors = _face_colors(mesh, faces)
    vertex_normals = _vertex_normals(mesh, vertices)
    vertices = vertices - (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    return {
        "vertices": vertices,
        "faces": faces,
        "face_colors": face_colors,
        "uv": uv,
        "texture": texture,
        "material_color": _material_color(mesh),
        "vertex_normals": vertex_normals,
    }


def _sq_face_colors(mesh: trimesh.Trimesh, faces: np.ndarray, primitives: list[dict[str, object]]) -> np.ndarray:
    if not primitives:
        return np.tile(SQ_FALLBACK_COLOR, (len(faces), 1))

    n_primitives = len(primitives)
    vertices_per_primitive = len(mesh.vertices) // n_primitives
    if vertices_per_primitive <= 0:
        return np.tile(SQ_FALLBACK_COLOR, (len(faces), 1))

    primitive_colors = np.array([
        SQ_LOW_COLOR if primitive.get("controlLevel") == "low" else SQ_HIGH_COLOR
        for primitive in primitives
    ])
    face_primitive_indices = np.clip(faces.min(axis=1) // vertices_per_primitive, 0, n_primitives - 1)
    return primitive_colors[face_primitive_indices]


def _sq_label_positions(vertices: np.ndarray, primitives: list[dict[str, object]]) -> list[tuple[str, np.ndarray]]:
    if not primitives:
        return []
    vertices_per_primitive = len(vertices) // len(primitives)
    labels = []
    for ordinal, primitive in enumerate(primitives):
        start = ordinal * vertices_per_primitive
        end = len(vertices) if ordinal == len(primitives) - 1 else (ordinal + 1) * vertices_per_primitive
        if start >= len(vertices) or end <= start:
            continue
        index = int(primitive.get("index", ordinal)) + 1
        labels.append((str(index), vertices[start:end].mean(axis=0)))
    return labels


def load_sq_mesh(path: Path, manifest: dict[str, object]) -> dict[str, object]:
    mesh = trimesh.load(path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError(f"Empty SQ mesh: {path}")
    primitives = primitive_rows(manifest)
    face_colors = _sq_face_colors(mesh, faces, primitives)
    label_positions = _sq_label_positions(vertices, primitives)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    vertices = vertices - center
    label_positions = [(label, position - center) for label, position in label_positions]
    return {
        "vertices": vertices,
        "faces": faces,
        "face_colors": face_colors,
        "uv": None,
        "texture": None,
        "material_color": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        "vertex_normals": _vertex_normals(mesh, np.asarray(mesh.vertices, dtype=np.float64)),
        "annotations": label_positions,
    }


def _rasterize_panel(mesh_info: dict[str, object], lim_x: float, lim_y: float, light_dir: np.ndarray) -> np.ndarray:
    resolution = PANEL_RENDER_SIZE
    image = np.ones((resolution, resolution, 3), dtype=np.float64)
    zbuffer = np.full((resolution, resolution), -np.inf, dtype=np.float64)

    vertices = np.asarray(mesh_info["vertices"], dtype=np.float64)
    faces = np.asarray(mesh_info["faces"], dtype=np.int64)
    normals = np.asarray(mesh_info["vertex_normals"], dtype=np.float64)
    face_colors = np.asarray(mesh_info["face_colors"], dtype=np.float64)
    uv = mesh_info.get("uv")
    uv = np.asarray(uv, dtype=np.float64) if uv is not None else None
    texture = mesh_info.get("texture")
    texture = np.asarray(texture, dtype=np.float64) if texture is not None else None
    material_color = np.asarray(mesh_info["material_color"], dtype=np.float64)[:3]

    screen_x = (vertices[:, 0] + lim_x) / max(2.0 * lim_x, 1e-8) * (resolution - 1)
    screen_y = (lim_y - vertices[:, 1]) / max(2.0 * lim_y, 1e-8) * (resolution - 1)
    screen_vertices = np.column_stack([screen_x, screen_y, vertices[:, 2]])

    for face_index, face in enumerate(faces):
        tri = screen_vertices[face]
        min_x = max(int(np.floor(tri[:, 0].min())), 0)
        max_x = min(int(np.ceil(tri[:, 0].max())), resolution - 1)
        min_y = max(int(np.floor(tri[:, 1].min())), 0)
        max_y = min(int(np.ceil(tri[:, 1].max())), resolution - 1)
        if min_x > max_x or min_y > max_y:
            continue

        x0, y0 = tri[0, :2]
        x1, y1 = tri[1, :2]
        x2, y2 = tri[2, :2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-10:
            continue

        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        px = xx.astype(np.float64) + 0.5
        py = yy.astype(np.float64) + 0.5
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        w2 = 1.0 - w0 - w1
        mask = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not np.any(mask):
            continue

        depth = w0 * tri[0, 2] + w1 * tri[1, 2] + w2 * tri[2, 2]
        local_zbuffer = zbuffer[min_y : max_y + 1, min_x : max_x + 1]
        update = mask & (depth > local_zbuffer)
        if not np.any(update):
            continue

        weights = np.stack([w0[update], w1[update], w2[update]], axis=1)
        if texture is not None and uv is not None:
            sample_uv = weights @ uv[face]
            colors = _sample_texture(texture, sample_uv)[:, :3] * material_color[None, :]
        else:
            colors = np.tile(face_colors[face_index], (len(weights), 1))

        pixel_normals = weights @ normals[face]
        pixel_normals /= np.maximum(np.linalg.norm(pixel_normals, axis=1, keepdims=True), 1e-8)
        intensity = np.clip(0.68 + 0.32 * np.abs(pixel_normals @ light_dir), 0.50, 1.0)
        colors = np.clip(colors * intensity[:, None], 0.0, 1.0)

        patch = image[min_y : max_y + 1, min_x : max_x + 1]
        patch[update] = colors
        local_zbuffer[update] = depth[update]

    return image


def _render_labeled_meshes(
    run_dir: Path,
    output_name: str,
    azim: float,
    elev: float,
    meshes: list[dict[str, object]],
    title: str,
    footer: str,
    condition_tiles: list[dict[str, object]],
) -> Path:
    rotation = camera_rotation(azim, elev)
    light_dir = np.array([-0.35, -0.45, 0.82])
    light_dir /= np.linalg.norm(light_dir)

    max_extent = 0.0
    for mesh_info in meshes:
        vertices = np.asarray(mesh_info["vertices"], dtype=np.float64)
        max_extent = max(max_extent, float(np.max(vertices.max(axis=0) - vertices.min(axis=0))))

    scale = 1.0 / max(max_extent, 1e-8)
    projected = []
    max_abs = np.array([0.0, 0.0])
    for mesh_info in meshes:
        render_yaw_deg = float(mesh_info.get("render_yaw_deg") or 0.0)
        mesh_rotation = camera_rotation(render_yaw_deg, 0.0) if render_yaw_deg else np.eye(3)
        vertices_camera = ((np.asarray(mesh_info["vertices"]) * scale) @ mesh_rotation.T) @ rotation.T
        vertices = vertices_camera.copy()
        vertices[:, 1] *= -1.0
        vertex_normals = (np.asarray(mesh_info["vertex_normals"], dtype=np.float64) @ mesh_rotation.T) @ rotation.T
        annotations = []
        for label, position in mesh_info["annotations"]:
            position_camera = ((position * scale) @ mesh_rotation.T) @ rotation.T
            annotations.append((
                label,
                np.array([position_camera[0], -position_camera[1], position_camera[2]], dtype=np.float64),
            ))
        projected.append({
            **mesh_info,
            "vertices": vertices,
            "vertex_normals": vertex_normals,
            "annotations": annotations,
        })
        max_abs = np.maximum(max_abs, np.max(np.abs(vertices[:, :2]), axis=0))

    lim_x = float(max_abs[0] * 1.10)
    lim_y = float(max_abs[1] * 1.16)

    fig_height = 6.9 if condition_tiles else 5.9
    fig, axes = plt.subplots(1, len(projected), figsize=(4.7 * len(projected), fig_height), dpi=220)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for ax, mesh_info in zip(axes, projected):
        label = str(mesh_info["label"])
        panel = _rasterize_panel(mesh_info, lim_x, lim_y, light_dir)
        ax.imshow(panel, extent=(-lim_x, lim_x, -lim_y, lim_y), origin="upper")
        for annotation, position in mesh_info["annotations"]:
            ax.text(
                position[0],
                position[1],
                annotation,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
                bbox={
                    "boxstyle": "circle,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 0.6,
                    "alpha": 0.82,
                },
                zorder=10,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-lim_x, lim_x)
        ax.set_ylim(-lim_y, lim_y)
        ax.axis("off")
        ax.set_title(label, fontsize=13, pad=6)

    fig.suptitle(title, fontsize=16, y=0.98)
    draw_condition_strip(fig, condition_tiles)
    footer_width = max(90, min(190, 48 * len(projected)))
    footer_wrapped = "\n".join(textwrap.wrap(footer, width=footer_width))
    fig.text(0.5, 0.045, footer_wrapped, ha="center", va="bottom", fontsize=8.5, color="#20242a")
    plt.subplots_adjust(left=0.015, right=0.985, top=0.665 if condition_tiles else 0.78, bottom=0.18, wspace=0.035)
    output_path = run_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output_path


def render_comparison(
    run_dir: Path,
    output_name: str,
    azim: float,
    elev: float,
    *,
    experiment_type: str | None = None,
) -> Path:
    variant_paths = complete_experiment_paths(run_dir, experiment_type=experiment_type)
    if variant_paths is None:
        raise RuntimeError(f"Missing one or more variants: {run_dir}")

    meta = run_meta(run_dir)
    manifest = asset_manifest(meta)

    meshes: list[dict[str, object]] = []
    control_mesh_path = sq_mesh_path(run_dir, meta)
    if control_mesh_path is not None:
        mesh_info = load_sq_mesh(control_mesh_path, manifest)
        meshes.append({
            **mesh_info,
            "label": "SQ controls\norange high / gray low",
        })

    for label, path, render_yaw_deg in variant_paths:
        mesh_info = load_mesh(path)
        meshes.append({
            **mesh_info,
            "label": label,
            "annotations": [],
            "render_yaw_deg": render_yaw_deg,
        })

    return _render_labeled_meshes(
        run_dir,
        output_name,
        azim,
        elev,
        meshes,
        title_for(run_dir, experiment_type=experiment_type),
        prompt_footer(run_dir, meta, manifest),
        condition_image_tiles(run_dir, meta),
    )


def render_single_result(run_dir: Path, output_name: str, azim: float, elev: float = 25.0) -> Path:
    result_path = single_result_path(run_dir)
    if result_path is None:
        raise RuntimeError(f"Missing tau-by-parts result: {run_dir}")

    meta = run_meta(run_dir)
    manifest = asset_manifest(meta)
    meshes: list[dict[str, object]] = []
    control_mesh_path = sq_mesh_path(run_dir)
    if control_mesh_path is not None:
        mesh_info = load_sq_mesh(control_mesh_path, manifest)
        meshes.append({
            **mesh_info,
            "label": "SQ controls\norange high / gray low",
        })

    meshes.append({
        **load_mesh(result_path),
        "label": tau_by_parts_label(meta),
        "annotations": [],
    })

    return _render_labeled_meshes(
        run_dir,
        output_name,
        azim,
        elev,
        meshes,
        single_title_for(run_dir),
        prompt_footer(run_dir, meta, manifest),
        condition_image_tiles(run_dir, meta),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", type=Path, help="Experiment run directories. Defaults to all complete experiments.")
    parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-name", default="variant_comparison.png")
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--single", action="store_true", help="Render one tau-by-parts result plus SQ controls.")
    args = parser.parse_args()

    run_dirs = args.runs or ([] if args.single else discover_experiments(args.root))
    if not run_dirs:
        raise SystemExit("No run directories provided." if args.single else "No complete experiment runs found.")

    for run_dir in run_dirs:
        try:
            if args.single:
                output_path = render_single_result(run_dir, args.output_name, args.azim, args.elev)
            else:
                variants = complete_experiment_paths(run_dir)
                if variants is None:
                    print(f"skip missing variants: {run_dir}")
                    continue
                output_path = render_comparison(run_dir, args.output_name, args.azim, args.elev)
        except Exception as exc:  # noqa: BLE001
            print(f"failed {run_dir}: {exc}")
            continue
        print(f"rendered {output_path}")


if __name__ == "__main__":
    main()
