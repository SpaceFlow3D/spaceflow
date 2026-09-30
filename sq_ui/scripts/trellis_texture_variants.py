#!/usr/bin/env python3
"""Helpers for texture-focused TRELLIS experiment variants."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import torch
import utils3d

import run_local_tau
from lib.opt import self_similarity
from lib.util import generation


TRELLIS_TEXTURE_SAMPLER_PARAMS = {
    "polyak_update_tau": 0.0,
}

log = logging.getLogger(__name__)


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.2f}s"


def _ensure_text_pipeline_ready(pipeline) -> None:
    start = time.perf_counter()
    pipeline.cuda()
    run_local_tau.move_trellis_text_conditioner(pipeline, "cuda")
    log.info("Prepared TRELLIS text pipeline on CUDA in %s", _elapsed(start))


def _write_metadata(output_dir: Path, metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "variant_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.is_file():
        log.info("Skipping missing comparison asset: %s", src)
        return
    try:
        if src.resolve() == dst.resolve():
            log.info("Source and destination are identical, skipping copy: %s", src)
            return
    except Exception:
        pass
    start = time.perf_counter()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    log.info("Copied comparison asset in %s: %s -> %s", _elapsed(start), src, dst)


def _copy_dir_if_exists(src: Path, dst: Path) -> None:
    if not src.is_dir():
        log.info("Skipping missing comparison asset directory: %s", src)
        return
    try:
        if src.resolve() == dst.resolve():
            log.info("Source and destination directory are identical, skipping copy: %s", src)
            return
    except Exception:
        pass
    start = time.perf_counter()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    log.info("Copied comparison asset directory in %s: %s -> %s", _elapsed(start), src, dst)


def _copy_fixed_structure_assets(
    source_dir: Path,
    output_dir: Path,
    structure_voxels_path: Path,
    *,
    include_partfield: bool,
) -> None:
    _copy_if_exists(structure_voxels_path, output_dir / "voxels" / "struct_voxels.ply")
    _copy_if_exists(source_dir / "sample.glb", output_dir / "sample.glb")
    _copy_if_exists(source_dir / "struct_mesh.glb", output_dir / "struct_mesh.glb")
    _copy_if_exists(source_dir / "struct_mesh_zup.glb", output_dir / "struct_mesh_zup.glb")
    _copy_if_exists(source_dir / "spatial_control_mesh.ply", output_dir / "spatial_control_mesh.ply")
    _copy_if_exists(source_dir / "struct_renders" / "mesh.ply", output_dir / "struct_renders" / "mesh.ply")
    if include_partfield:
        _copy_dir_if_exists(source_dir / "partfield", output_dir / "partfield")


def _coords_from_struct_voxels(path: Path) -> torch.Tensor:
    start = time.perf_counter()
    struct_coords = utils3d.io.read_ply(str(path))[0]
    struct_coords = torch.from_numpy(struct_coords).float().cuda()
    struct_coords = ((struct_coords + 0.5) * 64).long()
    zeros = torch.zeros(
        (struct_coords.size(0), 1),
        dtype=struct_coords.dtype,
        device=struct_coords.device,
    )
    coords = torch.cat([zeros, struct_coords], dim=1).int()
    log.info(
        "Loaded fixed sparse structure in %s: %s (%d voxels)",
        _elapsed(start),
        path,
        coords.shape[0],
    )
    return coords


def _export_sparse_structure(coords: torch.Tensor, output_dir: Path) -> None:
    start = time.perf_counter()
    coords_np = coords.detach().cpu().numpy()
    filtered_coords = coords_np[:, 1:]
    run_local_tau.sparse_voxels_to_glb(
        filtered_coords,
        grid_size=64,
        output_filename=str(output_dir / "sample.glb"),
    )
    log.info(
        "Exported raw TRELLIS sparse structure mesh in %s (%d voxels)",
        _elapsed(start),
        filtered_coords.shape[0],
    )


def _decode_text_slat(
    pipeline,
    output_dir: Path,
    prompt: str,
    coords: torch.Tensor,
    *,
    seed: int,
    reset_seed: bool = True,
) -> None:
    start = time.perf_counter()
    log.info(
        "Starting TRELLIS text SLAT decode: output_dir=%s, coords=%d, reset_seed=%s",
        output_dir,
        coords.shape[0],
        reset_seed,
    )
    _ensure_text_pipeline_ready(pipeline)
    cond_start = time.perf_counter()
    cond = pipeline.get_cond_text([prompt])
    log.info("Encoded TRELLIS texture prompt in %s", _elapsed(cond_start))
    if reset_seed:
        torch.manual_seed(seed)
    slat_start = time.perf_counter()
    slat = pipeline.sample_slat(cond, coords, TRELLIS_TEXTURE_SAMPLER_PARAMS)
    log.info(
        "Sampled TRELLIS text SLAT in %s: coords=%d, feat_dim=%d",
        _elapsed(slat_start),
        slat.coords.shape[0],
        slat.feats.shape[-1],
    )
    decode_start = time.perf_counter()
    generation.decode_slat(
        pipeline,
        slat.feats,
        slat.coords,
        str(output_dir / "out_sim.glb"),
        None,
        texture=True,
    )
    log.info("Decoded textured GLB in %s: %s", _elapsed(decode_start), output_dir / "out_sim.glb")
    log.info("Completed TRELLIS text SLAT decode in %s", _elapsed(start))


def run_trellis_raw_text_variant(
    pipeline,
    output_dir: Path,
    prompt: str,
    *,
    seed: int = 1,
) -> None:
    """Run unconstrained TRELLIS text-to-3D from the flattened prompt."""
    variant_start = time.perf_counter()
    log.info("Starting raw TRELLIS flat-prompt texture variant: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(
        output_dir,
        {
            "runner": "trellis_raw_text",
            "prompt": prompt,
            "seed": seed,
        },
    )
    _ensure_text_pipeline_ready(pipeline)
    Path("debug").mkdir(exist_ok=True)
    cond_start = time.perf_counter()
    cond = pipeline.get_cond_text([prompt])
    log.info("Encoded raw TRELLIS structure prompt in %s", _elapsed(cond_start))
    torch.manual_seed(seed)
    structure_start = time.perf_counter()
    coords = pipeline.sample_sparse_structure(
        cond,
        num_samples=1,
        sampler_params=TRELLIS_TEXTURE_SAMPLER_PARAMS,
    )
    log.info(
        "Sampled raw TRELLIS sparse structure in %s (%d voxels)",
        _elapsed(structure_start),
        coords.shape[0],
    )
    _export_sparse_structure(coords, output_dir)
    _decode_text_slat(pipeline, output_dir, prompt, coords, seed=seed, reset_seed=False)
    log.info("Completed raw TRELLIS flat-prompt texture variant in %s", _elapsed(variant_start))


def run_fixed_structure_appearance_fm_variant(
    pipeline,
    output_dir: Path,
    prompt: str,
    structure_voxels_path: Path,
    *,
    seed: int = 1,
) -> None:
    """Run normal TRELLIS text SLAT sampling on a fixed SpaceFlow structure."""
    variant_start = time.perf_counter()
    log.info("Starting fixed-structure TRELLIS appearance variant: %s", output_dir)
    if not structure_voxels_path.is_file():
        raise FileNotFoundError(f"Missing source structure voxels: {structure_voxels_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(
        output_dir,
        {
            "runner": "fixed_structure_appearance_fm",
            "prompt": prompt,
            "seed": seed,
            "structure_voxels_path": str(structure_voxels_path),
        },
    )

    source_dir = structure_voxels_path.parent.parent
    _copy_fixed_structure_assets(
        source_dir,
        output_dir,
        structure_voxels_path,
        include_partfield=False,
    )

    coords = _coords_from_struct_voxels(structure_voxels_path)
    _decode_text_slat(pipeline, output_dir, prompt, coords, seed=seed)
    log.info("Completed fixed-structure TRELLIS appearance variant in %s", _elapsed(variant_start))


def run_fixed_structure_guideflow_appearance_fm_variant(
    pipeline,
    cfg,
    output_dir: Path,
    prompt: str,
    structure_voxels_path: Path,
    *,
    seed: int = 1,
    steps: int = 300,
) -> None:
    """Run GuideFlow appearance optimization on a fixed SpaceFlow structure."""
    variant_start = time.perf_counter()
    log.info("Starting fixed-structure GuideFlow appearance variant (%d steps): %s", steps, output_dir)
    if not structure_voxels_path.is_file():
        raise FileNotFoundError(f"Missing source structure voxels: {structure_voxels_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(
        output_dir,
        {
            "runner": "fixed_structure_guideflow_appearance_fm",
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "structure_voxels_path": str(structure_voxels_path),
            "guidance": "global_flat_prompt",
        },
    )

    source_dir = structure_voxels_path.parent.parent
    _copy_fixed_structure_assets(
        source_dir,
        output_dir,
        structure_voxels_path,
        include_partfield=True,
    )
    partfield_path = output_dir / "partfield" / "part_feat_mesh_batch_part_plane.npy"
    if not partfield_path.is_file():
        raise FileNotFoundError(f"Missing copied PartField features for GuideFlow variant: {partfield_path}")

    cfg.sim_guidance.steps = int(steps)
    torch.manual_seed(seed)
    self_similarity.optimize_self_similarity(
        cfg,
        prompt,
        "text",
        str(output_dir),
        local_prompts=None,
        local_prompt_type=None,
        individual_sq_meshes=None,
        generation_pipeline=pipeline,
        decode_texture=True,
    )
    log.info("Completed fixed-structure GuideFlow appearance variant in %s", _elapsed(variant_start))


def run_fixed_structure_spaceflow_routing_variant(
    pipeline,
    cfg,
    output_dir: Path,
    global_prompt: str,
    local_prompts: list[str],
    shape_path: Path,
    structure_voxels_path: Path,
    *,
    seed: int = 1,
    steps: int = 25,
    routing_mode: str = "legacy",
    routing_margin: float = 2.0,
) -> None:
    """Run SpaceFlow appearance optimization with PartField routing on a fixed SpaceFlow structure."""
    variant_start = time.perf_counter()
    log.info("Starting fixed-structure SpaceFlow routing variant: %s", output_dir)
    if not structure_voxels_path.is_file():
        raise FileNotFoundError(f"Missing source structure voxels: {structure_voxels_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(
        output_dir,
        {
            "runner": "fixed_structure_spaceflow_routing",
            "global_prompt": global_prompt,
            "local_prompts": local_prompts,
            "shape_path": str(shape_path),
            "structure_voxels_path": str(structure_voxels_path),
            "routing_mode": routing_mode,
            "routing_margin": routing_margin,
            "steps": steps,
            "seed": seed,
        },
    )

    source_dir = structure_voxels_path.parent.parent
    _copy_fixed_structure_assets(
        source_dir,
        output_dir,
        structure_voxels_path,
        include_partfield=True,
    )
    partfield_path = output_dir / "partfield" / "part_feat_mesh_batch_part_plane.npy"
    if not partfield_path.is_file():
        raise FileNotFoundError(f"Missing copied PartField features for SpaceFlow routing variant: {partfield_path}")

    cfg.sim_guidance.steps = int(steps)
    cfg.sim_guidance.local_routing_mode = routing_mode
    cfg.sim_guidance.local_routing_margin = float(routing_margin)

    individual_sq_meshes = run_local_tau.build_individual_sq_meshes_normalized(str(shape_path))
    torch.manual_seed(seed)

    self_similarity.optimize_self_similarity(
        cfg,
        global_prompt,
        "text",
        str(output_dir),
        local_prompts=local_prompts if any(local_prompts) else None,
        local_prompt_type="text" if any(local_prompts) else None,
        individual_sq_meshes=individual_sq_meshes,
        generation_pipeline=pipeline,
        decode_texture=True,
    )
    log.info("Completed fixed-structure SpaceFlow routing variant in %s", _elapsed(variant_start))

