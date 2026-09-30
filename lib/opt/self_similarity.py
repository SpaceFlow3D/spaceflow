import copy
import json
import os.path as osp
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import utils3d
import logging
import open3d_pycg as o3d
import time

import third_party.TRELLIS.trellis.modules.sparse as sp
from third_party.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from lib.util import generation, partfield

# Global logger
log = logging.getLogger(__name__)


def _select_slat_flow_model(generation_pipeline, app_type):
    """Return the SLAT flow model key/model for old and split text/image pipeline configs."""
    if app_type == 'image':
        candidates = ['slat_flow_model_image', 'slat_flow_model']
    else:
        candidates = ['slat_flow_model_text', 'slat_flow_model']

    for key in candidates:
        if key in generation_pipeline.models:
            log.info(f"Using SLAT flow model: {key}")
            return key, generation_pipeline.models[key]

    available = ', '.join(sorted(generation_pipeline.models.keys()))
    raise KeyError(
        f"No compatible SLAT flow model found for {app_type} conditioning. "
        f"Tried {candidates}; available models: {available}"
    )


def _text_conditioner_to(generation_pipeline, device):
    text_cond_model = getattr(generation_pipeline, 'text_cond_model', None)
    if not text_cond_model:
        return
    if 'model' in text_cond_model:
        text_cond_model['model'].to(device)
    if 'null_cond' in text_cond_model:
        text_cond_model['null_cond'] = text_cond_model['null_cond'].to(device)


def _preprocess_condition_image(generation_pipeline, path):
    image = Image.open(path).convert('RGB')
    if hasattr(generation_pipeline, 'preprocess_image'):
        return generation_pipeline.preprocess_image(image)
    return image


def _voxelize_mesh(mesh):
    """Voxelize an Open3D mesh into a (1,1,64,64,64) binary float32 tensor on CPU."""
    mesh = copy.deepcopy(mesh)
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    voxel_indices = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    coords_dense = torch.zeros(1, 1, 64, 64, 64, dtype=torch.float32)
    if len(voxel_indices) > 0:
        coords_dense[0, 0, voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0
    return coords_dense


def _dense_cell_groups(struct_coords, device='cuda'):
    """Return occupied 32^3 cells and the 64^3 source-voxel group for each point."""
    dense_coords = (struct_coords[:, 1:].to(device).long() // 2).clamp(0, 31)
    linear = (
        dense_coords[:, 0] * 32 * 32
        + dense_coords[:, 1] * 32
        + dense_coords[:, 2]
    )
    unique_linear, inverse = torch.unique(linear, sorted=True, return_inverse=True)
    unique_coords = torch.stack(
        [
            unique_linear // (32 * 32),
            (unique_linear // 32) % 32,
            unique_linear % 32,
        ],
        dim=1,
    )
    return unique_coords, inverse


def _surface_distance_matrix(generated_coords, sq_coords_list):
    """Distance from each generated 64^3 voxel to each SQ surface."""
    n_sq = len(sq_coords_list)
    distances = torch.full(
        (n_sq, generated_coords.shape[0]),
        float('inf'),
        dtype=generated_coords.dtype,
        device=generated_coords.device,
    )
    for idx, sq_coords in enumerate(sq_coords_list):
        if sq_coords.shape[0] > 0:
            distances[idx] = torch.cdist(sq_coords, generated_coords).min(0).values
    return distances


def _aggregate_dense_min_distances(point_distances, inverse, n_cells):
    """Pool point-to-SQ distances into occupied 32^3 cells without last-write bias."""
    n_sq = point_distances.shape[0]
    cell_distances = torch.full(
        (n_sq, n_cells),
        float('inf'),
        dtype=point_distances.dtype,
        device=point_distances.device,
    )
    for sq_idx in range(n_sq):
        cell_distances[sq_idx].scatter_reduce_(
            0,
            inverse,
            point_distances[sq_idx],
            reduce='amin',
            include_self=True,
        )
    return cell_distances


def _aggregate_dense_votes(point_labels, inverse, n_cells, n_labels):
    """Pool per-point labels into 32^3 cells by majority vote."""
    votes = torch.zeros(
        (n_cells, n_labels),
        dtype=torch.int32,
        device=point_labels.device,
    )
    votes.index_put_(
        (inverse, point_labels.long()),
        torch.ones_like(point_labels, dtype=torch.int32),
        accumulate=True,
    )
    return votes.argmax(dim=1)


def compute_active_margin_indices(
    struct_coords,
    individual_sq_meshes,
    active_indices,
    device='cuda',
    margin=2.0,
):
    """Route active local conditions to nearby occupied cells with a distance margin.

    The legacy router assigns a cell only to the single nearest SQ.  This is brittle
    when the generated structure shifts a thin part by a few 64^3 voxels.  Here an
    active SQ may claim a cell when it is within ``margin`` voxels of the best SQ;
    competing active SQs are resolved by their distance to the cell.  The routing is
    still restricted to occupied generated cells, so the margin cannot create texture
    in empty space.
    """
    coords_dense_indices = torch.zeros(
        1, 1, 32, 32, 32, dtype=torch.int32, device=device
    )
    n_sq = len(individual_sq_meshes)
    active_indices = [idx for idx in active_indices if 0 <= idx < n_sq]
    if n_sq == 0 or not active_indices or struct_coords.shape[0] == 0:
        return coords_dense_indices

    generated_coords = struct_coords[:, 1:].float().to(device)
    sq_coords_list = []
    for mesh in individual_sq_meshes:
        sq_vox = _voxelize_mesh(mesh)
        sq_coords = torch.argwhere(sq_vox)[:, 2:].float().to(device)
        sq_coords_list.append(sq_coords)

    dense_coords, inverse = _dense_cell_groups(struct_coords, device=device)
    point_distances = _surface_distance_matrix(generated_coords, sq_coords_list)
    cell_distances = _aggregate_dense_min_distances(
        point_distances,
        inverse,
        dense_coords.shape[0],
    )

    active = torch.as_tensor(active_indices, dtype=torch.long, device=device)
    active_distances = cell_distances.index_select(0, active)
    best_active_distances, best_active_pos = active_distances.min(dim=0)
    best_all_distances = cell_distances.min(dim=0).values
    selected = best_active_distances <= (best_all_distances + float(margin))
    if selected.any():
        selected_coords = dense_coords[selected]
        selected_sq = active[best_active_pos[selected]] + 1
        coords_dense_indices[
            0,
            0,
            selected_coords[:, 0],
            selected_coords[:, 1],
            selected_coords[:, 2],
        ] = selected_sq.to(torch.int32)
    return coords_dense_indices


def compute_surface_coverage_indices(
    struct_coords,
    individual_sq_meshes,
    active_indices,
    device='cuda',
    min_cells=64,
    max_cells=512,
    candidate_margin=3.0,
):
    """Route local conditions from exact SQ surface coverage.

    PartField and nearest-SQ routing can lose thin or small parts after the
    structure is pooled into the 32^3 SLAT grid.  This router first reserves
    cells touched by the exact 64^3 SQ surface voxelization, then fills only a
    compact nearest-cell budget for each active SQ.  Small details therefore
    receive a guaranteed local condition without a broad, unconstrained
    dilation into neighboring parts.

    Values are SQ indices (1-indexed; zero means global/unassigned), matching
    ``compute_coords_dense_indices`` and the later condition remapping.
    """
    coords_dense_indices = torch.zeros(
        1, 1, 32, 32, 32, dtype=torch.int32, device=device
    )
    n_sq = len(individual_sq_meshes)
    active_indices = [idx for idx in active_indices if 0 <= idx < n_sq]
    min_cells = max(1, int(min_cells))
    max_cells = max(min_cells, int(max_cells))
    if n_sq == 0 or not active_indices or struct_coords.shape[0] == 0:
        return coords_dense_indices, {
            'routing_source': 'surface_coverage',
            'exact_surface_cells': {},
            'reserved_cells': {},
        }

    generated_coords = struct_coords[:, 1:].float().to(device)
    dense_coords, inverse = _dense_cell_groups(struct_coords, device=device)
    n_cells = int(dense_coords.shape[0])
    if n_cells == 0:
        return coords_dense_indices, {
            'routing_source': 'surface_coverage',
            'exact_surface_cells': {},
            'reserved_cells': {},
        }

    sq_coords_list = []
    for mesh in individual_sq_meshes:
        sq_vox = _voxelize_mesh(mesh)
        sq_coords_list.append(torch.argwhere(sq_vox)[:, 2:].float().to(device))
    point_distances = _surface_distance_matrix(generated_coords, sq_coords_list)
    cell_distances = _aggregate_dense_min_distances(
        point_distances,
        inverse,
        n_cells,
    )

    dense_linear = (
        dense_coords[:, 0] * 32 * 32
        + dense_coords[:, 1] * 32
        + dense_coords[:, 2]
    ).long()
    assigned = torch.zeros(n_cells, dtype=torch.int32, device=device)
    exact_counts = {}
    targets = {}
    candidate_orders = {}

    # Build a compact candidate list for each active SQ. Exact surface cells
    # are always preferred; the nearest cells fill the minimum budget only
    # when the detail is too small or shifted relative to the structure grid.
    for sq_idx in active_indices:
        surface = sq_coords_list[sq_idx]
        if surface.shape[0] > 0:
            surface_dense = (surface.long() // 2).clamp(0, 31)
            surface_linear = (
                surface_dense[:, 0] * 32 * 32
                + surface_dense[:, 1] * 32
                + surface_dense[:, 2]
            ).long()
            positions = torch.searchsorted(dense_linear, surface_linear)
            valid = positions < n_cells
            safe_positions = positions.clamp(max=n_cells - 1)
            valid = valid & (dense_linear[safe_positions] == surface_linear)
            exact = torch.unique(safe_positions[valid])
        else:
            exact = torch.empty(0, dtype=torch.long, device=device)

        exact_counts[sq_idx] = int(exact.numel())
        target = min(max_cells, max(min_cells, int(exact.numel())))
        targets[sq_idx] = target
        order = torch.argsort(cell_distances[sq_idx], stable=True)
        if candidate_margin > 0 and order.numel() > 0:
            best = cell_distances[sq_idx, order[0]]
            near = order[cell_distances[sq_idx, order] <= best + float(candidate_margin)]
            if near.numel() >= min_cells:
                order = near
        if exact.numel() > 0:
            order = torch.cat([exact, order])
            order = torch.unique(order, sorted=False)
        candidate_orders[sq_idx] = order[: min(max_cells, int(order.numel()))]

    # Reserve compact regions for the smallest details first. This prevents a
    # large neighboring part from consuming all cells needed by a tiny part.
    priority = sorted(active_indices, key=lambda idx: (exact_counts[idx], idx))
    for sq_idx in priority:
        order = candidate_orders[sq_idx]
        available = order[assigned[order] == 0]
        selected = available[: targets[sq_idx]]
        if selected.numel() < targets[sq_idx]:
            # Overlapping SQs may have no fully disjoint candidate budget. A
            # detail still gets its nearest cells, with later assignments
            # intentionally unable to overwrite this reservation.
            selected = order[: targets[sq_idx]]
        assigned[selected] = int(sq_idx + 1)

    coords_dense_indices[
        0,
        0,
        dense_coords[:, 0],
        dense_coords[:, 1],
        dense_coords[:, 2],
    ] = assigned
    reserved_counts = {
        sq_idx: int((assigned == sq_idx + 1).sum().item())
        for sq_idx in active_indices
    }
    return coords_dense_indices, {
        'routing_source': 'surface_coverage',
        'exact_surface_cells': exact_counts,
        'reserved_cells': reserved_counts,
        'target_cells': targets,
    }


def compute_coords_dense_indices(struct_coords, individual_sq_meshes, device='cuda', vox_cluster_labels=None):
    """Map each 64^3 voxel to its nearest superquadric (1-indexed; 0=unassigned).

    When vox_cluster_labels is provided (PartField cluster label per voxel), assigns
    voxels via cluster centroids: each cluster centroid is matched to the nearest SQ,
    and all voxels in that cluster inherit the SQ assignment. This is more accurate
    than per-voxel geometric nearest neighbor because PartField clusters capture
    semantic part structure.

    struct_coords: (M, 4) int tensor [batch, x, y, z] in 64x64x64 space.
    vox_cluster_labels: (M,) int tensor of PartField cluster indices per voxel.
    Returns a (1, 1, 32, 32, 32) int32 tensor.
    """
    coords_dense_indices = torch.zeros(1, 1, 32, 32, 32, dtype=torch.int32, device=device)
    n_sq = len(individual_sq_meshes)
    if n_sq == 0:
        return coords_dense_indices

    generated_coords = struct_coords[:, 1:].float().to(device)  # (M, 3)

    # Precompute SQ surface voxel coords once
    sq_coords_list = []
    for mesh in individual_sq_meshes:
        sq_vox = _voxelize_mesh(mesh)  # (1,1,64,64,64)
        sq_c = torch.argwhere(sq_vox)[:, 2:].float().to(device)  # (K,3)
        sq_coords_list.append(sq_c)

    if vox_cluster_labels is not None:
        # PartField-based assignment: cluster centroid → nearest SQ → voxels
        labels = vox_cluster_labels.to(device)
        num_clusters = int(labels.max().item()) + 1
        cluster_to_sq = torch.zeros(num_clusters, dtype=torch.long, device=device)
        for c in range(num_clusters):
            mask = labels == c
            if not mask.any():
                continue
            cluster_voxels = generated_coords[mask]
            mean_pos = cluster_voxels.mean(0, keepdim=True)
            centroid = cluster_voxels[torch.cdist(cluster_voxels, mean_pos).argmin()].unsqueeze(0)  # (1, 3) actual voxel
            min_dists = [
                torch.cdist(sq_c, centroid).min().item() if sq_c.shape[0] > 0 else float('inf')
                for sq_c in sq_coords_list
            ]
            cluster_to_sq[c] = int(np.argmin(min_dists))
        point_labels = cluster_to_sq[labels]
        dense_coords, inverse = _dense_cell_groups(struct_coords, device=device)
        dense_labels = _aggregate_dense_votes(
            point_labels,
            inverse,
            dense_coords.shape[0],
            n_sq,
        )
    else:
        # Fallback: per-voxel geometric nearest neighbor
        min_distances = _surface_distance_matrix(generated_coords, sq_coords_list)
        dense_coords, inverse = _dense_cell_groups(struct_coords, device=device)
        cell_distances = _aggregate_dense_min_distances(
            min_distances,
            inverse,
            dense_coords.shape[0],
        )
        dense_labels = cell_distances.argmin(dim=0)

    coords_dense_indices[
        0,
        0,
        dense_coords[:, 0],
        dense_coords[:, 1],
        dense_coords[:, 2],
    ] = (dense_labels + 1).to(torch.int32)

    return coords_dense_indices

def _dense_sq_counts(coords_dense_indices, n_sq):
    flat = coords_dense_indices.reshape(-1)
    return {
        sq_idx: int((flat == sq_idx + 1).sum().item())
        for sq_idx in range(n_sq)
    }

def _dense_condition_counts(coords_dense_indices, n_conditions):
    flat = coords_dense_indices.reshape(-1)
    return {
        cond_idx: int((flat == cond_idx).sum().item())
        for cond_idx in range(1, n_conditions)
    }

def _active_route_coverage(sq_counts, active_indices):
    counts = [sq_counts.get(idx, 0) for idx in active_indices]
    return sum(count > 0 for count in counts), sum(counts)

def _format_nonzero_counts(counts):
    nonzero = {idx: count for idx, count in counts.items() if count > 0}
    return nonzero if nonzero else {}

def _write_routing_warning(output_dir, warning):
    warning_path = osp.join(output_dir, "routing_warnings.json")
    try:
        if osp.isfile(warning_path):
            with open(warning_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        else:
            payload = {"warnings": []}
        warnings = payload.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
            payload["warnings"] = warnings
        warnings.append(warning)
        with open(warning_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not write routing warning metadata: %s", exc)

def _choose_local_routing(
    struct_coords,
    individual_sq_meshes,
    struct_labels,
    active_indices,
    device='cuda',
    routing_mode='legacy',
    routing_margin=2.0,
    surface_min_cells=64,
    surface_max_cells=512,
    surface_candidate_margin=3.0,
):
    if routing_mode in {'surface_coverage', 'surface_topk'} and active_indices:
        surface_routed, surface_stats = compute_surface_coverage_indices(
            struct_coords,
            individual_sq_meshes,
            active_indices,
            device=device,
            min_cells=surface_min_cells,
            max_cells=surface_max_cells,
            candidate_margin=surface_candidate_margin,
        )
        surface_counts = _dense_sq_counts(surface_routed, len(individual_sq_meshes))
        log.info(
            "Surface-coverage local routing (min=%d, max=%d, margin=%.2f) SQ counts: %s; exact cells: %s",
            int(surface_min_cells),
            int(surface_max_cells),
            float(surface_candidate_margin),
            _format_nonzero_counts(surface_counts),
            surface_stats.get('exact_surface_cells', {}),
        )
        return surface_routed, 'surface_coverage', surface_counts

    if routing_mode == 'active_margin' and active_indices:
        margin_routed = compute_active_margin_indices(
            struct_coords,
            individual_sq_meshes,
            active_indices,
            device=device,
            margin=routing_margin,
        )
        margin_counts = _dense_sq_counts(margin_routed, len(individual_sq_meshes))
        log.info(
            "Active-margin local routing (margin=%.2f) SQ counts: %s",
            routing_margin,
            _format_nonzero_counts(margin_counts),
        )
        return margin_routed, 'active_margin', margin_counts

    clustered = compute_coords_dense_indices(
        struct_coords, individual_sq_meshes, device, vox_cluster_labels=struct_labels)
    n_sq = len(individual_sq_meshes)
    clustered_counts = _dense_sq_counts(clustered, n_sq)
    log.info(f"PartField local routing SQ counts: {_format_nonzero_counts(clustered_counts)}")

    if not active_indices:
        return clustered, "partfield", clustered_counts

    clustered_coverage = _active_route_coverage(clustered_counts, active_indices)
    if clustered_coverage[0] == len(active_indices):
        return clustered, "partfield", clustered_counts

    geometric = compute_coords_dense_indices(
        struct_coords, individual_sq_meshes, device, vox_cluster_labels=None)
    geometric_counts = _dense_sq_counts(geometric, n_sq)
    geometric_coverage = _active_route_coverage(geometric_counts, active_indices)
    log.info(f"Geometric local routing SQ counts: {_format_nonzero_counts(geometric_counts)}")

    if geometric_coverage > clustered_coverage:
        missing = [idx for idx in active_indices if clustered_counts.get(idx, 0) == 0]
        log.warning(
            "PartField local routing missed active SQ(s) %s; using geometric routing "
            "for local texture conditions instead.",
            missing,
        )
        return geometric, "geometric", geometric_counts

    missing = [idx for idx in active_indices if clustered_counts.get(idx, 0) == 0]
    if missing:
        log.warning(
            "PartField local routing missed active SQ(s) %s, and geometric routing "
            "did not improve coverage; keeping PartField routing.",
            missing,
        )
    return clustered, "partfield", clustered_counts

def _dilate_condition_indices(coords_dense_indices, n_conditions, radius=1):
    if radius <= 0 or n_conditions <= 1:
        return coords_dense_indices

    dilated_indices = coords_dense_indices.clone()
    kernel_size = 2 * radius + 1
    for cond_idx in range(1, n_conditions):
        mask = (coords_dense_indices == cond_idx).float()
        dilated = F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=radius) > 0
        dilated_indices[(dilated_indices == 0) & dilated] = cond_idx
    return dilated_indices

def _nonnegative_int(value, default=0):
    if value is None:
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default

def _select_compact_candidate_cells(mask, candidates, needed):
    if needed <= 0:
        return candidates[:0]
    candidate_coords = candidates.nonzero(as_tuple=False)
    if candidate_coords.shape[0] <= needed:
        return candidate_coords

    kernel = torch.ones((1, 1, 3, 3, 3), dtype=mask.dtype, device=mask.device)
    support = F.conv3d(mask, kernel, stride=1, padding=1)[candidates]
    source_coords = mask.nonzero(as_tuple=False)[:, 2:].float()
    centroid = source_coords.mean(dim=0, keepdim=True)
    spatial = candidate_coords[:, 2:].float()
    dist2 = ((spatial - centroid) ** 2).sum(dim=1)
    linear = (
        candidate_coords[:, 2] * mask.shape[-2] * mask.shape[-1]
        + candidate_coords[:, 3] * mask.shape[-1]
        + candidate_coords[:, 4]
    ).float()
    priority = support.float() * 1_000_000.0 - dist2 - linear / 1_000_000.0
    selected = torch.topk(priority, k=needed, largest=True, sorted=False).indices
    return candidate_coords[selected]

def _expand_small_condition_indices(coords_dense_indices, n_conditions, min_cells=0, max_dilation=0):
    """Grow under-covered local conditions into global cells.

    The local routing grid is 32^3, so small SQs such as wheels can collapse to
    very few routed cells. This keeps local prompts active by expanding only
    conditions that are below min_cells, never overwriting another local prompt.
    """
    min_cells = _nonnegative_int(min_cells, 0)
    max_dilation = _nonnegative_int(max_dilation, 0)
    if min_cells <= 0 or max_dilation <= 0 or n_conditions <= 1:
        return coords_dense_indices, {}

    expanded_indices = coords_dense_indices.clone()
    stats = {}
    for cond_idx in range(1, n_conditions):
        initial_count = int((expanded_indices == cond_idx).sum().item())
        stats[cond_idx] = {
            "initial": initial_count,
            "final": initial_count,
            "dilation": 0,
            "added": 0,
        }
        if initial_count <= 0 or initial_count >= min_cells:
            continue

        count = initial_count
        for radius in range(1, max_dilation + 1):
            mask = (expanded_indices == cond_idx).float()
            grown = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1) > 0
            candidates = grown & (expanded_indices == 0)
            if not candidates.any():
                break
            needed = min_cells - count
            selected = _select_compact_candidate_cells(mask, candidates, needed)
            expanded_indices[
                selected[:, 0],
                selected[:, 1],
                selected[:, 2],
                selected[:, 3],
                selected[:, 4],
            ] = cond_idx
            count = int((expanded_indices == cond_idx).sum().item())
            stats[cond_idx]["final"] = count
            stats[cond_idx]["dilation"] = radius
            stats[cond_idx]["added"] = count - initial_count
            if count >= min_cells:
                break

    return expanded_indices, stats

def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix

def _auto_contrastive_chunk_size(num_voxels):
    if num_voxels <= 16_000:
        return 1024
    if num_voxels <= 28_000:
        return 512
    return 256


def chunked_contrastive_loss(feats, labels, eps=1e-08, chunk_size=None):
    """Supervised contrastive loss without materializing the full N×N similarity matrix.
    Peak memory is O(chunk_size × N) instead of O(N²).
    """
    # feats: (1, 1, N, C) -> (N, C)
    x = feats[0, 0]
    x_norm = x / x.norm(dim=1, keepdim=True).clamp(min=eps)  # (N, C), L2-normalized
    labels = labels.view(-1)  # (N,)
    N = x_norm.shape[0]
    chunk_size = chunk_size or _auto_contrastive_chunk_size(N)

    numerator = torch.zeros(N, device=x.device)
    denominator = torch.zeros(N, device=x.device)
    valid = torch.zeros(N, dtype=torch.bool, device=x.device)

    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        sim_row = x_norm[i:i_end] @ x_norm.T  # (cs, N)
        exp_sim = torch.exp(sim_row)

        labels_i = labels[i:i_end].unsqueeze(1)  # (cs, 1)
        same_label = labels_i == labels.unsqueeze(0)  # (cs, N)

        # Exclude self-similarity on diagonal
        diag_idx = torch.arange(i_end - i, device=x.device)
        self_mask = torch.zeros_like(same_label)
        self_mask[diag_idx, i + diag_idx] = True
        positive_mask = same_label & ~self_mask

        denominator[i:i_end] = exp_sim.masked_fill(self_mask, 0.0).sum(dim=1)
        valid[i:i_end] = positive_mask.any(dim=1)
        numerator[i:i_end] = exp_sim.masked_fill(~positive_mask, 0.0).sum(dim=1)

    loss = -torch.log(numerator / (denominator + 1e-8))
    return loss[valid].mean()

def optimize_self_similarity(cfg, app, app_type, output_dir,
                             local_prompts=None, local_prompt_type=None,
                             individual_sq_meshes=None,
                             generation_pipeline=None,
                             decode_texture=True):
    overall_start = time.perf_counter()
    log.info("Starting self-similarity optimization...")

    pipeline_start = time.perf_counter()
    if generation_pipeline is None:
        if app_type == 'image':
            generation_pipeline = TrellisImageTo3DPipeline.from_pretrained(cfg.trellis_img_model_name)
        else:
            generation_pipeline = TrellisTextTo3DPipeline.from_pretrained(cfg.trellis_text_model_name)
    else:
        log.info("Reusing preloaded TRELLIS pipeline for self-similarity guidance.")
    generation_pipeline.cuda()
    _text_conditioner_to(generation_pipeline, 'cuda')
    log.info("Prepared TRELLIS pipeline for self-similarity in %.2fs", time.perf_counter() - pipeline_start)

    if app_type == 'image':
        app = _preprocess_condition_image(generation_pipeline, osp.join(output_dir, 'app_image.png'))
    
    # Load Structure Data
    structure_start = time.perf_counter()
    struct_coords = utils3d.io.read_ply(osp.join(output_dir, 'voxels', 'struct_voxels.ply'))[0]
    struct_coords = torch.from_numpy(struct_coords).float().cuda()
    struct_coords = ((struct_coords + 0.5) * 64).long()
    
    zeros = torch.zeros((struct_coords.size(0), 1), dtype=struct_coords.dtype, device=struct_coords.device)
    struct_coords = torch.cat([zeros, struct_coords], dim=1)
    
    # Load partfield planes (extracted from Blender's struct_renders/mesh.ply — same mesh as voxels)
    path = osp.join(output_dir, "partfield", "part_feat_mesh_batch_part_plane.npy")
    struct_part_planes = torch.from_numpy(np.load(path, allow_pickle=True)).cuda()

    struct_labels = partfield.cluster_geoms(struct_coords, struct_part_planes, num_clusters=cfg.sim_guidance.num_part_clusters)

    # Optimization Starts...
    struct_labels = torch.from_numpy(struct_labels.flatten()).cuda()
    struct_feats_params = torch.nn.Parameter(torch.randn((struct_coords.shape[0], cfg.flow_model_in_channels)), requires_grad=True)
    log.info(
        "Loaded structure voxels, PartField planes, and labels in %.2fs (%d voxels)",
        time.perf_counter() - structure_start,
        struct_coords.shape[0],
    )
    
    param_list = [struct_feats_params]
    optimizer = torch.optim.AdamW(param_list, lr=cfg.sim_guidance.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
    
    best_loss = float('inf')
    feats = None
    
    cond_start = time.perf_counter()
    if app_type == "image":
        cond = generation_pipeline.get_cond([app]) if hasattr(generation_pipeline, 'get_cond') else generation_pipeline.get_cond_image([app])
    else:
        cond = generation_pipeline.get_cond_text([app])
    log.info("Encoded global %s guidance condition in %.2fs", app_type, time.perf_counter() - cond_start)

    # Build per-SQ local conditioning if local prompts were provided.
    cond_list = None
    coords_dense_indices = None
    if local_prompts and individual_sq_meshes:
        local_start = time.perf_counter()
        global_emb = cond['cond']  # (1, seq_len, dim)
        n_sq = len(local_prompts)
        active_indices = [
            idx
            for idx, prompt in enumerate(local_prompts)
            if str(prompt or "").strip()
        ]
        log.info(
            f"Building local {local_prompt_type} conditioning for "
            f"{len(active_indices)} / {n_sq} superquadrics; active SQ indices: {active_indices}"
        )
        if local_prompt_type == 'text':
            local_embs = []
            for sq_idx, prompt in enumerate(local_prompts):
                prompt = str(prompt or "").strip()
                if prompt:
                    log.info(
                        "Local text condition for SQ %d: %r",
                        sq_idx,
                        prompt,
                    )
                local_embs.append(
                    generation_pipeline.encode_text([prompt])
                    if prompt
                    else global_emb
                )
        else:  # 'image'
            local_embs = []
            for sq_idx, path in enumerate(local_prompts):
                path = str(path or '').strip()
                if path:
                    log.info(
                        "Local image condition for SQ %d: %s",
                        sq_idx,
                        path,
                    )
                    local_image = _preprocess_condition_image(generation_pipeline, path)
                    local_embs.append(generation_pipeline.encode_image([local_image]))
                else:
                    local_embs.append(global_emb)
        # Only include SQs that have a real (non-global-fallback) local condition.
        conditioned_mask = [emb is not global_emb for emb in local_embs]
        conditioned_embs = [emb for emb, is_cond in zip(local_embs, conditioned_mask) if is_cond]
        cond_list = [global_emb] + conditioned_embs

        # Build remap: SQ 1-index → new condition index (0=global fallback, 1..n_conditioned).
        remap = torch.zeros(n_sq + 1, dtype=torch.long, device='cuda')
        new_idx = 1
        condition_to_sq = {}
        for sq_i, is_cond in enumerate(conditioned_mask):
            remap[sq_i + 1] = new_idx if is_cond else 0
            if is_cond:
                condition_to_sq[new_idx] = sq_i
                new_idx += 1
        sq_cond_map = remap[1:].tolist()  # condition index per SQ (0=global)
        log.info(f"SQ condition mapping (0=global): {sq_cond_map}")
        log.info(f"Built cond_list with {len(cond_list)} entries "
                 f"(1 global + {len(conditioned_embs)} real local out of {n_sq} SQs)")

        local_routing_mode = str(
            getattr(cfg.sim_guidance, 'local_routing_mode', 'legacy')
        ).strip().lower()
        local_routing_margin = float(
            getattr(cfg.sim_guidance, 'local_routing_margin', 2.0)
        )
        surface_min_cells = _nonnegative_int(
            getattr(cfg.sim_guidance, 'surface_min_cells', 64), 64
        )
        surface_max_cells = _nonnegative_int(
            getattr(cfg.sim_guidance, 'surface_max_cells', 512), 512
        )
        surface_candidate_margin = float(
            getattr(cfg.sim_guidance, 'surface_candidate_margin', 3.0)
        )
        coords_dense_indices, routing_source, sq_route_counts = _choose_local_routing(
            struct_coords,
            individual_sq_meshes,
            struct_labels,
            active_indices,
            device='cuda',
            routing_mode=local_routing_mode,
            routing_margin=local_routing_margin,
            surface_min_cells=surface_min_cells,
            surface_max_cells=surface_max_cells,
            surface_candidate_margin=surface_candidate_margin,
        )
        coords_dense_indices = remap[coords_dense_indices.long()]
        condition_counts_before_dilation = _dense_condition_counts(coords_dense_indices, len(cond_list))
        local_condition_dilation = _nonnegative_int(
            getattr(cfg.sim_guidance, 'local_condition_dilation', 0), 0)
        coords_dense_indices = _dilate_condition_indices(
            coords_dense_indices, len(cond_list), radius=local_condition_dilation)
        condition_counts_after_dilation = _dense_condition_counts(coords_dense_indices, len(cond_list))
        local_condition_min_cells = _nonnegative_int(
            getattr(cfg.sim_guidance, 'local_condition_min_cells', 0), 0)
        local_condition_max_dilation = _nonnegative_int(
            getattr(cfg.sim_guidance, 'local_condition_max_dilation', 0), 0)
        coords_dense_indices, adaptive_expansion_stats = _expand_small_condition_indices(
            coords_dense_indices,
            len(cond_list),
            min_cells=local_condition_min_cells,
            max_dilation=local_condition_max_dilation,
        )
        condition_counts = _dense_condition_counts(coords_dense_indices, len(cond_list))
        missing_condition_sqs = [
            condition_to_sq.get(cond_idx, 'unknown')
            for cond_idx in range(1, len(cond_list))
            if condition_counts.get(cond_idx, 0) == 0
        ]
        if missing_condition_sqs:
            missing_sq_numbers = [
                sq_idx + 1 if isinstance(sq_idx, int) else sq_idx
                for sq_idx in missing_condition_sqs
            ]
            sq_label = ", ".join(str(sq_num) for sq_num in missing_sq_numbers)
            warning_message = (
                f"Local {local_prompt_type} prompt(s) for SQs {sq_label} "
                "received zero routed cells; those local overrides had no effect."
            )
            log.warning(
                "Local %s condition(s) for SQ(s) %s have zero routed cells; "
                "those local overrides will have no effect. Check condition_routing.mp4 "
                "or enlarge/adjust the selected superquadric.",
                local_prompt_type,
                missing_condition_sqs,
            )
            _write_routing_warning(
                output_dir,
                {
                    "kind": "local_routing_zero_cells",
                    "severity": "warning",
                    "message": warning_message,
                    "local_prompt_type": local_prompt_type,
                    "sq_indices": [
                        sq_idx for sq_idx in missing_condition_sqs
                        if isinstance(sq_idx, int)
                    ],
                    "sq_numbers": missing_sq_numbers,
                    "routing_source": routing_source,
                    "active_sq_counts": {
                        str(idx + 1): int(sq_route_counts.get(idx, 0))
                        for idx in active_indices
                    },
                    "condition_counts": {
                        str(cond_idx): int(count)
                        for cond_idx, count in condition_counts.items()
                    },
                },
            )
        active_sq_counts = {idx: sq_route_counts.get(idx, 0) for idx in active_indices}
        log.info(
            f"Local routing source: {routing_source}; active SQ counts: {active_sq_counts}"
        )
        if adaptive_expansion_stats:
            active_expansions = {
                f"cond_{cond_idx}_sq_{condition_to_sq.get(cond_idx, 'unknown')}": values
                for cond_idx, values in adaptive_expansion_stats.items()
                if values["dilation"] > 0 or values["initial"] < local_condition_min_cells
            }
            if active_expansions:
                log.info(
                    "Adaptive local condition expansion: min_cells=%d, max_dilation=%d, stats=%s",
                    local_condition_min_cells,
                    local_condition_max_dilation,
                    active_expansions,
                )
        log.info(
            f"coords_dense_indices: shape={coords_dense_indices.shape}, "
            f"condition_counts_before_dilation={condition_counts_before_dilation}, "
            f"condition_counts_after_dilation={condition_counts_after_dilation}, "
            f"dilation_radius={local_condition_dilation}, "
            f"adaptive_min_cells={local_condition_min_cells}, "
            f"adaptive_max_dilation={local_condition_max_dilation}, "
            f"condition_counts={condition_counts}, non-zero={int((coords_dense_indices > 0).sum())}"
        )

        log.info("Skipping condition routing visualization; routing indices are still used for local conditioning.")
        torch.cuda.empty_cache()
        log.info("Prepared local condition routing in %.2fs", time.perf_counter() - local_start)

        # condition routing visualization for routing
        log.info("Visualizing condition routing on structure mesh...")
        try:
            import trimesh
            from lib.util.visualization import visualize_and_save, map_voxel_labels_to_vertices
            _mesh_path = osp.join(output_dir, 'struct_renders', 'mesh.ply')
            if not osp.isfile(_mesh_path):
                _mesh_path = osp.join(output_dir, 'spatial_control_mesh.ply')
            if osp.isfile(_mesh_path):
                _sv_norm = ((struct_coords[:, 1:].float() + 0.5) / 64 - 0.5).cpu().numpy()
                _sq_labels = coords_dense_indices[0, 0,
                    struct_coords[:, 1] // 2,
                    struct_coords[:, 2] // 2,
                    struct_coords[:, 3] // 2].cpu().numpy()
                _mesh_vis = trimesh.load(_mesh_path, force='mesh')
                _vtx_labels = map_voxel_labels_to_vertices(_mesh_vis.vertices, _sv_norm, _sq_labels)
                visualize_and_save(_mesh_vis, _vtx_labels, output_dir, output_name='condition_routing.mp4')
                del _sv_norm, _sq_labels, _mesh_vis, _vtx_labels
                torch.cuda.empty_cache()
            else:
                log.info("No structure mesh found for routing visualization; continuing.")
        except Exception as _vis_err:
            log.warning("Could not render condition routing video: %s; continuing texture optimization.", _vis_err)

    else:
        log.info("No local SQ texture overrides provided; all voxels use the global condition.")

    _text_conditioner_to(generation_pipeline, 'cpu')

    flow_model_key, flow_model = _select_slat_flow_model(generation_pipeline, app_type)

    # get_cond() is done — only the selected SLAT flow model is needed for the loop. Offload everything else.
    for k, m in generation_pipeline.models.items():
        if k != flow_model_key:
            m.cpu()
    torch.cuda.empty_cache()
    
    if app_type == 'image':
        sampler_params = {
            "cfg_strength": cfg.img_model.cfg_strength,
            "cfg_interval": cfg.img_model.cfg_interval,
        }
        rescale_t = cfg.img_model.rescale_t
    
    else:
        sampler_params = {
            "cfg_strength": cfg.text_model.cfg_strength,
            "cfg_interval": cfg.text_model.cfg_interval,
        }
        rescale_t = cfg.text_model.rescale_t

    t_seq = np.linspace(1, 0, cfg.sim_guidance.steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(cfg.sim_guidance.steps))
    
    std = torch.tensor(generation_pipeline.slat_normalization['std'])[None].cuda()
    mean = torch.tensor(generation_pipeline.slat_normalization['mean'])[None].cuda()

    loop_start = time.perf_counter()
    log.info(f"Beginning self-similarity guidance + flow sampling loop for {len(t_pairs)} steps...")

    self_attn_region = None
    mixed_self_attn_boost = 1.0
    if coords_dense_indices is not None:
        mixed_self_attn_boost = float(
            getattr(cfg.sim_guidance, 'mixed_self_attn_boost', 1.0) or 1.0)
        if mixed_self_attn_boost > 1.0:
            self_attn_region = coords_dense_indices
            log.info(
                "Region-boosted self-attention ON: in-region weights x%.2f before softmax "
                "over %d condition regions, all %d steps.",
                mixed_self_attn_boost, int(coords_dense_indices.max().item()) + 1,
                len(t_pairs),
            )
        else:
            log.info("Region-boosted self-attention off (mixed_self_attn_boost=%.2f); "
                     "self-attention is stock.", mixed_self_attn_boost)

    for iteration, (t, t_prev) in enumerate(t_pairs):
        optimizer.zero_grad()
        
        # Diffusion
        struct_feats_params_clone = struct_feats_params.clone().cuda()
        noise = sp.SparseTensor(
            feats = struct_feats_params_clone,
            coords = struct_coords.int(),
        ).cuda()
        
        local_kwargs = {}
        if cond_list is not None:
            local_kwargs['cond_list'] = cond_list
            local_kwargs['coords_dense_indices'] = coords_dense_indices
        if self_attn_region is not None:
            local_kwargs['self_attn_region'] = self_attn_region
            local_kwargs['self_attn_region_boost'] = mixed_self_attn_boost

        with torch.no_grad():
            out = generation_pipeline.slat_sampler.sample_once(flow_model, noise, t, t_prev, **cond, **sampler_params, **local_kwargs)
            
        sample = out.pred_x_prev
        struct_feats_params.data = sample.feats

        # guideflow loss put to 0
        if cfg.sim_guidance.loss_weight == 0:
            feats = struct_feats_params.detach() * std + mean
    
        # Optimization - Structure Loss
        if cfg.sim_guidance.loss_weight > 0 and iteration < len(t_pairs) - 1:
            struct_loss = chunked_contrastive_loss(struct_feats_params[None, None, ...], struct_labels)

            total_loss = cfg.sim_guidance.loss_weight * struct_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (iteration == 0) or (iteration + 1) % cfg.log_every == 0:
                message = f"Step: {iteration}, Structure Loss: {struct_loss.item():.4f}, Total Loss: {total_loss.item():.4f}"
                log.info(message)
                
            if total_loss < best_loss:
                best_loss = total_loss.item()
                feats = struct_feats_params.detach() * std + mean
    log.info(
        "Completed self-similarity guidance + flow sampling loop in %.2fs",
        time.perf_counter() - loop_start,
    )
    
    # Move SLAT decoders back to GPU for decoding.
    decoder_load_start = time.perf_counter()
    decoder_keys = ['slat_decoder_mesh']
    if decode_texture:
        decoder_keys.append('slat_decoder_gs')
    for k in decoder_keys:
        if k in generation_pipeline.models:
            generation_pipeline.models[k].cuda()
        elif decode_texture:
            raise KeyError(f"Texture decode requested but required decoder is missing: {k}")
    log.info("Moved SLAT decoder(s) to CUDA in %.2fs: %s", time.perf_counter() - decoder_load_start, decoder_keys)

    # Decode SLAT
    log.info("Decoding output SLAT...")
    decode_start = time.perf_counter()
    out_meshpath = osp.join(output_dir,  'out_sim.glb')
    generation.decode_slat(generation_pipeline, feats, struct_coords, out_meshpath, None, texture=decode_texture)
    log.info("Decoded output SLAT in %.2fs: %s", time.perf_counter() - decode_start, out_meshpath)
    log.info("Completed self-similarity optimization in %.2fs", time.perf_counter() - overall_start)
