from html import parser
import copy
import json
import os
import os.path as osp
import gc
import shutil
import trimesh
from PIL import Image
import logging as log
from omegaconf import OmegaConf
import argparse
import random
import time
import numpy as np
from skimage import measure

import torch
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import open3d_pycg as o3d

import utils3d

import sys
sys.path.append('.')

from third_party.PartField.partfield.model_trainer_pvcnn_only_demo import Model
from lib.opt import self_similarity
from lib.util import common, render, pointcloud
from third_party.TRELLIS.trellis.pipelines import TrellisTextTo3DPipeline
from third_party.TRELLIS.trellis.utils import postprocessing_utils
from utils import merge_meshes

log.getLogger().setLevel(log.INFO)
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

STEPS_SHAPE_GEN = 12
CFG_SHAPE_GEN = 7.5
INPUT_SQ_COLORED_GLB = "input_superquadrics_colored.glb"
INPUT_SQ_HIGH_COLOR = np.array([0xf5, 0x9e, 0x0b, 0xff], dtype=np.uint8)
INPUT_SQ_LOW_COLOR = np.array([0xf8, 0xfa, 0xfc, 0xff], dtype=np.uint8)

def reset_run_state(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trellis_pipeline_path_from_args(args, cfg):
    return (
        args.trellis_pipeline_path
        or os.environ.get("SPACEFLOW_TRELLIS_PIPELINE_PATH")
        or cfg.trellis_text_model_name
    )


def load_trellis_pipeline(args, cfg):
    trellis_pipeline_path = trellis_pipeline_path_from_args(args, cfg)
    log.info(f"Loading TRELLIS pipeline from: {trellis_pipeline_path}")
    start = time.perf_counter()
    pipeline = TrellisTextTo3DPipeline.from_pretrained(trellis_pipeline_path)
    log.info(f"Loaded TRELLIS pipeline weights and CLIP in {time.perf_counter() - start:.2f}s")
    return pipeline


def move_trellis_text_conditioner(pipeline, device):
    text_cond_model = getattr(pipeline, 'text_cond_model', None)
    if not text_cond_model:
        return
    if 'model' in text_cond_model:
        text_cond_model['model'].to(device)
    if 'null_cond' in text_cond_model:
        text_cond_model['null_cond'] = text_cond_model['null_cond'].to(device)


def offload_trellis_pipeline(pipeline):
    if pipeline is None:
        return
    pipeline.cpu()
    move_trellis_text_conditioner(pipeline, 'cpu')
    torch.cuda.empty_cache()


def init_args(argv=None):
    parser = argparse.ArgumentParser(description='GuideFlow3D - 3D Shape Generation')

    # Guidance mode selection
    parser.add_argument('--guidance_mode', type=str, required=True, choices=['similarity'],
                        help='Guidance mode. Only "similarity" is supported.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--convert_yup_to_zup', action='store_true',
                        help='Convert Y-up coordinate system to Z-up')

    parser.add_argument('--appearance_image', type=str,
                        help='Path to global image prompt for similarity guidance')
    parser.add_argument('--appearance_text', type=str, default='',
                        help='Global text prompt for similarity guidance')

    # SapceControl parameters
    parser.add_argument('--shape_superquadric_path', type=str, required=True,
                        help='Path to shape superquadrics file')
    parser.add_argument('--shape_superquadric_high_control_path', type=str, default=None,
                        help='Path to high control shape superquadrics file (used local_tau_mode is guidance or masking)')
    parser.add_argument('--low_control_superquadric_mask_path', type=str, default=None,
                        help='Path to low control superquadric mask (Only used when local_tau_mode is low_control_mask)')


    parser.add_argument('--spatial_control_mesh_path', type=str, default=None,
                        help='Path to save the spatial control mesh (defaults to <output_dir>/spatial_control_mesh.ply)')
    
    parser.add_argument('--shape_tau', type=float, default=6.0, required=True,
                        help='Value of tau for superquadric control')
    parser.add_argument('--shape_tau_high_control', type=float, default=None, required=False,
                        help='Value of tau for superquadric control')
    parser.add_argument('--polyak_update_tau', type=float, default=0.18, required=False,
                        help='Tau value for Polyak averaging of the high-control model (if using high control). If set to 0 it just perfoms spacecontrol with the lowest provided tau value.')
    parser.add_argument('--local_tau_mode', type=str, choices=['guidance', 'masking', 'low_control_mask'], default='guidance',
                        help='Whether to use local tau guidance, masking or low control mask mode. ')
    parser.add_argument('--full_pipeline', action='store_true',
                        help='Continue past structure generation into PartField and similarity optimization. Default keeps the legacy structure-only behavior.')
    parser.add_argument('--n_repaint_steps', type=int, default=10,
                        help='Number of repaint resampling steps to perform during structure generation to improve blending (default: 10). Set to 0 to disable.')                        
    parser.add_argument('--structure_seed', type=int, default=1,
                        help='Random seed for TRELLIS sparse-structure generation (default: 1).')
    parser.add_argument('--texture_optim_steps', type=int, default=None,
                        help='Override config/default.yaml sim_guidance.steps for texture similarity optimization. Minimum: 2.')
    parser.add_argument('--mixed_self_attn_boost', type=float, default=None,
                        help='Override config/default.yaml sim_guidance.mixed_self_attn_boost. '
                             'Multiply self-attention weights by this factor before the softmax '
                             'between voxels sharing a routed condition. 1.0 = stock attention.')
    parser.add_argument('--trellis_pipeline_path', type=str, default=None,
                        help='TRELLIS pipeline config/model path. Defaults to SPACEFLOW_TRELLIS_PIPELINE_PATH or config/default.yaml trellis_text_model_name.')
    parser.add_argument('--geometry_only_decode', action='store_true',
                        help='Decode out_sim.glb as white geometry only. Use with text-only lightweight pipeline configs that do not include a Gaussian decoder.')


    parser.add_argument('--text_prompt', type=str, required=True,
                        help='Text prompt for 3D shape generation')
    parser.add_argument('--local_text_prompts', type=str, default=None,
                        help='JSON-encoded list of per-SQ local text prompts (text-similarity mode)')
    parser.add_argument('--local_image_paths', type=str, default=None,
                        help='JSON-encoded list of per-SQ local image file paths (image-similarity mode)')

    args = parser.parse_args(argv)

    if args.appearance_text and args.appearance_image:
        parser.error("Provide either --appearance_image or --appearance_text for similarity guidance, not both.")

    if not args.appearance_text and not args.appearance_image:
        parser.error("Provide either --appearance_image or --appearance_text for similarity guidance.")

    return args

def add_superquadric_compact_rot_mat(
    scalings: np.array=np.array([1.0, 1.0, 1.0]),
    exponents: np.array=np.array([2.0, 2.0, 2.0]),
    translation: np.array=np.array([0.0, 0.0, 0.0]),
    rotation: np.array=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0.0, 0.0,1.0]]),
    tapering=None,
    bending=None,
    resolution: int=10,
    visible: bool=True):
    """Adds a superqiadroc mesh to the scene."""

    def apply_taper(x, y, z, c, kx, ky):
        c = float(c) if abs(float(c)) > 1e-8 else 1e-8
        z_norm = z / c
        x *= float(kx) * z_norm + 1.0
        y *= float(ky) * z_norm + 1.0

    def apply_bending_axis(x, y, z, kb, alpha, axis):
        kb = float(kb)
        if abs(kb) < 1e-3:
            return
        alpha = float(alpha)
        if axis == "z":
            u, v, w = x.copy(), y.copy(), z.copy()
        elif axis == "x":
            u, v, w = y.copy(), z.copy(), x.copy()
        elif axis == "y":
            u, v, w = z.copy(), x.copy(), y.copy()
        else:
            raise ValueError(axis)

        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)
        beta = np.arctan2(v, u)
        r = np.sqrt(u * u + v * v) * np.cos(alpha - beta)
        inv_kb = 1.0 / kb
        gamma = w * kb
        rho = inv_kb - r
        rb = inv_kb - rho * np.cos(gamma)
        expr = rb - r
        u = u + expr * cos_alpha
        v = v + expr * sin_alpha
        w = rho * np.sin(gamma)

        if axis == "z":
            x[:], y[:], z[:] = u, v, w
        elif axis == "x":
            x[:], y[:], z[:] = w, u, v
        else:
            x[:], y[:], z[:] = v, w, u

    def create_superquadric_mesh(A, B, C, e1, e2, N):
        def f(o, m):
            return np.sign(np.sin(o)) * np.abs(np.sin(o))**m
        def g(o, m):
            return np.sign(np.cos(o)) * np.abs(np.cos(o))**m
        u = np.linspace(-np.pi, np.pi, N, endpoint=True)
        v = np.linspace(-np.pi/2.0, np.pi/2.0, N, endpoint=True)
        u = np.tile(u, N)
        v = (np.repeat(v, N))
        if np.linalg.det(rotation) < 0:
            u = u[::-1]
        triangles = []

        x = A * g(v, e1) * g(u, e2)
        y = B * g(v, e1) * f(u, e2)
        z = C * f(v, e1)
        # Set poles to zero to account for numerical instabilities in f and g due to ** operator
        x[:N] = 0.0
        x[-N:] = 0.0
        if tapering is not None:
            apply_taper(x, y, z, C, tapering[0], tapering[1])
        if bending is not None:
            # Packed as [k_z, alpha_z, k_x, alpha_x, k_y, alpha_y].
            apply_bending_axis(x, y, z, bending[4], bending[5], "y")
            apply_bending_axis(x, y, z, bending[2], bending[3], "x")
            apply_bending_axis(x, y, z, bending[0], bending[1], "z")
        vertices =  np.concatenate([np.expand_dims(x, 1),
                                    np.expand_dims(y, 1),
                                    np.expand_dims(z, 1)], axis=1)
        vertices =  (rotation @ vertices.T).T +translation  # TODO verify left or right apply rotation

        triangles = []
        for i in range(N-1):
            for j in range(N-1):
                triangles.append([i*N+j, i*N+j+1, (i+1)*N+j])
                triangles.append([(i+1)*N+j, i*N+j+1, (i+1)*N+(j+1)])
        # Connect first and last vertex in each row
        for i in range(N - 1):
            triangles.append([i * N + (N - 1), i * N, (i + 1) * N + (N - 1)])
            triangles.append([(i + 1) * N + (N - 1), i * N, (i + 1) * N])

        triangles.append([(N-1)*N+(N-1), (N-1)*N, (N-1)])
        triangles.append([(N-1), (N-1)*N, 0])

        return vertices, triangles


    vertices, triangles = create_superquadric_mesh(scalings[0], scalings[1], scalings[2],
                                                exponents[0], exponents[1],
                                                resolution)
    return vertices, triangles

def load_superquadric_from_file(file_path: str) -> list:
    par_dict = np.load(file_path)
    print(par_dict)
    scale = par_dict['scales']        # 3 (3x1 vector)
    rotate = par_dict['rotations']    # 3 (3x3 rotation matrix)
    shapes = par_dict['shapes']       # 2 (2x1 vector)
    trans = par_dict['translations']  # 3 (3x1 vector)
    num_el = scale.shape[0]           # number of superquadrics
    tapering = par_dict['tapering'] if 'tapering' in par_dict else np.zeros((num_el, 2))
    bending = par_dict['bending'] if 'bending' in par_dict else np.zeros((num_el, 6))
    control_levels = par_dict['control_levels'] if 'control_levels' in par_dict else np.ones((num_el,))

    superquadrics = {}
    for k in range(num_el):
        superquadric_dict = {}
        superquadric_dict['scale'] = scale[k, :]
        superquadric_dict['shape'] = shapes[k]
        superquadric_dict['rotation'] = rotate[k, :]
        superquadric_dict['translation'] = trans[k, :]
        superquadric_dict['tapering'] = tapering[k, :]
        superquadric_dict['bending'] = bending[k, :]
        superquadric_dict['control_level'] = 'low' if float(control_levels[k]) < 0.5 else 'high'
        superquadric_dict['color'] = [90, 200, 255]
        superquadrics[k] = superquadric_dict
    return superquadrics

def export_colored_superquadrics_glb(npz_path, output_path, aabb=None, center=None, scale=None):
    start = time.perf_counter()
    superquadrics = load_superquadric_from_file(npz_path)
    meshes = []
    raw_meshes = []
    for sq_id in superquadrics:
        sq = superquadrics[sq_id]
        vertices, triangles = add_superquadric_compact_rot_mat(
            sq['scale'],
            sq['shape'],
            sq['translation'],
            sq['rotation'],
            sq['tapering'],
            sq['bending'],
            resolution=100,
        )
        raw_mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(triangles), process=False)
        raw_meshes.append(raw_mesh)

    if not raw_meshes:
        raise ValueError(f"No superquadrics found in {npz_path}")

    if aabb is None or center is None or scale is None:
        all_vertices = np.concatenate([mesh.vertices for mesh in raw_meshes], axis=0)
        aabb = np.stack([all_vertices.min(0), all_vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = 1.0 / ((aabb[1] - aabb[0]).max())

    for raw_mesh, sq_id in zip(raw_meshes, superquadrics):
        sq = superquadrics[sq_id]
        mesh = raw_mesh.copy()
        mesh.vertices = (mesh.vertices - center) * scale
        color = INPUT_SQ_LOW_COLOR if sq.get('control_level') == 'low' else INPUT_SQ_HIGH_COLOR
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh,
            vertex_colors=np.tile(color, (mesh.vertices.shape[0], 1)),
        )
        meshes.append(mesh)

    scene = trimesh.Scene()
    for sq_id, mesh in zip(superquadrics, meshes):
        scene.add_geometry(mesh, node_name=f"superquadric_{sq_id}", geom_name=f"superquadric_{sq_id}")
    scene.export(output_path)
    log.info(
        "Colored input superquadrics GLB exported in %.2fs: %s (%d SQs)",
        time.perf_counter() - start,
        output_path,
        len(meshes),
    )

def load_superquadrics(path, spatial_control_mesh_path, aabb=None, center=None, scale=None):
    start = time.perf_counter()
    # Generate spatial control mesh from superquadric primitives and write to spatial_control_mesh_path
    superquadrics = load_superquadric_from_file(path)

    meshes = []
    for superquadric_id in superquadrics.keys():
        vertices, triangles = add_superquadric_compact_rot_mat(
        superquadrics[superquadric_id]['scale'],
        superquadrics[superquadric_id]['shape'],
        superquadrics[superquadric_id]['translation'],
        superquadrics[superquadric_id]['rotation'],
        superquadrics[superquadric_id]['tapering'],
        superquadrics[superquadric_id]['bending'],
        resolution=100)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        meshes.append(mesh)
    merged_mesh = merge_meshes(meshes)
    aabb = np.stack([np.asarray(merged_mesh.vertices).min(0), np.asarray(merged_mesh.vertices).max(0)]) if aabb is None else aabb
    center = (aabb[0] + aabb[1]) / 2 if center is None else center
    scale = 1/((aabb[1] - aabb[0]).max())  if scale is None else scale

    merged_mesh.translate(-center)
    merged_mesh.scale(scale, (0,0,0))
    o3d.io.write_triangle_mesh(spatial_control_mesh_path, merged_mesh)
    log.info(
        "Spatial control mesh generated from superquadrics in %.2fs: %s (%d SQs)",
        time.perf_counter() - start,
        spatial_control_mesh_path,
        len(superquadrics),
    )

    if aabb is not None and center is not None and scale is not None:
        return aabb, center, scale

def build_individual_sq_meshes_normalized(npz_path):
    """Build individual superquadric meshes in the same normalised space as the merged mesh.

    The merged mesh is centred and scaled to unit size (same as load_superquadrics).
    Returns a list of open3d TriangleMesh objects in the [-0.5, 0.5] coordinate space.
    """
    superquadrics = load_superquadric_from_file(npz_path)
    meshes = []
    for sq_id in superquadrics:
        vertices, triangles = add_superquadric_compact_rot_mat(
            superquadrics[sq_id]['scale'],
            superquadrics[sq_id]['shape'],
            superquadrics[sq_id]['translation'],
            superquadrics[sq_id]['rotation'],
            superquadrics[sq_id]['tapering'],
            superquadrics[sq_id]['bending'],
            resolution=100)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        meshes.append(mesh)

    # Compute normalization from merged mesh (mirrors load_superquadrics)
    merged = merge_meshes(meshes)
    all_verts = np.asarray(merged.vertices)
    aabb = np.stack([all_verts.min(0), all_verts.max(0)])
    center = (aabb[0] + aabb[1]) / 2
    scale = 1.0 / ((aabb[1] - aabb[0]).max())

    normalized = []
    for mesh in meshes:
        m = copy.deepcopy(mesh)
        m.translate(-center)
        m.scale(scale, (0, 0, 0))
        normalized.append(m)
    return normalized


def count_superquadrics(npz_path):
    with np.load(npz_path) as par_dict:
        return int(par_dict['scales'].shape[0])


def parse_local_condition_list(raw_json, num_superquadrics, label, require_existing_files=False):
    if not raw_json:
        return None
    values = json.loads(raw_json)
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a JSON list")
    if len(values) > num_superquadrics:
        raise ValueError(f"{label} has {len(values)} entries but only {num_superquadrics} superquadrics")
    normalized = [str(value or "").strip() for value in values]
    normalized.extend([""] * (num_superquadrics - len(normalized)))
    if require_existing_files:
        missing = [path for path in normalized if path and not osp.exists(path)]
        if missing:
            raise FileNotFoundError(f"Local image path(s) not found: {missing}")
    return normalized if any(value for value in normalized) else None


def sparse_voxels_to_glb(sparse_points, grid_size=64, output_filename="output.glb"):
    """
    Converts sparse voxel coordinates to a GLB mesh.

    :param sparse_points: List or array of (x, y, z) coordinates (integers).
    :param grid_size: The size of the voxel bounding box (e.g., 64).
    :param output_filename: The name of the output GLB file.
    """

    start = time.perf_counter()
    log.info("Creating %dx%dx%d sparse voxel grid...", grid_size, grid_size, grid_size)
    # Init grid
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    # Populate grid with sparse points
    sparse_points = np.round(sparse_points).astype(int)
    for x, y, z in sparse_points:
        # Check bounds just in case
        if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
            voxel_grid[x, y, z] = True

    # Padding the grid (Needed for marching cubes)
    padded_grid = np.pad(voxel_grid, pad_width=1, mode='constant', constant_values=False)

    log.info("Running Marching Cubes algorithm...")
    # Marching Cubes (Level set at 0.5 to extract the surface between occupied and empty voxels)
    verts, faces, normals, values = measure.marching_cubes(padded_grid, level=0.5)

    # Shift vertices back by 1 to account for the padding we added
    verts = verts - 1.0

    log.info("Generating mesh and exporting to GLB...")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # # Smoothing of the mesh
    # trimesh.smoothing.filter_taubin(mesh, iterations=50)

    # Export to GLB format
    mesh.export(output_filename)
    log.info(
        "Successfully exported sparse voxel mesh in %.2fs: %s",
        time.perf_counter() - start,
        output_filename,
    )

def predict_part(obj_path, output_dir):
    start = time.perf_counter()
    log.info("Extracting PartField feature planes...")
    partfield_config = 'third_party/PartField/config.yaml'
    partfield_cfg = OmegaConf.load(partfield_config)
    partfield_cfg.dataset.val_num_workers = 0
    partfield_ckpt = partfield_cfg.continue_ckpt
    if partfield_ckpt and not osp.isabs(partfield_ckpt):
        partfield_ckpt = osp.join(osp.dirname(osp.abspath(partfield_config)), partfield_ckpt)

    seed_everything(partfield_cfg.seed)

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Lightning defaults to ./lightning_logs under cwd (repo root); team members often
    # cannot mkdir there. Keep all PL artifacts under this run's output_dir.
    pl_root = osp.join(output_dir, 'pl_partfield')
    common.ensure_dir(pl_root)

    trainer = Trainer(devices=1,
                      accelerator="gpu",
                      precision="16-mixed",
                      max_epochs=partfield_cfg.training_epochs,
                      log_every_n_steps=1,
                      limit_train_batches=3500,
                      limit_val_batches=None,
                      default_root_dir=pl_root,
                      logger=False,
                      enable_checkpointing=False,
                      enable_progress_bar=False,
                      enable_model_summary=False,
                     )

    partfield_model = Model(partfield_cfg, obj_path)
    output = trainer.predict(partfield_model, ckpt_path=partfield_ckpt)
    part_planes, uid = output[0]
    np.save(f'{output_dir}/part_feat_{uid}_batch_part_plane.npy', part_planes)

    del partfield_model
    gc.collect() # Free up memory
    log.info("Extracted PartField feature planes in %.2fs", time.perf_counter() - start)

def run(args, cfg=None, generation_pipeline=None):
    run_start = time.perf_counter()
    reset_run_state()
    cfg = cfg or OmegaConf.load('config/default.yaml')
    if args.texture_optim_steps is not None:
        if args.texture_optim_steps < 2:
            raise ValueError("--texture_optim_steps must be an integer at least 2")
        cfg = copy.deepcopy(cfg)
        cfg.sim_guidance.steps = int(args.texture_optim_steps)
        log.info("Overriding texture optimization steps: %d", int(cfg.sim_guidance.steps))
    if args.mixed_self_attn_boost is not None:
        if args.mixed_self_attn_boost < 1.0:
            raise ValueError("--mixed_self_attn_boost must be at least 1.0 (1.0 = stock attention)")
        cfg = copy.deepcopy(cfg)
        cfg.sim_guidance.mixed_self_attn_boost = float(args.mixed_self_attn_boost)
        log.info("Overriding mixed self-attention boost: %.2f",
                 float(cfg.sim_guidance.mixed_self_attn_boost))

    common.ensure_dir(args.output_dir)

    def copy_input_npz(src_path, output_name):
        if not src_path:
            return
        dst_path = osp.join(args.output_dir, output_name)
        shutil.copy2(src_path, dst_path)
        log.info(f"Copied input superquadrics: {dst_path}")

    copy_input_npz(args.shape_superquadric_path, 'input_superquadrics_all.npz')
    copy_input_npz(args.shape_superquadric_high_control_path, 'input_superquadrics_high_control.npz')
    copy_input_npz(args.low_control_superquadric_mask_path, 'input_superquadrics_low_control_bbox.npz')

    # Generate spatial control mesh from superquadrics
    spatial_control_mesh_path = osp.join(args.output_dir, 'spatial_control_mesh.ply')
    control_mesh_start = time.perf_counter()
    aabb, center, scale = load_superquadrics(args.shape_superquadric_path, spatial_control_mesh_path)
    export_colored_superquadrics_glb(
        args.shape_superquadric_path,
        osp.join(args.output_dir, INPUT_SQ_COLORED_GLB),
        aabb=aabb,
        center=center,
        scale=scale,
    )


    low_control_superquadric_mask_path = None
    if args.shape_tau_high_control is not None:
        assert args.shape_tau_high_control >= args.shape_tau, "shape_tau_high_control must be greater than or equal to shape_tau"
        
        print(f"Using high control tau: {args.shape_tau_high_control} and low control tau: {args.shape_tau}, with local tau mode: {args.local_tau_mode}")
        high_control_spatial_control_mesh_path = osp.join(args.output_dir, 'high_control_spatial_control_mesh.ply')
        load_superquadrics(args.shape_superquadric_high_control_path, high_control_spatial_control_mesh_path, aabb=aabb, center=center, scale=scale)
        
        if args.local_tau_mode == 'low_control_mask':
            low_control_superquadric_mask_path = osp.join(args.output_dir, 'low_control_superquadric_mask.ply')
            load_superquadrics(args.low_control_superquadric_mask_path, low_control_superquadric_mask_path, aabb=aabb, center=center, scale=scale)
    log.info("Prepared all superquadric control meshes in %.2fs", time.perf_counter() - control_mesh_start)


    # Load structure mesh
    log.info("Creating structure mesh with SpaceControl code...")

    owns_pipeline = generation_pipeline is None
    if owns_pipeline:
        pipeline = load_trellis_pipeline(args, cfg)
    else:
        pipeline = generation_pipeline
        log.info(f"Reusing preloaded TRELLIS pipeline from: {trellis_pipeline_path_from_args(args, cfg)}")
    cuda_start = time.perf_counter()
    pipeline.cuda()
    move_trellis_text_conditioner(pipeline, 'cuda')
    log.info(f"Moved TRELLIS pipeline to CUDA in {time.perf_counter() - cuda_start:.2f}s")

    text_prompt = args.text_prompt

    # Sparse voxels
    structure_start = time.perf_counter()
    log.info("Generating TRELLIS sparse structure with seed %d", int(args.structure_seed))
    coords = pipeline.gen_structure_v2(text_prompt, seed=int(args.structure_seed), vis_output_dir=None, sparse_structure_sampler_params={
        "steps": STEPS_SHAPE_GEN,
        "cfg_strength": CFG_SHAPE_GEN,
        "t0_idx_value": args.shape_tau,
        "spatial_control_mesh_path": spatial_control_mesh_path,
        "high_control_spatial_control_mesh_path": high_control_spatial_control_mesh_path if args.shape_tau_high_control is not None else None,
        "low_control_superquadric_mask_path": low_control_superquadric_mask_path if low_control_superquadric_mask_path is not None else None,
        "t0_idx_value_high_control": args.shape_tau_high_control if args.shape_tau_high_control is not None else None,
        "polyak_update_tau": args.polyak_update_tau,
        "local_tau_mode": args.local_tau_mode,
        "n_repaint_steps": args.n_repaint_steps,
    })
    log.info(
        "Generated TRELLIS sparse structure in %.2fs (%d voxels)",
        time.perf_counter() - structure_start,
        coords.shape[0],
    )

    # Convert sparse voxels to mesh
    log.info("Converting sparse voxels to mesh...")

    coords_np = coords.detach().cpu().numpy()

    filtered_coords = coords_np[:, 1:]
    print(f"Sparse voxel tensor shape: {coords_np.shape}")
    print(f"Number of valid voxels: {filtered_coords.shape[0]}")

    sparse_voxels_to_glb(filtered_coords, grid_size=64, output_filename=osp.join(args.output_dir, "sample.glb"))

    # glb.export("sample.glb")

    # log.info("Loading generated mesh...")

    mesh_export_start = time.perf_counter()
    struct_mesh = trimesh.load(osp.join(args.output_dir, "sample.glb"), force='mesh')
    # Generator / marching-cubes output in Y-up; keep a copy for debugging.
    struct_mesh.export(osp.join(args.output_dir, 'struct_mesh.glb'))
    log.info("Loaded and exported raw structure mesh in %.2fs", time.perf_counter() - mesh_export_start)

    if args.full_pipeline and args.guidance_mode == 'similarity':
        log.info("Keeping TRELLIS pipeline loaded for similarity guidance; offloading it until refinement.")
        offload_trellis_pipeline(pipeline)
    else:
        if owns_pipeline:
            del pipeline
            gc.collect() # Free up memory
        else:
            offload_trellis_pipeline(pipeline)

    # Canonical mesh for renders, voxels, and PartField must share one frame (Z-up if converting).
    normalize_start = time.perf_counter()
    if args.convert_yup_to_zup:
        struct_mesh = pointcloud.convert_mesh_yup_to_zup(struct_mesh)
    struct_mesh.export(osp.join(args.output_dir, 'struct_mesh_zup.glb'))
    log.info("Prepared canonical structure mesh in %.2fs", time.perf_counter() - normalize_start)
    struct_mesh_for_pipeline = osp.join(args.output_dir, 'struct_mesh_zup.glb')

    struct_render_dir = osp.join(args.output_dir, 'struct_renders')
    common.ensure_dir(struct_render_dir)
    if args.full_pipeline and args.guidance_mode == 'similarity':
        render_start = time.perf_counter()
        log.info("Exporting Blender-normalized structure mesh without PNG renders...")
        out_renderviews = render.export_normalized_mesh(struct_mesh_for_pipeline, struct_render_dir)
        log.info("Exported Blender-normalized structure mesh in %.2fs", time.perf_counter() - render_start)
    else:
        render_start = time.perf_counter()
        log.info(f"Rendering structure mesh for {cfg.num_views // 10} views...")
        out_renderviews = render.render_all_views(struct_mesh_for_pipeline, struct_render_dir, num_views=cfg.num_views // 10)
        log.info("Rendered structure mesh views in %.2fs", time.perf_counter() - render_start)

    # struct_renders/mesh.ply is the Blender-normalized mesh; use it as the single source of truth
    # for both voxelization and PartField feature extraction so that both operate in the same
    # coordinate system (Blender's normalize_scene + GLTF→Blender axis convention).
    struct_blender_ply = osp.join(struct_render_dir, 'mesh.ply')

    voxel_dir = osp.join(args.output_dir, 'voxels')
    common.ensure_dir(voxel_dir)
    voxel_start = time.perf_counter()
    log.info("Voxelizing structure mesh...")
    pointcloud.voxelize_mesh(struct_blender_ply, save_path=osp.join(voxel_dir, 'struct_voxels.ply'))
    log.info("Voxelized structure mesh in %.2fs", time.perf_counter() - voxel_start)

    if not args.full_pipeline:
        log.info("Structure-only mode complete. Pass --full_pipeline to continue into PartField and refinement.")
        return

    log.info("Extracting Structure Mesh PartField feature planes...")
    partfield_dir = osp.join(args.output_dir, 'partfield')
    common.ensure_dir(partfield_dir)
    # Use the same Blender-normalized PLY so the PartField triplane canonical space
    # matches the coordinate system of struct_voxels.ply.
    partfield_start = time.perf_counter()
    predict_part(struct_blender_ply, partfield_dir)
    log.info("Completed Structure Mesh PartField extraction in %.2fs", time.perf_counter() - partfield_start)

    log.info("Visualizing PartField clusters on structure mesh...")
    from lib.util.visualization import visualize_and_save, map_voxel_labels_to_vertices
    from lib.util.partfield import cluster_geoms
    _sv = utils3d.io.read_ply(osp.join(voxel_dir, 'struct_voxels.ply'))[0]
    _sc = torch.from_numpy(_sv).float().cuda()
    _sc4d = torch.cat([torch.zeros(_sc.shape[0], 1, dtype=torch.long, device='cuda'),
                       ((_sc + 0.5) * 64).long()], dim=1)
    _planes = torch.from_numpy(np.load(
        osp.join(partfield_dir, 'part_feat_mesh_batch_part_plane.npy'),
        allow_pickle=True)).cuda()
    _vlabels = cluster_geoms(_sc4d, _planes, num_clusters=cfg.sim_guidance.num_part_clusters)
    _mesh_vis = trimesh.load(struct_blender_ply, force='mesh')
    _vtx_labels = map_voxel_labels_to_vertices(_mesh_vis.vertices, _sv, _vlabels)
    visualize_and_save(_mesh_vis, _vtx_labels, args.output_dir, output_name='partfield_clusters.mp4')
    del _sv, _sc, _sc4d, _planes, _vlabels, _mesh_vis, _vtx_labels
    gc.collect()

    if not out_renderviews:
        log.info("Structure rendering failed!")

    log.info("Running similarity-guided optimization...")

    if args.appearance_image:
        app_type = 'image'
        app = args.appearance_image

        app_image = Image.open(args.appearance_image).convert('RGB')
        app_image.save(osp.join(args.output_dir, 'app_image.png'))

    else:
        app_type = 'text'
        app = args.appearance_text

    log.info(f"Using {app_type} for self-similarity guidance...")

    # Parse per-SQ local conditioning args
    num_superquadrics = count_superquadrics(args.shape_superquadric_path)
    local_text_prompts = (
        parse_local_condition_list(args.local_text_prompts, num_superquadrics, "local_text_prompts")
        if app_type == 'text'
        else None
    )
    local_image_paths = (
        parse_local_condition_list(
            args.local_image_paths,
            num_superquadrics,
            "local_image_paths",
            require_existing_files=True,
        )
        if app_type == 'image'
        else None
    )

    local_prompts     = local_text_prompts if app_type == 'text' else local_image_paths
    local_prompt_type = app_type if local_prompts else None
    individual_sq_meshes = build_individual_sq_meshes_normalized(args.shape_superquadric_path) if local_prompts else None

    # Self-Similarity Optimization
    sim_start = time.perf_counter()
    self_similarity.optimize_self_similarity(
        cfg, app, app_type, args.output_dir,
        local_prompts=local_prompts,
        local_prompt_type=local_prompt_type,
        individual_sq_meshes=individual_sq_meshes,
        generation_pipeline=pipeline,
        decode_texture=not args.geometry_only_decode,
    )
    log.info("Completed SpaceFlow run in %.2fs (similarity optimization %.2fs)",
             time.perf_counter() - run_start,
             time.perf_counter() - sim_start)


def main(argv=None):
    args = init_args(argv)
    run(args)

if __name__ == "__main__":
    main()
