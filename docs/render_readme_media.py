"""Generate README media from a completed SpaceFlow run.

This script uses the same CPU rasterizer as the experiment-comparison helper,
so it works on headless login nodes without Blender or Xvfb.
"""

from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pyvista as pv
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sq_ui.scripts import render_spaceflow_experiment_comparison as render_utils  # noqa: E402


LOCAL_VARIANT = "output/01_local_tau3_tau10_polyak0p18"


def _project_meshes(meshes: list[dict[str, object]], azim: float, elev: float) -> tuple[list[dict[str, object]], float, float]:
    rotation = render_utils.camera_rotation(azim, elev)
    max_extent = 0.0
    for mesh_info in meshes:
        vertices = np.asarray(mesh_info["vertices"], dtype=np.float64)
        max_extent = max(max_extent, float(np.max(vertices.max(axis=0) - vertices.min(axis=0))))

    scale = 1.0 / max(max_extent, 1e-8)
    projected = []
    max_abs = np.array([0.0, 0.0])
    for mesh_info in meshes:
        vertices_camera = (np.asarray(mesh_info["vertices"], dtype=np.float64) * scale) @ rotation.T
        vertices = vertices_camera.copy()
        vertices[:, 1] *= -1.0
        vertex_normals = np.asarray(mesh_info["vertex_normals"], dtype=np.float64) @ rotation.T
        annotations = []
        for label, position in mesh_info.get("annotations", []):
            position_camera = (np.asarray(position, dtype=np.float64) * scale) @ rotation.T
            annotations.append((label, np.array([position_camera[0], -position_camera[1], position_camera[2]])))
        projected.append({
            **mesh_info,
            "vertices": vertices,
            "vertex_normals": vertex_normals,
            "annotations": annotations,
        })
        max_abs = np.maximum(max_abs, np.max(np.abs(vertices[:, :2]), axis=0))

    return projected, float(max_abs[0] * 1.12), float(max_abs[1] * 1.18)


def _draw_panel(ax: plt.Axes, mesh_info: dict[str, object], lim_x: float, lim_y: float, title: str) -> None:
    light_dir = np.array([-0.35, -0.45, 0.82], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    panel = render_utils._rasterize_panel(mesh_info, lim_x, lim_y, light_dir)
    ax.imshow(panel, extent=(-lim_x, lim_x, -lim_y, lim_y), origin="upper")
    for annotation, position in mesh_info.get("annotations", []):
        ax.text(
            position[0],
            position[1],
            annotation,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "circle,pad=0.20",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.6,
                "alpha": 0.80,
            },
            zorder=10,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_y, lim_y)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=7)


def render_case_study(run_dir: Path, output_path: Path) -> None:
    manifest = render_utils.asset_manifest(render_utils.run_meta(run_dir))
    variant_dir = run_dir / LOCAL_VARIANT
    meshes = [
        {
            **render_utils.load_sq_mesh(variant_dir / "spatial_control_mesh.ply", manifest),
            "title": "Superquadric controls",
        },
        {
            **render_utils.load_mesh(variant_dir / "struct_mesh_zup.glb"),
            "title": "TRELLIS structure",
            "annotations": [],
        },
        {
            **render_utils.load_mesh(variant_dir / "out_sim.glb"),
            "title": "Similarity-refined asset",
            "annotations": [],
        },
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.7), dpi=220)
    fig.patch.set_facecolor("white")
    for ax, mesh_info in zip(np.atleast_1d(axes), meshes):
        projected, lim_x, lim_y = _project_meshes([mesh_info], azim=-38.0, elev=18.0)
        _draw_panel(ax, projected[0], lim_x, lim_y, str(mesh_info["title"]))
    fig.suptitle("Sailboat Case Study: Local-Tau Spatial Control", fontsize=16, y=0.98)
    footer = (
        "Prompt: Sailboat. Global appearance: white. Local appearance: yellow on four low-control primitives. "
        "Local tau variant: low tau 3, high tau 10, Polyak tau 0.18."
    )
    fig.text(0.5, 0.040, "\n".join(textwrap.wrap(footer, width=145)), ha="center", va="bottom", fontsize=8.8)
    plt.subplots_adjust(left=0.015, right=0.985, top=0.84, bottom=0.16, wspace=0.035)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def render_spin(run_dir: Path, output_prefix: Path) -> None:
    variant_dir = run_dir / LOCAL_VARIANT
    mesh = trimesh.load(variant_dir / "out_sim.glb", force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = vertices - (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    face_colors = (render_utils._face_colors(mesh, faces) * 255.0).clip(0, 255).astype(np.uint8)
    vtk_faces = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    poly = pv.PolyData(vertices, vtk_faces)
    poly.cell_data["rgb"] = face_colors

    frames = []
    frame_count = 48
    radius = 2.25
    z = 0.64
    for index in range(frame_count):
        angle = np.deg2rad(-36.0 + index * 360.0 / frame_count)
        plotter = pv.Plotter(off_screen=True, window_size=(720, 560))
        plotter.set_background("white")
        plotter.add_mesh(poly, scalars="rgb", rgb=True, smooth_shading=True)
        plotter.camera_position = [
            (radius * np.cos(angle), radius * np.sin(angle), z),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 0.72
        frame = plotter.screenshot(return_img=True)
        plotter.close()
        frames.append(frame)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_prefix.with_suffix(".mp4"), frames, fps=18, quality=9, macro_block_size=16)
    imageio.imwrite(output_prefix.with_name(f"{output_prefix.name}_poster.png"), frames[0])


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit("Usage: python docs/render_readme_media.py /path/to/completed/sailboat/run")
    run_dir = Path(argv[0]).expanduser().resolve()
    media_dir = REPO_ROOT / "docs" / "media"
    render_case_study(run_dir, media_dir / "sailboat_case_study.png")
    render_spin(run_dir, media_dir / "sailboat_spin")


if __name__ == "__main__":
    main()
