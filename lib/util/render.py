import os
import json
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlretrieve
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = os.environ.get('SPACEFLOW_BLENDER_ROOT', str(REPO_ROOT))
BLENDER_PATH = os.environ.get(
    'SPACEFLOW_BLENDER_PATH',
    f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender',
)
BLENDER_RENDER_SCRIPT = os.environ.get(
    'SPACEFLOW_BLENDER_RENDER_SCRIPT',
    str(REPO_ROOT / 'third_party' / 'TRELLIS' / 'dataset_toolkits' / 'blender_script' / 'render.py'),
)

def _install_blender():
    if shutil.which(BLENDER_PATH):
        return
    if os.environ.get('SPACEFLOW_BLENDER_PATH'):
        raise FileNotFoundError(
            f'Blender is not executable at SPACEFLOW_BLENDER_PATH={BLENDER_PATH}. '
            'Set it to an installed Blender 3.x executable.'
        )
    installation = Path(BLENDER_INSTALLATION_PATH)
    installation.mkdir(parents=True, exist_ok=True)
    archive = installation / 'blender-3.0.1-linux-x64.tar.xz'
    try:
        urlretrieve(BLENDER_LINK, archive)
        subprocess.run(['tar', '-xf', str(archive), '-C', str(installation)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            'Blender installation failed. Install Blender 3.x separately and set '
            'SPACEFLOW_BLENDER_PATH to its executable.'
        ) from exc
    if not shutil.which(BLENDER_PATH):
        raise FileNotFoundError(f'Blender installation did not produce an executable: {BLENDER_PATH}')


def _run_blender(args):
    result = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError(
            f'Blender failed with exit code {result.returncode}. '
            f'Executable: {BLENDER_PATH}. Error output:\n{result.stderr[-3000:]}'
        )

def render_all_views(file_path, output_folder, num_views=150):
    _install_blender()
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        BLENDER_PATH, '-b', '-P', BLENDER_RENDER_SCRIPT,
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        '--save_mesh',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    _run_blender(args)
    
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return True


def export_normalized_mesh(file_path, output_folder):
    _install_blender()
    args = [
        BLENDER_PATH, '-b', '-P', BLENDER_RENDER_SCRIPT,
        '--',
        '--views', '[]',
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        '--save_mesh',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)

    _run_blender(args)

    if os.path.exists(os.path.join(output_folder, 'mesh.ply')):
        return True

# ===============LOW DISCREPANCY SEQUENCES================

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]
