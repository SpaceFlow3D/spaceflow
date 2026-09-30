#!/usr/bin/env python3
"""Replay saved experiment parameters into a new directory."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]

def prepare(example: Path, destination: Path, only: list[str]) -> Path:
    source = example / 'experiment_runner_config.json'
    config = json.loads(source.read_text())
    old_root = str(config['run_dir']).rstrip('/')
    variants = config['variants']
    names = {v['name'] for v in variants}
    unknown = set(only)-names
    if unknown:
        raise ValueError(f'Unknown variants: {sorted(unknown)}')
    selected = set(only)
    if selected:
        # Baselines and copied variants can depend on a generated structure.
        while True:
            dependencies = {v['source_variant'] for v in variants
                            if v['name'] in selected and v.get('source_variant')}
            if dependencies <= selected:
                break
            if not dependencies <= names:
                raise ValueError(f'Unknown dependency variants: {sorted(dependencies-names)}')
            selected.update(dependencies)
    if not (example/'inputs/all.npz').is_file():
        raise FileNotFoundError(f'Missing superquadric input: {example}/inputs/all.npz')
    if destination.exists():
        raise FileExistsError(f'Choose a new output directory: {destination}')
    destination.mkdir(parents=True)
    shutil.copytree(example / 'inputs', destination / 'inputs')
    def relocate(value):
        if isinstance(value, str) and (value == old_root or value.startswith(old_root+'/')):
            return str(destination)+value[len(old_root):]
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, dict):
            return {key: relocate(item) for key,item in value.items()}
        return value
    config = relocate(config)
    config['spaceflow_config'] = str(REPO / 'config/default.yaml')
    if selected:
        config['variants']=[v for v in config['variants'] if v['name'] in selected]
    output = destination / 'experiment_runner_config.json'
    output.write_text(json.dumps(config, indent=2)+'\n')
    metadata_source = example/'run_meta.json'
    if metadata_source.is_file():
        metadata = relocate(json.loads(metadata_source.read_text()))
        metadata.update(status='prepared', run_id=destination.name, run_dir=str(destination),
                        output_dir=str(destination/'output'))
        metadata.pop('finished_at', None)
        (destination/'run_meta.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (destination/'replay_provenance.json').write_text(json.dumps({
        'source_example': str(example), 'source_config_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'parameters_changed': [], 'paths_relocated': {'from':old_root, 'to':str(destination)},
        'variants': [v['name'] for v in config['variants']],
    }, indent=2)+'\n')
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--only', nargs='*', default=[])
    parser.add_argument('--prepare-only', action='store_true')
    args=parser.parse_args()
    config=prepare(args.example_dir.expanduser().resolve(),args.output_dir.expanduser().resolve(),args.only)
    print(f'Relocated config: {config}', flush=True)
    if not args.prepare_only:
        return subprocess.call([sys.executable,str(REPO/'sq_ui/scripts/run_spaceflow_experiment.py'),'--config',str(config)],cwd=REPO)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
