#!/usr/bin/env python3
"""Validate successful run markers and nonempty, finite output geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    config=json.loads((args.run_dir/'experiment_runner_config.json').read_text())
    results=[]
    for variant in config['variants']:
        output=Path(variant['output_dir'])
        errors=[]
        geometry={}
        status=(output/'status.txt').read_text().strip() if (output/'status.txt').exists() else 'missing'
        if status!='succeeded':errors.append('run_status='+status)
        mesh_path=output/'out_sim.glb'
        if not mesh_path.exists():errors.append('missing out_sim.glb')
        else:
            try:
                mesh=trimesh.load(mesh_path,force='mesh',process=False)
                if not len(mesh.vertices) or not len(mesh.faces):errors.append('empty mesh')
                if not np.isfinite(mesh.vertices).all():errors.append('nonfinite mesh vertices')
                geometry={'vertices':len(mesh.vertices),'faces':len(mesh.faces),'bounds':mesh.bounds.tolist(),
                          'visual_kind':mesh.visual.kind}
                uv=getattr(mesh.visual,'uv',None)
                texture=getattr(getattr(mesh.visual,'material',None),'baseColorTexture',None)
                if mesh.visual.kind!='texture' or texture is None:
                    errors.append('missing baked texture')
                elif not all(texture.size):
                    errors.append('empty baked texture')
                else:
                    geometry['texture_size']=list(texture.size)
                if uv is None or not np.isfinite(uv).all():errors.append('missing or nonfinite texture coordinates')
            except Exception as error:errors.append(repr(error))
        results.append({'name':variant['name'],'runner':variant.get('runner','spaceflow'),
            'status':status,'output':str(mesh_path),'errors':errors,
            'geometry':geometry,
            'sha256':hashlib.sha256(mesh_path.read_bytes()).hexdigest() if mesh_path.exists() else None})
    report={'passed':all(not x['errors'] for x in results),'variants':results,
        'scope':'Execution, geometry, and texture validity; this does not assert perceptual equivalence or reproduce the entire dataset.'}
    (args.run_dir/'output-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
