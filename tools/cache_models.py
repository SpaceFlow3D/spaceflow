#!/usr/bin/env python3
"""Stage the recorded model revisions for an offline SpaceFlow run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,default=REPO/'spaceflow_runtime/huggingface')
    args=parser.parse_args()
    cache=args.cache_dir.expanduser().resolve()
    hub=cache/'hub';hub.mkdir(parents=True,exist_ok=True)
    pins=json.loads((REPO/'requirements/model-revisions.json').read_text())
    revisions=pins['huggingface'];records=[]
    models=json.loads((REPO/'config/trellis_pipeline/pipeline.json').read_text())['args']['models']
    for model in models.values():
        parts=model.split('/');repo='/'.join(parts[:2]);stem='/'.join(parts[2:])
        for extension in ['.json','.safetensors']:
            name=stem+extension
            path=hf_hub_download(repo,name,revision=revisions[repo],cache_dir=str(hub))
            records.append({'repo':repo,'revision':revisions[repo],'file':name,'bytes':Path(path).stat().st_size})
            print(f'Cached {repo}/{name}',flush=True)
    clip='openai/clip-vit-large-patch14'
    files=HfApi().list_repo_files(clip,revision=revisions[clip])
    weight='model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
    snapshot_download(clip,revision=revisions[clip],cache_dir=str(hub),max_workers=4,
                      allow_patterns=['*.json','merges.txt','vocab.json',weight])
    records.append({'repo':clip,'revision':revisions[clip],'weight':weight})
    partfield='mikaelaangel/partfield-ckpt'
    checkpoint=Path(hf_hub_download(partfield,'model_objaverse.ckpt',revision=revisions[partfield],cache_dir=str(hub)))
    digest=hashlib.sha256()
    with checkpoint.open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    if digest.hexdigest()!=pins['partfield_sha256']:
        raise ValueError('PartField checkpoint does not match the recovered checkpoint SHA256')
    target=REPO/'third_party/PartField/models/model_objaverse.ckpt'
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():shutil.copy2(checkpoint,target)
    else:
        existing=hashlib.sha256()
        with target.open('rb') as stream:
            while block:=stream.read(1024*1024):existing.update(block)
        if existing.hexdigest()!=pins['partfield_sha256']:
            raise ValueError('Existing PartField checkpoint differs; preserve it and choose a separate release folder')
    records.append({'repo':partfield,'revision':revisions[partfield],'sha256':digest.hexdigest()})
    # The recovered loader requests "main". Point this explicitly selected
    # cache to the recorded revisions before enabling offline mode.
    for repo,revision in revisions.items():
        ref=hub/('models--'+repo.replace('/','--'))/'refs/main'
        ref.parent.mkdir(parents=True,exist_ok=True);ref.write_text(revision)
    (cache/'spaceflow-models.json').write_text(json.dumps({'revisions':revisions,'files':records},indent=2)+'\n')
    print(f'Model cache ready: {cache}',flush=True)
    print('Set HF_HOME to this directory and HF_HUB_OFFLINE=1 for the replay.',flush=True)

if __name__=='__main__':main()
