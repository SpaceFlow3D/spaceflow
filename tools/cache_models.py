#!/usr/bin/env python3
"""Stage the recorded model revisions for SpaceFlow text and optional image runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from lib.util.model_revisions import dinov2_hub_repository


def file_hash(path, algorithm='sha256'):
    digest=hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def stage_image_models(cache, pins, records, torch_cache, u2net_cache):
    """Stage the image pipeline, DINOv2, and background removal without CUDA."""
    hub=cache/'hub'
    repo='microsoft/TRELLIS-image-large'
    revision=pins['huggingface'][repo]
    config_path=hf_hub_download(repo,'pipeline.json',revision=revision,cache_dir=str(hub))
    config=json.loads(Path(config_path).read_text())
    records.append({'repo':repo,'revision':revision,'file':'pipeline.json'})
    for stem in config['args']['models'].values():
        for extension in ('.json','.safetensors'):
            name=stem+extension
            path=Path(hf_hub_download(repo,name,revision=revision,cache_dir=str(hub)))
            records.append({'repo':repo,'revision':revision,'file':name,'bytes':path.stat().st_size})
            print(f'Cached image component {repo}/{name}',flush=True)
    # The original CLI's model name is an upstream redirect. Make that exact
    # name resolve to the same selected snapshot when Hugging Face is offline.
    alias=hub/'models--JeffreyXiang--TRELLIS-image-large'
    target=hub/'models--microsoft--TRELLIS-image-large'
    if not alias.exists() and not alias.is_symlink():alias.symlink_to(target.name,target_is_directory=True)
    elif alias.resolve()!=target.resolve():
        raise ValueError('The selected image cache already contains a separate JeffreyXiang snapshot. Choose a fresh cache directory.')

    import torch
    os.environ['TORCH_HOME']=str(torch_cache)
    torch.hub.set_dir(str(torch_cache/'hub'))
    model=torch.hub.load(dinov2_hub_repository(),pins['dinov2']['model'],pretrained=True,
                         trust_repo=True,skip_validation=True)
    del model
    checkpoint=torch_cache/'hub/checkpoints'/pins['dinov2']['checkpoint']
    if file_hash(checkpoint)!=pins['dinov2']['sha256']:
        raise ValueError('DINOv2 checkpoint does not match the recorded SHA256.')
    records.append({'torch_hub':pins['dinov2'],'checkpoint':str(checkpoint)})
    os.environ['U2NET_HOME']=str(u2net_cache)
    from rembg.sessions.u2net import U2netSession
    background=Path(U2netSession.download_models())
    if file_hash(background,'md5')!=pins['u2net']['md5']:
        raise ValueError('U2Net checkpoint does not match the checksum in rembg 2.0.69.')
    records.append({'u2net':pins['u2net'],'checkpoint':str(background),'sha256':file_hash(background)})
    return {'dinov2':pins['dinov2'],'torch_cache':str(torch_cache),
            'u2net':pins['u2net'],'u2net_cache':str(u2net_cache),
            'trellis_image_revision':revision}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,default=REPO/'spaceflow_runtime/huggingface')
    parser.add_argument('--include-image',action='store_true',help='Also stage the image pipeline, pinned DINOv2 source/weights, and U2Net.')
    parser.add_argument('--torch-cache-dir',type=Path,default=Path(os.environ.get('TORCH_HOME',REPO/'spaceflow_runtime/torch')))
    parser.add_argument('--u2net-cache-dir',type=Path,default=Path(os.environ.get('U2NET_HOME',REPO/'spaceflow_runtime/u2net')))
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
    marker={'revisions':revisions,'files':records}
    if args.include_image:
        marker['image']=stage_image_models(cache,pins,records,args.torch_cache_dir.expanduser().resolve(),
                                          args.u2net_cache_dir.expanduser().resolve())
    (cache/'spaceflow-models.json').write_text(json.dumps(marker,indent=2)+'\n')
    print(f'Model cache ready: {cache}',flush=True)
    print('Set HF_HOME to this directory and HF_HUB_OFFLINE=1 for the replay.',flush=True)
    if args.include_image:
        print(f"Set TORCH_HOME={marker['image']['torch_cache']} and U2NET_HOME={marker['image']['u2net_cache']} for image conditioning.",flush=True)

if __name__=='__main__':main()
