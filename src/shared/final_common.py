"""Pinned inputs and common routines for the final corrected Protocol B analysis."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import stat
import sys
from zipfile import ZipFile
import numpy as np

FROZEN_SHA = '856d1724988275704854984023a5c9be4d39869fb02187bcd011118c0de20200'
MODELS = ('efficientnetv2_s','convnext_tiny','swin_small','beit_base')
DISPLAY = {'efficientnetv2_s':'EfficientNetV2-S','convnext_tiny':'ConvNeXt-Tiny',
           'swin_small':'Swin-Small','beit_base':'BEiT-Base','ensemble':'Ensemble'}
CLASSES = ['Healthy','Cowpox','Monkeypox','Measles','Chickenpox','HFMD']
LABELS = ['Healthy','Cowpox','Mpox','Measles','Chickenpox','HFMD']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''): h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def extract_small(archive,destination):
    destination=Path(destination).resolve()
    with ZipFile(archive) as zipped:
        if sum(i.file_size for i in zipped.infolist())>100*2**20: raise ValueError('Expected the small verified input ZIP.')
        for info in zipped.infolist():
            rel=PurePosixPath(info.filename.replace('\\','/'))
            if rel.is_absolute() or '..' in rel.parts or any(':' in p for p in rel.parts) or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError('Unsafe ZIP path.')
            target=destination.joinpath(*rel.parts).resolve()
            if not target.is_relative_to(destination): raise ValueError('ZIP path leaves its destination.')
            if info.is_dir():target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with zipped.open(info) as source,target.open('wb') as output:shutil.copyfileobj(source,output)

def prepare_inputs(search_root,work,expected_sha):
    search_root,work=Path(search_root),Path(work)
    candidates=list(search_root.rglob('pb_final_verified_manifest.json'))
    if not any(sha(p)==expected_sha for p in candidates):
        for archive in search_root.rglob('PB_Final_Verified_Inputs.zip'):
            cache=work/'input_cache'/sha(archive);extract_small(archive,cache)
            candidates.extend(cache.rglob('pb_final_verified_manifest.json'))
    candidates=[p for p in candidates if sha(p)==expected_sha]
    if not candidates: raise FileNotFoundError('Attach the NEW PB_Final_Verified_Inputs.zip. The old twelve archives are already combined in it.')
    marker=sorted(candidates)[0];root=marker.parent
    index=json.loads(marker.read_text())
    if index['schema']!='pb_final_verified_inputs_v1' or index['frozen_manifest_sha256']!=FROZEN_SHA:
        raise ValueError('Unexpected protocol inputs.')
    for rel,record in index['files'].items():
        target=(root/rel).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file() or sha(target)!=record['sha256'] or target.stat().st_size!=record['bytes']:
            raise ValueError('Verified input file missing or changed: '+rel)
    sys.path.insert(0,str(root/'frozen/code'))
    from corrected_core import verify_inputs
    verify_inputs(root/'frozen',FROZEN_SHA)
    return root

def read_results(root):
    import pandas as pd
    from corrected_core import verify_inputs,check_group_separation,pool_predictions
    from merge_corrected import load_completed,prediction
    root=Path(root);frozen=root/'frozen';manifest=verify_inputs(frozen,FROZEN_SHA)
    records=load_completed(root/'results',FROZEN_SHA)
    pooled={};splits={}
    for f in range(1,6):
        frame=pd.read_csv(frozen/f'splits/fold{f}.csv')
        check_group_separation(frame,pd.read_csv(frozen/f'augmented/fold{f}.csv'))
        splits[f]=frame
    for model in MODELS:
        pooled[model]={}
        for token,role in [('test','test'),('val','val_calibration')]:
            parts=[]
            for f in range(1,6):
                rec=records[(model,f)];meta=rec['meta'];ident=meta['identity']
                if hashlib.sha256(json.dumps(ident,sort_keys=True,allow_nan=False).encode()).hexdigest()!=meta['identity_sha256']:
                    raise ValueError('Job identity digest differs.')
                if ident['training_seed']!=42+100*MODELS.index(model)+f or ident['training']!=manifest['training'] or ident['pretrained_source']!=manifest['models'][model]:
                    raise ValueError('Job recipe differs from the frozen rerun.')
                part=prediction(rec,model,f,token);a=splits[f];g=a[a.role.eq(role)]
                expected=np.sort(g.patient_cluster.unique());labelmap=g.groupby('patient_cluster').label.first().to_dict()
                if not np.array_equal(part['pids'],expected) or not np.array_equal(part['labels'],[labelmap[p] for p in expected]):
                    raise ValueError('Prediction cohort differs from frozen assignments.')
                parts.append(part)
            pooled[model][token]=pool_predictions(parts)
    ref=pooled[MODELS[0]]
    for model in MODELS:
        for token in ['test','val']:
            if any(not np.array_equal(pooled[model][token][key],ref[token][key]) for key in ['pids','labels']):
                raise ValueError('Models are not aligned.')
    keep=~np.isin(ref['val']['pids'],ref['test']['pids'])
    if (len(ref['test']['pids']),len(ref['val']['pids']),int(keep.sum()))!=(215,217,138):
        raise ValueError('Unexpected pooled cohorts.')
    return manifest,records,pooled,splits,keep
