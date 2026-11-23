import hashlib,json,os,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def put(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)
def record(p):return {'sha256':sha(p),'bytes':p.stat().st_size}
def check(p,r):
    if not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('File mismatch: '+str(p))
def verify_package(package,expected):
    manifest=package/'revision_manifest.json'
    if sha(manifest)!=expected:raise ValueError('Unexpected revision input package.')
    for rel,rec in read(manifest)['files'].items():
        p=(package/rel).resolve()
        if not p.is_relative_to(package.resolve()):raise ValueError(rel)
        check(p,rec)
def locate_data(package,search,need_aug=True,verify=True):
    pool=pd.read_csv(package/'design/pool.csv')
    if not need_aug:pool=pool[pool.is_original]
    roots={};verified={}
    for mark in (['FOLDS','FOLDS_AUG'] if need_aug else ['FOLDS']):
        probe=next(r for r in pool.relative if mark in Path(r).parts)
        tail=Path(*Path(probe).parts[Path(probe).parts.index(mark)+1:])
        candidates=sorted(p for p in Path(search).rglob(mark) if p.is_dir() and (p/tail).is_file())
        if len(candidates)!=1:raise ValueError(f'Expected one MSLD {mark} directory with {tail}; found {len(candidates)}. Attach the original release once.')
        roots[mark]=candidates[0]
    def resolve(row):
        rel=Path(row.relative);mark='FOLDS' if row.is_original else 'FOLDS_AUG';p=roots[mark].joinpath(*rel.parts[rel.parts.index(mark)+1:])
        if verify:check(p,dict(bytes=row.bytes,sha256=row.sha256))
        return row.relative,str(p)
    with ThreadPoolExecutor(max_workers=8) as ex:verified.update(ex.map(resolve,pool.itertuples(index=False)))
    return verified
def check_done(folder,job,manifest,weights=False):
    d=read(folder/'complete.json')
    if d['job']!=job or d['manifest']!=manifest or d['status']!='complete' or d['epochs']!=(100 if job['model']=='vit' else 25):
        raise ValueError('Invalid finished job: '+job['id'])
    m,f=job['model'],job['index']
    if job['family'] in ('release','matched'):
        needed={'initial.json','history.json','final_weights.pt','test.npz'}
        needed.add('diagnostic.npz' if job['family']=='matched' else 'training_items_seen.npy')
    else:
        needed={'initial.json','history.json',f'{m}_fold{f}_rank1.pt',f'{m}_fold{f}_rank2.pt'}
        needed.update(f'{m}_fold{f}_{split}_{key}.npy' for split in ('val','test') for key in ('labels','pids','probs'))
    if not needed.issubset(d['files']):raise ValueError('Missing registered results or weights for '+job['id'])
    for name,rec in d['files'].items():
        if Path(name).name!=name:raise ValueError('Invalid result filename')
        if weights or not name.endswith('.pt'):check(folder/name,rec)
    return d

