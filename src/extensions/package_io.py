from pathlib import Path
import hashlib, json, os, platform, sys, time, zipfile
from importlib.metadata import version,PackageNotFoundError

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def put(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
    os.replace(temp,path)

def get(path):return json.loads(Path(path).read_text(encoding='utf8'))

def identity(package,out):
    name=sha(package/'extension_manifest.json')
    meta=out/'input_identity.json'
    env={'python':platform.python_version()}
    for pkg in ('numpy','scipy','scikit-learn','pandas','Pillow','torch','torchvision','timm'):
        try:env[pkg]=version(pkg)
        except PackageNotFoundError:pass
    info=dict(input_manifest_sha256=name,environment=env,worker=out.name)
    if meta.exists() and get(meta)!=info:raise ValueError('Runtime or input changed on resume.')
    put(meta,info)
    return name

def finish(out,who,seconds,complete=True):
    files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*'))
           if p.is_file() and p.suffix not in ('.pt','.tmp') and '_resume_' not in p.name and p!=out/'result_index.json'}
    put(out/'result_index.json',dict(worker=who,complete=bool(complete),seconds_this_session=seconds,files=files))
    archive=out.parent/f'EXT_{who}_RESULTS_ONLY.zip'
    tmp=archive.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for name in [*files,'result_index.json']:z.write(out/name,name)
    os.replace(tmp,archive)
    print(f'{who} complete={complete}; results: {archive}',flush=True)

