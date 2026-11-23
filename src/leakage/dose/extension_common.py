"""Additive analyses: frozen inputs and evidence are never overwritten."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import json, shutil, stat
import numpy as np
import pandas as pd
from supplemental_common import sha, digest, write_json

SUPPLEMENTAL_SHA='8d6fe11c06d5ce9af25f227d99fd63fdc4854dba924e01a584b0126efad4c5f5'

def check_files(folder, meta, weights=False):
    folder=Path(folder).resolve()
    for name,r in meta['files'].items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder):raise ValueError('Result path escaped its job.')
        if p.suffix=='.pt' and not weights and not p.exists():continue
        if not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:
            raise ValueError('Missing or changed artifact: '+str(p))

def seal(folder, names):
    return {n:{'bytes':(Path(folder)/n).stat().st_size,'sha256':sha(Path(folder)/n)} for n in names}

def verify_extension(root, expected):
    root=Path(root)
    if sha(root/'extension_manifest.json')!=expected:raise ValueError('Additional-check input manifest changed.')
    m=json.loads((root/'extension_manifest.json').read_text())
    check_files(root,m,True)
    if m['supplemental_manifest_sha256']!=SUPPLEMENTAL_SHA:raise ValueError('Wrong parent study.')
    return json.loads((root/'extension_plan.json').read_text())

def baseline(extension, seed, condition):
    folder=Path(extension)/f'baseline/seed_{seed}/{condition}'
    meta=json.loads((folder/'ablation_complete.json').read_text())
    check_files(folder,meta)
    with np.load(folder/'test_predictions.npz',allow_pickle=False) as z:pred={k:z[k] for k in z.files}
    return meta,pred

def dose_frames(inputs, extension, seed, dose):
    clean=pd.read_csv(Path(inputs)/f'ablation/seed_{seed}/clean.csv')
    eight=pd.read_csv(Path(inputs)/f'ablation/seed_{seed}/contaminated.csv')
    test=pd.read_csv(Path(inputs)/f'ablation/seed_{seed}/test.csv')
    frame=pd.read_csv(Path(extension)/f'doses/seed_{seed}/dose_{dose}.csv')
    if dose not in (2,4) or len(frame)!=len(clean):raise ValueError('Wrong dose or training budget.')
    if not np.array_equal(frame.slot,clean.slot) or not np.array_equal(frame.label,clean.label):raise ValueError('Slots/class sequence changed.')
    inserted=frame.intentional_contaminant.astype(str).str.lower().eq('true')
    original=eight.intentional_contaminant.astype(str).str.lower().eq('true')
    if (inserted & ~original).any():raise ValueError('New exposure outside original eight-relative intervention.')
    pd.testing.assert_frame_equal(frame.loc[inserted].reset_index(drop=True),eight.loc[inserted].reset_index(drop=True))
    pd.testing.assert_frame_equal(frame.loc[~inserted].reset_index(drop=True),clean.loc[~inserted].reset_index(drop=True))
    counts=frame.loc[inserted].groupby('filename').size()
    if set(counts.index)!=set(test.filename) or not counts.eq(dose).all():raise ValueError('Dose is not exact for every original test image.')
    if frame.loc[inserted,'variant'].eq(0).any() or clean.loc[inserted,'variant'].eq(0).any():raise ValueError('Unaugmented originals must not be replaced/inserted.')
    if set(clean.patient_cluster)&set(test.patient_cluster):raise ValueError('Clean group overlap.')
    return clean,frame,test

def copy_prior(search,out,marker):
    """Cumulative full output stays small (<~5GB); conflicting archives are errors."""
    out=Path(out).resolve();found=0
    for p in sorted(Path(search).rglob(marker)):
        src=p.parent
        if src.resolve()==out:continue
        for f in src.rglob('*'):
            if not f.is_file() or f.name.endswith('.tmp'):continue
            d=out/f.relative_to(src);d.parent.mkdir(parents=True,exist_ok=True)
            if d.exists() and sha(d)!=sha(f):raise ValueError('Conflicting prior output: '+str(d))
            if not d.exists():shutil.copyfile(f,d)
        found+=1
    return found

def safe_extract(archive,destination):
    destination=Path(destination).resolve()
    with ZipFile(archive) as z:
        if z.testzip() is not None or sum(i.file_size for i in z.infolist())>2*2**30:raise ValueError('Invalid result ZIP.')
        for i in z.infolist():
            rel=PurePosixPath(i.filename.replace('\\','/'));p=destination.joinpath(*rel.parts).resolve()
            if rel.is_absolute() or '..' in rel.parts or any(':' in t for t in rel.parts) or stat.S_ISLNK(i.external_attr>>16) or not p.is_relative_to(destination):raise ValueError('Unsafe ZIP.')
            if i.is_dir():p.mkdir(parents=True,exist_ok=True)
            else:
                p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(i) as a,p.open('wb') as b:shutil.copyfileobj(a,b)
