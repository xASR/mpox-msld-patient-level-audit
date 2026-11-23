"""Collect the forty selected inference weights from full saved notebook outputs.

Reads extracted folders or downloaded ZIPs. Leaves all source files unchanged.
Does not accept the two rolling resume checkpoints in place of selected weights.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import shutil
import sys
from zipfile import ZipFile, ZIP_STORED

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def read_required(archive):
    with ZipFile(archive) as z:
        expected={}
        for name in z.namelist():
            if name.endswith('/fold_complete.json'):
                meta=json.loads(z.read(name))
                if meta.get('schema')!='pb_known_duplicate_corrected_v1' or meta.get('epochs_completed')!=25:
                    raise ValueError('Wrong result input.')
                for w in meta['weights']:expected[w['file']]={'model':meta['model'],'fold':meta['fold'],**w}
    if len(expected)!=40:raise ValueError('Expected all forty selected inference weights.')
    return expected

def collect(source,verified_input,output):
    source=Path(source).resolve();output=Path(output).resolve()
    required=read_required(verified_input);found={};missing=set(required)
    if not source.is_dir():raise FileNotFoundError(source)
    # Prefer direct files, then stream candidates inside full download ZIPs.
    for p in sorted(source.rglob('*.pt')):
        name=p.name
        if name in missing and p.stat().st_size==required[name]['bytes'] and digest(p)==required[name]['sha256']:
            found[name]={'path':p,'archive':None,'member':None};missing.remove(name)
    for archive in sorted(source.rglob('*.zip')):
        if archive==output or archive.name.startswith('PB_Ensemble_Selected_Weights'):continue
        with ZipFile(archive) as z:
            for entry in z.infolist():
                name=Path(entry.filename.replace('\\','/')).name
                if name not in missing or entry.file_size!=required[name]['bytes']:continue
                h=hashlib.sha256()
                with z.open(entry) as f:
                    for b in iter(lambda:f.read(2**20),b''):h.update(b)
                if h.hexdigest()==required[name]['sha256']:
                    found[name]={'path':archive,'archive':archive,'member':entry.filename};missing.remove(name)
        if not missing:break
    if missing:
        print('\nNo upload ZIP was created. Still missing matching selected weights:',flush=True)
        for name in sorted(missing):print('  '+name,flush=True)
        print('\nDownload the full saved outputs for those model/fold jobs, including BEiT fold2 and fold3 recovery versions. Then run this helper again.',flush=True)
        return False
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix('.building.zip')
    if temp.exists():raise FileExistsError(f'A previous temporary file exists: {temp}. Rename it before retrying.')
    rows=[]
    try:
        with ZipFile(temp,'w',compression=ZIP_STORED,allowZip64=True) as z:
            for name in sorted(required):
                item=found[name];h=hashlib.sha256();target='PB_Ensemble_Selected_Weights/weights/'+name
                if item['archive'] is None:
                    with item['path'].open('rb') as f,z.open(target,'w',force_zip64=True) as out:
                        for b in iter(lambda:f.read(2**20),b''):h.update(b);out.write(b)
                else:
                    with ZipFile(item['archive']) as original,original.open(item['member']) as f,z.open(target,'w',force_zip64=True) as out:
                        for b in iter(lambda:f.read(2**20),b''):h.update(b);out.write(b)
                if h.hexdigest()!=required[name]['sha256']:raise ValueError('A source weight changed while it was copied: '+name)
                rows.append({**required[name],'relative_path':'weights/'+name})
                print(f'Verified and packed {len(rows)}/40: {name}',flush=True)
            z.writestr('PB_Ensemble_Selected_Weights/pb_selected_weights_manifest.json',json.dumps({
                'schema':'pb_selected_weights_v1','weight_count':40,
                'source_verified_input_zip_sha256':digest(verified_input),'weights':rows},indent=2))
        if output.exists():raise FileExistsError(f'Output already exists: {output}. Keep or rename it before retrying.')
        temp.replace(output)
    except Exception:
        # Keep a failed temporary ZIP for inspection; never alter source files.
        raise
    print('\nALL 40 SELECTED WEIGHTS VERIFIED',flush=True)
    print('Upload this ZIP as a private Kaggle dataset: '+str(output),flush=True)
    return True

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--verified-input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if not collect(a.source,a.verified_input,a.output):raise SystemExit(2)

if __name__=='__main__':main()
