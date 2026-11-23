"""Run two independent model jobs per T4 ×2 session, then preserve a cumulative account output."""
import argparse,os,shutil,subprocess,sys,tempfile,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import pandas as pd
from revision_io import read,put,sha,record,check_done,verify_package,locate_data

def validate_splits(package):
    p=read(package/'plan.json');jobs=p['jobs'];ids=[j['id'] for j in jobs]
    if len(ids)!=46 or len(set(ids))!=len(ids):raise ValueError('Unexpected queue')
    pool=pd.read_csv(package/'design/pool.csv');assert len(pool)==read(package/'design/pool_inventory.json')['canonical_items']
    assert pool.is_original.sum()==755 and pool.group.nunique()==520 and not pool.duplicated(['source','rgb']).any()
    for seed in range(5):
        s=pd.read_csv(package/f'design/release_s{seed}.csv');assert np.array_equal(s.item,pool.item)
        for col in ['random_test','group_test']:
            tr=s[col].eq(0);te=~tr; assert set(s[col])=={0,1} and tr.sum()>=p['release_epoch_slots'][str(seed)] and set(pool.label[te])==set(range(6))
            if col=='group_test':assert not set(pool[tr].group)&set(pool[te].group)
    for seed in range(5,10):
        df=[pd.read_csv(package/f'design/matched_s{seed}_{a}.csv') for a in ['clean','contaminated','test','diagnostic']]
        clean,dirty,test,diag=df
        assert np.array_equal(clean.label,dirty.label) and not set(clean.group)&set(test.group)
        assert len(set(clean.group))==416 and len(set(test.group))==104 and dirty.source.isin(test.source).sum()==8*len(test)
        assert np.array_equal(clean[clean.variant==0].source,dirty[dirty.variant==0].source)
        assert set(zip(dirty.source,dirty.variant)).isdisjoint(zip(diag.source,diag.variant))
    for f in range(1,6):
        for who in ['a','b']:
            df=pd.read_csv(package/f'design/{who}_fold{f}.csv')
            df=df[~df.role.str.startswith('excluded')];assert df.groupby('patient_cluster').role.nunique().max()==1
    return p
def transform_check(pkg,paths):
    from PIL import Image
    import torch,hashlib
    from torchvision import transforms as T
    pool=pd.read_csv(pkg/'design/pool.csv'); lookup=pool[pool.is_original].set_index('source').relative
    aug=T.Compose([T.RandomRotation(30),T.RandomAffine(0,translate=(.1,.1),scale=(.8,1.2),shear=15),T.ColorJitter(.15,.2,.15,.1)])
    checks=[]
    for r in pd.read_csv(pkg/'design/vit_transform_control.csv').itertuples():
        with Image.open(paths[lookup[r.filename]]) as raw:im=raw.convert('RGB')
        if r.variant:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(r.augmentation_seed));im=aug(im)
        t=T.PILToTensor()(T.Resize((128,128))(im))
        checks.append((r.filename,int(r.variant),hashlib.sha256(t.numpy().tobytes()).hexdigest(),r.input_uint8_sha256))
    return checks

def previous_output(search,owner,manifest,required):
    roots=[]
    for p in search.rglob('account_progress.json'):
        d=read(p)
        if d.get('owner')==owner and d.get('manifest')==manifest:roots.append(p.parent)
    if len(roots)>1:raise ValueError('Attach only the latest full saved output for this account.')
    if required and not roots:raise FileNotFoundError('Attach the full saved version containing account_progress.json, job folders and weight/checkpoint files.')
    return roots[0] if roots else None
def restore(previous,root,jobs,manifest):
    for j in jobs:
        src=previous/j['id'];dst=root/j['id']
        if not src.is_dir():continue
        if (src/'complete.json').is_file():check_done(src,j,manifest,weights=True)
        elif (src/'status.json').is_file() and read(src/'status.json').get('completed_epochs',0)>0:
            if not any(p.with_suffix('.pt').is_file() for p in src.glob('revision_fold*_resume_?.json')):
                raise ValueError('Full checkpoint tensors are missing for '+j['id'])
        if dst.exists():raise ValueError('Resume from a fresh saved session to avoid mixing output generations.')
        shutil.copytree(src,dst,ignore=shutil.ignore_patterns('*.tmp'))
def archive(root):
    target=root.parent/(root.name+'_RESULTS_ONLY.zip');tmp=target.with_suffix('.partial')
    with ZipFile(tmp,'w',compression=ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix not in ('.pt','.tmp') and '_resume_' not in p.name:z.write(p,p.relative_to(root.parent))
    os.replace(tmp,target)
def main(x):
    verify_package(x.package,x.manifest);plan=validate_splits(x.package)
    jobs=[j for j in plan['jobs'] if j['owner']==x.owner];checkall=x.owner=='cpu'
    scratch=Path(tempfile.gettempdir())/'msld_revision_scratch'/x.manifest
    scratch.mkdir(parents=True,exist_ok=True)
    images=locate_data(x.package,x.search,need_aug=checkall or any(j['family']=='release' for j in jobs))
    controls=transform_check(x.package,images)
    failed_controls=[(name,variant) for name,variant,observed,expected in controls if observed!=expected]
    if failed_controls:
        import torch,torchvision,PIL
        raise RuntimeError(f'Historical pixel control mismatch {failed_controls}. Torch {torch.__version__}, torchvision {torchvision.__version__}, Pillow {PIL.__version__}. Use the original Kaggle environment, torch 2.11.0 and torchvision 0.26.0, with the notebook-pinned image library. No jobs started.')
    print(f'Historical image controls: {len(controls)}/{len(controls)}.',flush=True)
    if checkall:print('CPU PREFLIGHT PASS; release image content, historical transform controls and all 46 assignments verified.',flush=True);return
    put(scratch/'image_paths.json',images);root=x.work/('REV_'+x.owner)
    root.mkdir(parents=True,exist_ok=True)
    prev=previous_output(x.search,x.owner,x.manifest,x.resume)
    if prev:restore(prev,root,jobs,x.manifest)
    import torch
    ng=torch.cuda.device_count()
    if ng<1:raise RuntimeError('Enable GPU T4 ×2 in notebook settings.')
    ng=min(2,ng);print(f'PREFLIGHT PASS. {len(jobs)} jobs assigned; {ng} independent GPU worker(s).',flush=True)
    failed={};paused=set();done=set()
    for j in jobs:
        if (root/j['id']/'complete.json').exists():check_done(root/j['id'],j,x.manifest,weights=True);done.add(j['id'])
    pending=[j for j in jobs if j['id'] not in done]
    def status():
        states={}
        for j in jobs:
            p=root/j['id']/'status.json';states[j['id']]=read(p) if p.exists() else {'status':'waiting'}
        put(root/'account_progress.json',dict(owner=x.owner,manifest=x.manifest,complete=len(done)==len(jobs),completed=sorted(done),
            failed=failed,paused=sorted(paused),jobs=states,expected=len(jobs)))
    def child(j,gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        cmd=[sys.executable,str(x.package/'code/train_revision.py'),'--id',j['id'],'--work',str(root),'--package',str(x.package),
             '--paths',str(scratch/'image_paths.json'),'--scratch',str(scratch),'--deadline',str(x.deadline),'--manifest',x.manifest]
        folder=root/j['id'];folder.mkdir(exist_ok=True);log=folder/'worker.log'
        print('Starting',j['id'],'on GPU',gpu,flush=True)
        with log.open('a',encoding='utf-8') as f,subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1) as proc:
            for line in proc.stdout:
                f.write(line);f.flush()
                if 'Checkpoint saved' not in line:print(line.rstrip(),flush=True)
            return proc.wait()
    def discard_finished_cache(j):
        if j['family'] not in ('release','matched'):return
        related=[k for k in jobs if k['family']==j['family'] and k['model']==j['model'] and (j['family']=='release' or k['index']==j['index'])]
        if not all(k['id'] in done for k in related):return
        tag=j['family']+'_'+j['model']+('_s'+str(j['index']) if j['family']=='matched' else '')
        for suffix in ['.npy','.hashes.json']:(scratch/(tag+suffix)).unlink(missing_ok=True)
    status();active={};free=list(range(ng))
    with ThreadPoolExecutor(max_workers=ng) as ex:
        while pending or active:
            while pending and free and not failed and not paused and time.time()+600<x.deadline:
                j=pending.pop(0);gpu=free.pop(0);active[ex.submit(child,j,gpu)]=(j,gpu)
            if not active:break
            for fut in wait(active,return_when=FIRST_COMPLETED).done:
                j,gpu=active.pop(fut);free.append(gpu);code=fut.result()
                if code==0:
                    check_done(root/j['id'],j,x.manifest,weights=True);done.add(j['id']);discard_finished_cache(j)
                elif code==75:paused.add(j['id'])
                else:failed[j['id']]=dict(exit_code=code,log=j['id']+'/worker.log')
                status();archive(root)
    status();archive(root)
    print('ACCOUNT COMPLETE:',len(done)==len(jobs),f'({len(done)}/{len(jobs)}).',flush=True)
    if failed:print('Worker failed. See the named worker.log and keep full outputs.',failed,flush=True)
    elif len(done)<len(jobs):print('Paused at the session budget. Save full outputs and use the account resume notebook.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--owner',choices=['alve','friend1','friend2','cpu'],required=True)
    for n in ['package','search','work']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--deadline',type=float,default=0);p.add_argument('--manifest',required=True);p.add_argument('--resume',action='store_true');main(p.parse_args())

