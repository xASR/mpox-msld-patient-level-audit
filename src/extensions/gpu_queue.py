import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
import gc, shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from package_io import get,put,sha,identity,finish
from gpu_study import base,images,frames,evaluate,train

def discover_weights(pkg,search):
    wanted=get(pkg/'weights.json')
    byname={}
    for p in search.rglob('*.pt'):
        byname.setdefault(p.name,[]).append(p)
    found={}; report=[]
    for r in wanted:
        paths=byname.get(r['filename'],[])
        path=next((p for p in paths if sha(p)==r['sha256']),None)
        if path:found[r['arm'],r['fold'],r['rank']]=path
        report.append(dict(**r,matched=path is not None,candidates=len(paths)))
    return found,report

def historical_pixels(pkg,paths):
    from PIL import Image
    from torchvision import transforms as T
    import hashlib
    import pandas as pd
    augment=T.Compose([T.RandomRotation(30),T.RandomAffine(0,translate=(.1,.1),scale=(.8,1.2),shear=15),T.ColorJitter(.15,.2,.15,.1)])
    results=[]
    for row in pd.read_csv(pkg/'design/vit_transform_control.csv').itertuples():
        with Image.open(paths[row.filename]) as f:image=f.convert('RGB')
        if row.variant:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(row.augmentation_seed))
                image=augment(image)
        pixels=T.PILToTensor()(T.Resize((128,128))(image))
        actual=hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
        results.append(dict(filename=row.filename,variant=int(row.variant),actual=actual,expected=row.input_uint8_sha256))
    return results

def crossover(pkg,out,paths,workers,found,deadline):
    a,b=workers
    for arm in ('B','A'):
        for fold in range(1,6):
            dest=out/f'cross_{arm}_{fold}'
            if (dest/'complete.json').exists():continue
            keys=[(arm,fold,k) for k in (1,2)]
            if not all(k in found for k in keys):continue
            if time.time()+240>deadline:return False
            dest.mkdir(exist_ok=True)
            model,_=base.network(get(pkg/'design/protocol_a_manifest.json')['models']['efficientnetv2_s'],42+fold,False,'cuda:0')
            parts=frames(pkg,paths,fold,True)
            evaluate(model,[found[k] for k in keys],parts,a,b,[arm],dest)
            errors={}
            for side in ('val','test'):
                ref=np.load(pkg/f'ref_{arm}_{fold}_{side}.npz')
                new=np.load(dest/f'{arm}_{side}.npz')
                assert np.array_equal(ref['pids'],new['pids']) and np.array_equal(ref['labels'],new['labels'])
                errors[side]=float(np.abs(ref['probs']-new['probs']).max())
                if errors[side]>2e-5:raise ValueError(f'Historical native inference mismatch in {arm}/{fold}/{side}: {errors[side]}')
            evaluate(model,[found[k] for k in keys],parts,a,b,['B' if arm=='A' else 'A'],dest)
            put(dest/'complete.json',dict(arm=arm,fold=fold,native_max_error=errors))
            del model;gc.collect();torch.cuda.empty_cache()
    return True

def main(pkg,out,search,owner,deadline):
    start=time.time();out.mkdir(parents=True,exist_ok=True)
    manifest=identity(pkg,out)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Select a Tesla T4 accelerator; this worker exposes only device 0.')
    weights={}
    if owner=='Me':
        weights,report=discover_weights(pkg,search)
        put(out/'crossover_inventory.json',report)
        n=sum(r['matched'] and r['arm']=='A' for r in report)
        print('Aligned historical tensors:',n,'/10. All independently available pairs will run.',flush=True)
    paths=images(pkg,search)
    controls=historical_pixels(pkg,paths)
    put(out/'historical_pixel_controls.json',controls)
    if not all(v['actual']==v['expected'] for v in controls):
        raise RuntimeError('Historical transform controls differ. Select the matched Kaggle image (torch 2.11, torchvision 0.26, Pillow 13).')
    sys.path.insert(0,str(pkg/'design'))
    import original_worker_v34 as b
    a=base.original_worker(pkg/'design')
    torch.set_num_threads(1)
    if owner=='Me' and not crossover(pkg,out,paths,(a,b),weights,deadline):
        finish(out,owner,time.time()-start,False);return
    jobs=[j for j in get(pkg/'schedule.json') if j['owner']==owner]
    firstseed={'Me':344,'Friend1':346,'Friend2':347}[owner]
    jobs.sort(key=lambda j:(0 if j['family']=='beit' and j['seed']==firstseed else
                            1 if j.get('arm')=='2a' else 2 if j['family']=='dose' else 3))
    for job in jobs:
        folder=out/job['id'];folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():
            done=get(folder/'complete.json')
            assert done['manifest']==manifest and done['job']==job
            assert done['epochs']==25 or (job['family']=='beit' and done.get('outcome')=='diverged')
            for name,h in done['weights'].items():assert sha(folder/name)==h
            continue
        if time.time()+600>=deadline:break
        torch.use_deterministic_algorithms(job['family']!='dose',warn_only=True)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        complete=train(pkg,job,folder,paths,(a,b),deadline,manifest)
        gc.collect();torch.cuda.empty_cache()
        if not complete:break
    finished=[j['id'] for j in jobs if (out/j['id']/'complete.json').exists()]
    put(out/'queue_status.json',dict(assigned=[j['id'] for j in jobs],finished=finished))
    finish(out,owner,time.time()-start,len(finished)==len(jobs))

if __name__=='__main__':
    pkg,out,search=map(Path,sys.argv[1:4])
    main(pkg,out,search,sys.argv[4],float(sys.argv[5]))

