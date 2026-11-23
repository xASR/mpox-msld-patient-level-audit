"""One active Tesla T4. Frozen historical workers supply transforms, inference and optimizer."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gc,hashlib,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Sampler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from sklearn.metrics import accuracy_score,f1_score,log_loss
import protocol_a_train as base
from protocol_a_core import aggregate_images
from checkpoint_engine import CheckpointManager,capture_rng,restore_rng,seed_everything,cpu_copy,atomic_torch_save
from fixed_ancestry import FixedAncestryDataset
from package_io import get,put,sha,identity,finish

def state_hash(model):
    h=hashlib.sha256()
    for key,val in sorted(model.state_dict().items()):
        x=val.detach().cpu().contiguous()
        for part in (key,str(x.dtype),str(tuple(x.shape))):h.update(part.encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()

def images(package,search):
    rows=pd.read_csv(package/'originals.csv')
    roots=list(search.rglob('FOLDS'))
    result={}
    for r in rows.itertuples():
        parts=Path(r.relative).parts; tail=parts[parts.index('FOLDS')+1:]
        paths=[f.joinpath(*tail) for f in roots if f.joinpath(*tail).is_file()]
        match=next((p for p in paths if sha(p)==r.sha256),None)
        if match is None:raise FileNotFoundError('Missing original with frozen hash: '+r.source)
        result[r.source]=str(match)
    return result

def frames(pkg,paths,fold,official):
    prefix='b' if official else 'a'
    d=pd.read_csv(pkg/f'design/{prefix}_fold{fold}.csv')
    if official:
        d.role=d.role.map(dict(train_original='train',val_selection='selection',val_calibration='calibration',
                               test='test',excluded_known_group_overlap='excluded'))
    assert d.role.notna().all()
    d['filepath']=d.filename.map(paths);d['patient_id']=d.patient_cluster
    assert d.filepath.notna().all()
    assert d[d.role!='excluded'].groupby('patient_id').role.nunique().max()==1
    return {k:d[d.role==k].reset_index(drop=True) for k in ('train','selection','calibration','test')}

class Cycles(Sampler):
    def __init__(self,n,count,gen):self.n,self.count,self.gen=n,count,gen
    def __len__(self):return self.count
    def __iter__(self):
        left=self.count
        while left:
            take=min(left,self.n)
            yield from torch.randperm(self.n,generator=self.gen)[:take].tolist()
            left-=take

def load_dose(pkg,job,paths,generator):
    folder=pkg/'dose'/str(job['seed'])
    clean=pd.read_csv(folder/'clean.csv');dirty=pd.read_csv(folder/'contaminated.csv');test=pd.read_csv(folder/'test.csv')
    mask=get(folder/f"mask{job['n']}.json")
    slots=mask['slots'];group_set=set(mask['groups']);frame=clean.copy()
    frame.loc[slots]=dirty.loc[slots]
    expected=test[test.patient_cluster.isin(group_set)]
    assert set(frame.patient_cluster)&set(test.patient_cluster)==group_set
    assert sorted(frame.loc[slots,'filename'])==sorted(expected.filename)
    assert (frame.loc[slots,'variant']>0).all() and np.array_equal(frame.label,clean.label)
    opts=dict(batch_size=16,num_workers=2)
    loader=DataLoader(IndexedAncestry(frame,paths),shuffle=True,drop_last=True,generator=generator,pin_memory=True,**opts)
    return loader,DataLoader(FixedAncestryDataset(test,paths),shuffle=False,**opts),test,group_set,clean.label

class IndexedAncestry(FixedAncestryDataset):
    def __getitem__(self,index):
        image,label,_=super().__getitem__(index)
        return image,label,index

def track_slots(loader,seen):
    for x,y,slots in loader:
        seen[slots]=True
        yield x,y,slots

class ObservedBatches:
    def __init__(self,loader,seen):self.loader,self.seen=loader,seen
    def __len__(self):return len(self.loader)
    def __iter__(self):return track_slots(self.loader,self.seen)

def train_beit_step(net,opt,scaler,mix,hard,soft,epoch,x,y,regime,micro):
    """Mix the complete batch once; weighted CE uses the complete batch's denominator."""
    assert len(y)==16 and 16%micro==0
    use_amp=regime!='fp32'
    opt.zero_grad(set_to_none=True)
    soft_phase=epoch<10
    if soft_phase:x,y=mix(x,y)
    denominator=len(y) if soft_phase else hard.weight[y].sum()
    total=0.
    for start in range(0,len(y),micro):
        xb,yb=x[start:start+micro],y[start:start+micro]
        with torch.autocast('cuda',dtype=torch.float16,enabled=use_amp):
            logits=net(xb)
            if soft_phase:loss=soft(logits,yb)*len(yb)/denominator
            else:loss=torch.nn.functional.cross_entropy(logits,yb,weight=hard.weight,
                           label_smoothing=.1,reduction='sum')/denominator
        scaler.scale(loss).backward()
        total+=float(loss.detach())
    scaler.unscale_(opt)
    grads=[p.grad.detach() for p in net.parameters() if p.grad is not None]
    norms=torch.stack([g.norm() for g in grads])
    l2=float(norms.norm());amax=float(torch.stack([g.abs().amax() for g in grads]).amax())
    nans=bool(torch.stack([torch.isnan(g).any() for g in grads]).any())
    infs=bool(torch.stack([torch.isinf(g).any() for g in grads]).any())
    before=scaler.get_scale()
    if regime=='clip' and np.isfinite(l2):torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
    diverged=(not np.isfinite(total)) or (not use_amp and (nans or infs))
    if not diverged:
        scaler.step(opt);scaler.update()
    after=scaler.get_scale()
    return dict(loss=total,gradient_l2=l2,gradient_max_abs=amax,gradient_nan=nans,gradient_inf=infs,
                loss_nonfinite=not np.isfinite(total),skipped=int(diverged or (use_amp and after<before)),diverged=bool(diverged),
                scale_before=before if use_amp else None,scale_after=after if use_amp else None,
                effective_batch=16,microbatch=micro)

def memory_probe(cfg,job,worker,out):
    # Discard probe optimizer and all updates. Real training initializes afresh.
    model,_=base.network(cfg,job['seed'],False,'cuda:0')
    opt,_=base.optimizer_factory(worker)(model,'full')
    model.train()
    hard=torch.nn.CrossEntropyLoss(weight=torch.ones(6,device='cuda'),label_smoothing=.1)
    amp=job['regime']!='fp32'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    mix=Mixup(mixup_alpha=.2,cutmix_alpha=1.,label_smoothing=.1,num_classes=6)
    torch.cuda.reset_peak_memory_stats()
    for epoch in (0,10):
        x=torch.randn(16,3,384,384,device='cuda');y=torch.arange(16,device='cuda')%6
        rec=train_beit_step(model,opt,scaler,mix,hard,SoftTargetCrossEntropy(),epoch,x,y,job['regime'],4)
        if rec['diverged']:raise RuntimeError('Nonfinite memory-probe computation with synthetic inputs.')
    torch.cuda.synchronize()
    put(out/'memory_probe.json',dict(batches=2,regime=job['regime'],microbatch=4,accum_steps=4,
        peak_allocated=torch.cuda.max_memory_allocated(),device_bytes=torch.cuda.get_device_properties(0).total_memory))
    del x,y,opt,model,scaler
    gc.collect();torch.cuda.empty_cache()

def evaluate(model,snapshots,data,worker,bworker,recipes,out):
    """Run both historical operators, including checkpoint and aggregation order."""
    for side,role in [('val','calibration'),('test','test')]:
        frame=data[role]
        for recipe in recipes:
            source=worker if recipe=='A' else bworker
            tf=source.get_transforms()[1]
            loader=DataLoader(source.MpoxDataset(frame,tf),batch_size=16,num_workers=2,shuffle=False)
            probs=[]
            for path in snapshots:
                model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
                if recipe=='A':p,y,g=worker.validate(model,loader)
                else:
                    logits,y,g=bworker.validate(model,loader,use_tta=True)
                    p,y,g=bworker.get_patient_level_arrays(logits,y,g)
                if probs:
                    assert np.array_equal(y,old_y) and np.array_equal(g,old_g)
                old_y,old_g=y,g
                probs.append(p)
            p=np.mean(probs,axis=0)
            if recipe=='A':
                v=aggregate_images(p,y,g);p,y,g=v['probs'],v['labels'],v['pids']
            np.savez_compressed(out/f'{recipe}_{side}.npz',probs=p,labels=y,pids=np.asarray(g,str))

def dose_score(net,loader,test,group_set,out):
    net.eval()
    with torch.no_grad():
        prob=np.concatenate([net(x.cuda()).softmax(1).cpu().numpy() for x,_,_ in loader])
    y=test.label.to_numpy();g=test.patient_cluster.to_numpy(str);seen=test.patient_cluster.isin(group_set).to_numpy()
    np.savez_compressed(out/'dose.npz',probs=prob,labels=y,groups=g,filenames=test.filename.to_numpy(str),exposed=seen)
    records=[]
    for name,mask in [('all',np.ones(len(y),bool)),('exposed',seen),('unexposed',~seen)]:
        records.append(dict(subset=name,images=int(mask.sum()),groups=int(len(np.unique(g[mask]))),
            accuracy=float(accuracy_score(y[mask],prob[mask].argmax(1))) if mask.any() else None,
            macro_f1=float(f1_score(y[mask],prob[mask].argmax(1),labels=range(6),average='macro',zero_division=0)) if mask.any() else None))
    put(out/'dose_metrics.json',records)

def train(pkg,job,out,paths,workers,deadline,manifest):
    a,b=workers;fam=job['family'];s=job['seed'];modelkey={'beit':'beit_base','eff':'efficientnetv2_s','dose':'convnext_tiny'}[fam]
    init_seed=9000+s if fam=='dose' else s
    cfg=get(pkg/'design/protocol_a_manifest.json')['models'][modelkey]
    meta=dict(job=job,manifest=manifest,environment=base.environment_record())
    manager=CheckpointManager(out,'fit',job.get('fold',s),meta)
    state=manager.load_auto([out]); history=[];best=[];start=0
    if not state and (out/'history.json').exists():raise RuntimeError('Started fit requires full saved checkpoints.')
    if fam=='beit' and not (out/'memory_probe.json').exists():memory_probe(cfg,job,a,out)
    model,prehash=base.network(cfg,init_seed,state is None,'cuda:0')
    ctrl=get(pkg/'controls.json')
    ref=ctrl['beit344' if fam=='beit' else f'eff{job["fold"]}' if fam=='eff' else f'dose{s}']
    if state is None:
        sig=state_hash(model)
        assert prehash==ref['pretrained_file_sha256']
        if fam!='beit' or s==344:assert sig==ref.get('tensor_sha256',ref.get('parameter_sha256'))
        put(out/'init.json',dict(pretrained=prehash,initial=sig,seed=init_seed))
    else:assert (out/'init.json').exists()
    gen=torch.Generator().manual_seed(19000+s if fam=='dose' else init_seed+10000)
    if fam=='dose':
        loader,eval_dose,test,subset,labels=load_dose(pkg,job,paths,gen)
        seed_everything(init_seed)
    else:
        parts=frames(pkg,paths,job['fold'],fam=='eff');evals=base.make_loaders(a,parts,gen)
        loader=evals['train'];labels=parts['train'].label
        if job.get('arm')=='2a':
            batches=[919,868,910,886,914][job['fold']-1]
            loader=DataLoader(loader.dataset,batch_size=16,sampler=Cycles(len(loader.dataset),16*batches,gen),
                              generator=gen,num_workers=2,drop_last=True,pin_memory=True)
    counts=np.bincount(labels,minlength=6);swap=job.get('arm')=='2b'
    weights=(1-.999)/(1-.999**counts) if swap else 1./counts
    # The A balanced CE is n/(6*class_count), including its absolute scale.
    weights=weights/weights.sum()*6 if swap else weights*sum(counts)/6
    hard=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device='cuda'),label_smoothing=.1)
    soft=SoftTargetCrossEntropy()
    mix=Mixup(mixup_alpha=.1 if swap else .2,cutmix_alpha=0 if swap else 1.,label_smoothing=.1,num_classes=6)
    scaler=torch.amp.GradScaler('cuda',enabled=job.get('regime')!='fp32')
    factory=base.optimizer_factory(a);phase=state['phase'] if state else 'head'
    opt,sch=factory(model,phase)
    seen=torch.zeros(len(loader.dataset),dtype=torch.bool) if fam=='dose' else None
    if state:
        start=state['next_epoch'];history=state['history'];best=state['best']
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        sch.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler']);restore_rng(state['rng'],gen)
        if fam=='dose':seen=state['seen'].cpu()
        del state
    def save(epoch):
        manager.save(dict(next_epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),
            scaler=scaler.state_dict(),rng=capture_rng(gen),phase=phase,best=best,history=history,seen=seen))
        put(out/'history.json',history)
    if start==0 and manager.last_path is None:save(0)
    for ep in range(start,25):
        reserve=max(180,max((x['seconds'] for x in history[-3:]),default=600 if job.get('arm')=='2a' else 180)*1.6)
        if time.time()+reserve+180>=deadline:return False
        if ep==3:phase='full';opt,sch=factory(model,phase)
        t=time.time();model.train();applied=[0]
        def counted(*args):applied[0]+=1
        hook=opt.register_step_post_hook(counted)
        try:
            if fam=='beit':
                trace=[]
                for i,(x,y,_) in enumerate(loader):
                    before_steps=applied[0]
                    rec=train_beit_step(model,opt,scaler,mix,hard,soft,ep,x.cuda(),y.cuda(),job['regime'],4)
                    rec['skipped']=int(applied[0]==before_steps)
                    trace.append(dict(epoch=ep+1,batch=i+1,**rec))
                    if rec['diverged']:
                        pd.DataFrame(trace).to_csv(out/f'steps_{ep+1:02d}.csv',index=False)
                        put(out/'complete.json',dict(**meta,outcome='diverged',epochs=ep,batch=i+1,
                            reason='Nonfinite unscaled loss or unrecoverable FP32 gradient',weights={}))
                        return True
                pd.DataFrame(trace).to_csv(out/f'steps_{ep+1:02d}.csv',index=False)
                loss=float(np.mean([r['loss'] for r in trace]))
            else:loss=a.train_one_epoch(model,ObservedBatches(loader,seen) if fam=='dose' else loader,opt,scaler,soft,hard,mix,ep)
        finally:hook.remove()
        sch.step(ep if ep<3 else ep-3)
        row=dict(epoch=ep+1,loss=float(loss),scheduled=len(loader),applied=applied[0],skipped=len(loader)-applied[0])
        if fam!='dose':
            p,y,g=a.validate(model,evals['selection'])
            if fam=='beit' and not np.isfinite(p).all():
                put(out/'complete.json',dict(**meta,outcome='diverged',epochs=ep+1,
                    reason='Nonfinite selection probabilities',weights={}))
                return True
            v=aggregate_images(p,y,g);ll=float(log_loss(v['labels'],v['probs'],labels=range(6)))
            row['selection_logloss']=ll
            if len(best)<2 or ll<best[-1]['loss']:
                best.append(dict(epoch=ep+1,loss=ll,state=cpu_copy(model.state_dict())))
                best=sorted(best,key=lambda b:b['loss'])[:2]
        row['seconds']=time.time()-t;history.append(row);save(ep+1)
        print(job['id'],ep+1,'/25',flush=True)
    if time.time()+240>=deadline:return False
    del opt,sch,scaler;gc.collect();torch.cuda.empty_cache()
    if fam=='dose':
        slot_ids=get(pkg/'dose'/str(s)/f"mask{job['n']}.json")['slots']
        np.save(out/'observed_training_slots.npy',seen.numpy())
        if not seen[slot_ids].all():raise RuntimeError('A designated inserted relative never entered a training batch.')
        dose_score(model,eval_dose,test,subset,out)
        atomic_torch_save(out/'final.pt',model.state_dict())
    else:
        snapshots=[]
        for i,v in enumerate(best,1):
            path=out/f'rank{i}.pt';atomic_torch_save(path,v['state']);snapshots.append(path)
        evaluate(model,snapshots,parts,a,b,['A','B'] if fam=='eff' else ['A'],out)
    put(out/'complete.json',dict(**meta,epochs=len(history),weights={p.name:sha(p) for p in out.glob('*.pt') if '_resume_' not in p.name}))
    for p in out.glob('fit_fold*_resume_?.*'):
        if p.suffix in ('.json','.pt'):p.unlink()
    return True

