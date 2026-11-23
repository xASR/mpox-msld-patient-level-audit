"""Train one supplementary job on one visible GPU, preserving every completed epoch."""
import argparse,copy,hashlib,math,os,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import log_loss
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
import protocol_a_train as base
import legacy_vit
from checkpoint_engine import CheckpointManager,atomic_torch_save,seed_everything,capture_rng,restore_rng,cpu_copy
from protocol_a_core import aggregate_images
from revision_io import read,put,sha,record,check_done
from revision_data import design,PixelStore,ImageRows,BudgetSampler,verify_matched_pixels

def env():
    import importlib.metadata as md
    return dict(base.environment_record(),huggingface_hub=md.version('huggingface_hub'),safetensors=md.version('safetensors'))
def tensor_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        x=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()
def standard_frames(package,paths,job):
    f=job['index'];who='a' if job['family']=='diagnostic' else 'b'
    df=pd.read_csv(package/f'design/{who}_fold{f}.csv')
    if who=='b':
        df=df.copy();df['role']=df.role.map({'train_original':'train','val_selection':'selection','val_calibration':'calibration','test':'test','excluded_known_group_overlap':'excluded'})
        if df.role.isna().any():raise ValueError('Unrecognized official assignment')
    pool=pd.read_csv(package/'design/pool.csv');images={r.source:paths[r.relative] for r in pool[pool.is_original].itertuples()}
    df['filename_patient_id']=df.patient_id;df['patient_id']=df.patient_cluster;df['filepath']=df.filename.map(images)
    if df.filepath.isna().any():raise ValueError('Missing original')
    active=df[df.role!='excluded']
    if active.groupby('patient_cluster').role.nunique().max()!=1:raise ValueError('Group appears in multiple roles')
    return df,{r:df[df.role==r].reset_index(drop=True) for r in ('train','selection','calibration','test')}
def bn_state(net):
    return {k:t.detach().cpu().clone() for k,t in net.named_buffers() if k.endswith(('running_mean','running_var','num_batches_tracked'))}
def train_logged(worker,net,loader,opt,scaler,soft,hard,mix,epoch,detailed):
    calls=[];before=bn_state(net);scale_start=scaler.get_scale()
    def observe(optimizer,args,kwargs):
        if detailed:
            gs=[p.grad.detach() for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            calls.append(torch.linalg.vector_norm(torch.stack([g.norm(2) for g in gs])).detach())
        else:calls.append(None)
    hook=opt.register_step_pre_hook(observe)
    try:loss=worker.train_one_epoch(net,loader,opt,scaler,soft,hard,mix,epoch)
    finally:hook.remove()
    after=bn_state(net)
    info=dict(batches=len(loader),amp_optimizer_steps=len(calls),amp_skipped_steps=len(loader)-len(calls),
              scaler_start=float(scale_start),scaler_end=float(scaler.get_scale()),batchnorm_buffers=len(before),
              batchnorm_changed=sum(not torch.equal(v,after[k]) for k,v in before.items()))
    for suffix in ('running_mean','running_var'):
        updates=[(after[k]-v).float().flatten() for k,v in before.items() if k.endswith(suffix)]
        if updates:info['batchnorm_'+suffix+'_rms_change']=float(torch.cat(updates).square().mean().sqrt())
    if detailed and calls:
        norms=torch.stack(calls).float();info.update(mean_unscaled_gradient_l2=float(norms.mean()),max_unscaled_gradient_l2=float(norms.max()))
    return loss,info
def predict_ablation(model,loader,vit):
    model.eval();ps=[];labs=[]
    with torch.inference_mode():
        for x,y,_ in loader:
            x=x.cuda(non_blocking=True)
            if vit:x=x.float().div_(255).sub_(.5).div_(.5)
            ps.append(model(x).softmax(1).cpu().numpy());labs.extend(y.tolist())
    return np.concatenate(ps),np.asarray(labs)

def run(package,job,root,paths,scratch,deadline,manifest):
    out=root/job['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').is_file():
        check_done(out,job,manifest,weights=True);return True
    if torch.cuda.device_count()!=1:raise RuntimeError('Each worker needs one isolated GPU.')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    family=job['family'];m=job['model'];n=job['index'];vit=m=='vit';abl=family in ('matched','release')
    epochs=100 if vit else 25;training_seed=(n if vit else 9000+n) if abl else 342+n if family=='diagnostic' else 42+n
    if family in ('alignment','diagnostic'):
        torch.use_deterministic_algorithms(True,warn_only=True)
    identity=dict(job=job,manifest=manifest,environment=env(),training_seed=training_seed)
    mgr=CheckpointManager(out,'revision',n,identity)
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:raise ValueError('Training identity or runtime changed. Restore the same input and runtime.')
    put(out/'identity.json',identity)
    saved=mgr.load_auto([out])
    if (out/'status.json').exists() and saved is None and read(out/'status.json').get('completed_epochs',0)>0:
        raise ValueError('Missing full checkpoint for a started job.')
    if not saved and time.time()>deadline-600:return False
    cfg=read(package/'design/protocol_a_manifest.json')['models']
    worker=None;sampler=None;store=None
    initial_path=out/'initial.json'
    if vit:
        legacy_vit.setup_runtime(install=False)
        if not saved:
            pretrained=legacy_vit.fetch_pretrained();init=legacy_vit.initial_state(n,pretrained)
            model=legacy_vit.new_network();model.load_state_dict(init);del init
            prehash=legacy_vit.PRETRAINED['sha256']
        else:model=legacy_vit.new_network()
        model=model.cuda();seed_everything(n)
    else:
        worker=base.original_worker(package/'design')
        modelkey='convnext_tiny' if m=='conv' else m
        model,prehash=base.network(cfg[modelkey],training_seed,not bool(saved),torch.device('cuda:0'))
        if not saved:
            ref=read(package/f'baseline/a/{modelkey}/{n if not abl else 1}/pretrained_source.json')['file_sha256']
            if prehash!=ref:raise ValueError('Pretrained backbone checksum differs from the historical run.')
        if abl:seed_everything(training_seed)
    if not saved:put(initial_path,dict(tensor_sha256=tensor_digest(model),pretrained_file_sha256=prehash,training_seed=training_seed))
    if not initial_path.is_file():raise ValueError('Original initialization record is missing.')
    generator=torch.Generator().manual_seed(n if vit else 19000+n if abl else training_seed+10000)
    if abl:
        cache,parts,budget,weight_labels=design(package,job)
        tag=family+'_'+m+('_s'+str(n) if family=='matched' else '')
        store=PixelStore(cache,paths,128 if vit else 384,scratch,tag,deadline)
        if family=='matched':verify_matched_pixels(store,parts,scratch)
        train=ImageRows(parts['train'],store,vit);bs=32 if vit else 16
        if family=='release':sampler=BudgetSampler(len(train),budget,generator)
        loader=DataLoader(train,batch_size=bs,shuffle=sampler is None,sampler=sampler,generator=generator,
                          drop_last=not vit,num_workers=0 if vit else 2,pin_memory=True)
        evals={r:DataLoader(ImageRows(df,store,vit),batch_size=128 if vit else 16,shuffle=False,num_workers=0 if vit else 2) for r,df in parts.items() if r!='train'}
        assert len(loader)==(math.ceil(budget/bs) if vit else budget//bs)
    else:
        assignment,parts=standard_frames(package,paths,job);evals=base.make_loaders(worker,parts,generator);loader=evals['train'];weight_labels=parts['train'].label
    if vit:
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0);sch=None;scaler=None;phase='full'
    else:
        phase=saved['phase'] if saved else 'head'
        factory=base.optimizer_factory(worker);opt,sch=factory(model,phase)
        cw=torch.tensor(compute_class_weight(class_weight='balanced',classes=np.arange(6),y=weight_labels),dtype=torch.float32,device='cuda')
        hard=torch.nn.CrossEntropyLoss(weight=cw,label_smoothing=.1);soft=SoftTargetCrossEntropy()
        mix=Mixup(mixup_alpha=.2,cutmix_alpha=1.,label_smoothing=.1,num_classes=6);scaler=GradScaler('cuda')
    start=0;hist=[];best=[];used=torch.zeros(len(loader.dataset),dtype=torch.bool)
    if saved:
        start=saved['next_epoch'];hist=saved['history'];best=saved['snapshots'];used=saved['used'].cpu()
        if start!=len(hist) or [x['epoch'] for x in hist]!=list(range(1,start+1)):raise ValueError('Incomplete epoch history')
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
        if sch:sch.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng'],generator);del saved
    def save(i):
        mgr.save(dict(next_epoch=i,phase=phase,model=model.state_dict(),optimizer=opt.state_dict(),
                      scheduler=sch.state_dict() if sch else {},scaler=scaler.state_dict() if scaler else {},
                      history=hist,snapshots=best,rng=capture_rng(generator),used=used))
        put(out/'status.json',dict(job=job['id'],completed_epochs=i,required_epochs=epochs,status='checkpointed'))
    if start==0 and mgr.last_path is None:save(0)
    for ep in range(start,epochs):
        recent=max((r['wall_seconds'] for r in hist[-3:]),default=90.)
        if time.time()+max(recent*1.8,100)+300>deadline:return False
        if not vit and ep==3:phase='full';opt,sch=factory(model,phase)
        t=time.time();info={}
        if vit:
            model.train();loss_sum=0.;seen=0
            for x,y,_ in loader:
                x=x.cuda(non_blocking=True).float().div_(255).sub_(.5).div_(.5);y=y.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(x),y)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();opt.step();loss_sum+=float(loss.detach())*len(y);seen+=len(y)
            loss_value=loss_sum/seen;info['optimizer_steps_applied']=len(loader)
        else:
            loss_value,info=train_logged(worker,model,loader,opt,scaler,soft,hard,mix,ep,family=='diagnostic')
            info['optimizer_steps_applied']=info['amp_optimizer_steps']
            sch.step(ep if ep<3 else ep-3)
        if not np.isfinite(loss_value):raise FloatingPointError('Nonfinite training loss')
        if sampler is not None:
            kept=sampler.last[:len(loader)*loader.batch_size] if loader.drop_last else sampler.last
            used[torch.from_numpy(kept)]=True
        if not abl:
            p,y,ids=worker.validate(model,evals['selection']);g=aggregate_images(p,y,ids)
            score=float(log_loss(g['labels'],g['probs'],labels=np.arange(6)))
            if not np.isfinite(score):raise FloatingPointError('Nonfinite selection score')
            info['selection_logloss']=score
            if len(best)<2 or score<best[-1]['selection_logloss']:
                best.append(dict(epoch=ep+1,state_dict=cpu_copy(model.state_dict()),selection_logloss=score))
                best=sorted(best,key=lambda x:x['selection_logloss'])[:2]
        hist.append(dict(epoch=ep+1,phase=phase,train_loss=float(loss_value),planned_batches=len(loader),wall_seconds=time.time()-t,
                         lr=[v['lr'] for v in opt.param_groups],**info))
        save(ep+1)
        print(job['id'],f'{ep+1}/{epochs}',flush=True)
    if time.time()+190>deadline:return False
    if abl:
        for role,el in evals.items():
            df=parts[role];p,y=predict_ablation(model,el,vit)
            if not np.array_equal(y,df['vit_label' if vit else 'label']):raise ValueError('Inference order differs.')
            if vit:
                mapping=read(package/'plan.json')['vit_to_main'];p=p[:,np.argsort(mapping)]
            arr=dict(labels=df.label.to_numpy(),probs=p,keys=df.key.to_numpy(str),source=df.source.to_numpy(str),
                     groups=df.group.to_numpy(str),is_original=df.is_original.to_numpy(bool))
            if family=='release':
                seen_train=parts['train'].iloc[np.flatnonzero(used.numpy())]
                for key,col in [('source','source'),('group','group'),('pixel','rgb')]:
                    arr['seen_'+key]=df[col].isin(seen_train[col]).to_numpy();arr['eligible_'+key]=df[col].isin(parts['train'][col]).to_numpy()
                arr['train_items_ever_used']=np.array(len(seen_train))
            np.savez_compressed(out/(role+'.npz'),**arr)
        atomic_torch_save(out/'final_weights.pt',model.state_dict())
        if family=='release':np.save(out/'training_items_seen.npy',parts['train'].item.to_numpy()[used.numpy()])
    else:
        base.export_job(worker,model,parts,evals,dict(complete=True,completed_epochs=epochs,history=hist,snapshots=best),out,
                        dict(identity,model=m,fold=n),assignment.drop(columns=['filepath']))
    put(out/'history.json',hist)
    files={p.name:record(p) for p in out.iterdir() if p.is_file() and '_resume_' not in p.name and p.name not in ('complete.json','status.json','worker.log') and p.suffix!='.tmp'}
    put(out/'complete.json',dict(job=job,manifest=manifest,status='complete',epochs=epochs,initial=read(initial_path),
        steps_per_epoch=len(loader),total_batches=sum(r['planned_batches'] for r in hist),
        optimizer_steps_applied=sum(r['optimizer_steps_applied'] for r in hist),files=files))
    check_done(out,job,manifest,weights=True)
    for slot in ['a','b']:
        for suffix in ['.pt','.json']:(out/f'revision_fold{n}_resume_{slot}{suffix}').unlink(missing_ok=True)
    put(out/'status.json',dict(status='complete',completed_epochs=epochs));return True

if __name__=='__main__':
    a=argparse.ArgumentParser()
    for key in ['package','work','paths','scratch']:a.add_argument('--'+key,type=Path,required=True)
    for key in ['id','manifest']:a.add_argument('--'+key,required=True)
    a.add_argument('--deadline',type=float,required=True);x=a.parse_args()
    j=next(j for j in read(x.package/'plan.json')['jobs'] if j['id']==x.id)
    try:finished=run(x.package,j,x.work,read(x.paths),x.scratch,x.deadline,x.manifest)
    except TimeoutError as e:print(str(e),flush=True);finished=False
    raise SystemExit(0 if finished else 75)

