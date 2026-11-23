"""Ten additional fixed-dose conditions; exact epoch-boundary training resume."""
from pathlib import Path
import argparse,gc,json,time,subprocess,sys
import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from supplemental_common import sha,digest,write_json,verify_inputs,bundle
from checkpoint_engine import CheckpointManager,run_epochs,atomic_torch_save,seed_everything
from protocol_a_core import find_images
import protocol_a_train as training
from supplemental_train import FixedAncestryDataset,tensor_state_sha
from extension_common import SUPPLEMENTAL_SHA,verify_extension,dose_frames,baseline,check_files,seal,copy_prior

JOBS=[{'seed':s,'dose':d} for s in range(5) for d in (2,4)]
def name(job):return f'seed_{job["seed"]}_dose_{job["dose"]}'

def identity(inputs,ext,job):
    mc,_=baseline(ext,job['seed'],'clean');me,_=baseline(ext,job['seed'],'contaminated')
    env=training.environment_record()
    if env!=mc['identity']['environment'] or env!=me['identity']['environment']:
        raise ValueError('Environment differs from completed matched baseline. Use T4 x2 and the original Kaggle image; do not silently pair different environments.')
    return {'schema':'dose384_extension_v1','job':job,'training_seed':9000+job['seed'],
            'supplemental_manifest_sha256':SUPPLEMENTAL_SHA,'extension_manifest_sha256':sha(Path(ext)/'extension_manifest.json'),
            'script_sha256':sha(__file__),'environment':env,'recipe':mc['identity']['recipe'],
            'initial_state':mc['initial_state'],'training_device_index':0 if job['dose']==2 else 1}

def inspect(folder,ident,weights=True):
    folder=Path(folder);meta=json.loads((folder/'dose_complete.json').read_text())
    if meta['status']!='complete' or meta['epochs']!=25 or meta['identity']!=ident or meta['identity_sha256']!=digest(ident):raise ValueError('Completed dose job differs.')
    check_files(folder,meta,weights)
    if meta['initial_state']!=ident['initial_state']:raise ValueError('Initial weights differ from baseline.')
    history=json.loads((folder/'history.json').read_text())
    if [h['epoch'] for h in history]!=list(range(1,26)) or any(h['phase']!=('head' if h['epoch']<=3 else 'full') or not np.isfinite(h['mean_training_loss']) for h in history):raise ValueError('Incomplete/nonfinite history.')
    return meta

def train(inputs,ext,job,images,out,left,max_epochs=None):
    clean,frame,test=dose_frames(inputs,ext,job['seed'],job['dose']);ident=identity(inputs,ext,job)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manager=CheckpointManager(out,f'dose{job["dose"]}',job['seed'],ident)
    resume=manager.load_auto([out])
    cfg=json.loads((Path(inputs)/'a_design/seed_42/protocol_a_manifest.json').read_text())['models']['convnext_tiny']
    device=torch.device(f'cuda:{ident["training_device_index"]}')
    torch.cuda.set_device(device)
    net,source_sha=training.network(cfg,9000+job['seed'],resume is None,device)
    if resume is None:
        initial={'parameter_sha256':tensor_state_sha(net.state_dict()),'training_seed':9000+job['seed'],'pretrained_file_sha256':source_sha}
        if initial!=ident['initial_state']:raise ValueError('Fresh initial weights differ from the existing matched clean/eight-relative runs. No training was started.')
        write_json(out/'initial_state.json',initial)
    elif json.loads((out/'initial_state.json').read_text())!=ident['initial_state']:raise ValueError('Resume lost initial-state provenance.')
    generator=torch.Generator().manual_seed(19000+job['seed'])
    loader=DataLoader(FixedAncestryDataset(frame,images),batch_size=16,shuffle=True,num_workers=2,drop_last=True,generator=generator,pin_memory=True)
    test_loader=DataLoader(FixedAncestryDataset(test,images),batch_size=16,shuffle=False,num_workers=2)
    weights=torch.tensor(compute_class_weight(class_weight='balanced',classes=np.arange(6),y=clean.label),dtype=torch.float32,device=device)
    soft=SoftTargetCrossEntropy();hard=torch.nn.CrossEntropyLoss(weight=weights,label_smoothing=.1)
    mix=Mixup(mixup_alpha=.2,cutmix_alpha=1.,label_smoothing=.1,num_classes=6)
    worker=training.original_worker(Path(inputs)/'a_design/seed_42');worker.CFG.DEVICE=device
    seed_everything(9000+job['seed'])
    scaler=GradScaler('cuda',enabled=True)
    def epoch(model,opt,scaler,ep):return worker.train_one_epoch(model,loader,opt,scaler,soft,hard,mix,ep)
    result=run_epochs(net,training.optimizer_factory(worker),scaler,generator,epoch,lambda n:0.,manager,
        resume_state=resume,session_seconds=max(0,left-600),minimum_epoch_budget=300,
        max_epochs_this_session=max_epochs,status_path=out/'job_status.json')
    if not result['complete']:
        del net,resume,result,loader,test_loader,weights,mix,scaler;gc.collect();torch.cuda.empty_cache();return False
    net.eval();parts=[]
    with torch.no_grad():
        for x,_,_ in test_loader:parts.append(net(x.cuda()).softmax(1).cpu().numpy())
    np.savez_compressed(out/'test_predictions.npz',filenames=test.filename.to_numpy(dtype=str),groups=test.patient_cluster.to_numpy(dtype=str),labels=test.label.to_numpy(),probs=np.concatenate(parts))
    atomic_torch_save(out/'final_weights.pt',net.state_dict());write_json(out/'history.json',result['history'])
    files=seal(out,['test_predictions.npz','final_weights.pt','history.json','initial_state.json'])
    write_json(out/'dose_complete.json',{'status':'complete','schema':'dose384_extension_v1','job':job,'epochs':25,
        'identity':ident,'identity_sha256':digest(ident),'initial_state':ident['initial_state'],
        'training_entries':len(frame),'steps_per_epoch':len(frame)//16,'exposed_augmented_slots':job['dose']*len(test),
        'test_originals_used_for_training_or_selection':False,'test_relatives_intentionally_exposed':True,'files':files})
    inspect(out,ident)
    for p in out.glob(f'dose{job["dose"]}_fold{job["seed"]}_resume_?.*'):
        if p.suffix in ('.pt','.json'):p.unlink()
    del net,resume,result,loader,test_loader,weights,mix,scaler;gc.collect();torch.cuda.empty_cache();return True

def run_worker(args):
    out=Path(args.work)/'SUPP_Dose_Response'
    images={k:Path(v) for k,v in json.loads((out/'verified_image_paths.json').read_text()).items()}
    torch.cuda.set_device(0 if args.worker_dose==2 else 1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    end=args.session_started+args.session_hours*3600
    progress={'status':'paused','dose':args.worker_dose,'completed_jobs':0,'active_job':None}
    try:
        for job in [j for j in JOBS if j['dose']==args.worker_dose]:
            folder=out/name(job);ident=identity(args.inputs,args.extension,job)
            if (folder/'dose_complete.json').exists():inspect(folder,ident);progress['completed_jobs']+=1;continue
            if end-time.monotonic()<1800:break
            progress['active_job']=job;write_json(out/f'worker_dose{args.worker_dose}_progress.json',progress)
            print('AUTO DOSE JOB',name(job),'GPU',ident['training_device_index'],flush=True)
            if not train(args.inputs,args.extension,job,images,folder,end-time.monotonic()):break
            progress['completed_jobs']+=1;progress['active_job']=None
        if progress['completed_jobs']==5:progress['status']='complete';progress['active_job']=None
    finally:write_json(out/f'worker_dose{args.worker_dose}_progress.json',progress)

def main(args):
    if args.worker_dose is not None:return run_worker(args)
    verify_inputs(args.inputs,SUPPLEMENTAL_SHA);verify_extension(args.extension,args.expected_extension_sha)
    out=Path(args.work)/'SUPP_Dose_Response';out.mkdir(parents=True,exist_ok=True)
    if not args.resume and any(Path(args.input_root).rglob('dose_queue_plan.json')):
        raise ValueError('A previous dose output is attached. Set RESUME=True to verify/continue it instead of silently starting another fresh queue.')
    if args.resume and not copy_prior(args.input_root,out,'dose_queue_plan.json'):
        raise FileNotFoundError('RESUME=True requires the previous FULL saved SUPP_Dose_Response output, including .pt/.json states.')
    if not args.resume and any(out.rglob('*resume_?.pt')):raise ValueError('Set RESUME=True for existing interrupted training.')
    queue={'schema':'dose_queue_v1','jobs':JOBS,'extension_manifest_sha256':args.expected_extension_sha}
    marker=out/'dose_queue_plan.json'
    if marker.exists() and json.loads(marker.read_text())!=queue:raise ValueError('Different queue attached.')
    write_json(marker,queue)
    clinical=pd.read_csv(Path(args.inputs)/'a_design/seed_42/clinical_images.csv')
    images=find_images(args.input_root,clinical,True)
    for job in JOBS:dose_frames(args.inputs,args.extension,job['seed'],job['dose']);identity(args.inputs,args.extension,job)
    if torch.cuda.device_count()!=2:raise RuntimeError('Select T4 x2: both GPUs are required for the two independent dose workers.')
    if torch.cuda.get_device_name(1)!=torch.cuda.get_device_name(0):raise RuntimeError('Both matched workers require the same GPU type.')
    write_json(out/'verified_image_paths.json',{k:str(v) for k,v in images.items()})
    print('PREFLIGHT PASS: all 755 originals, 10 new dose conditions, frozen cohorts and baseline environments verified. Two independent GPU workers.',flush=True)
    if args.check_only:return
    if not torch.cuda.is_available():raise RuntimeError('Select T4 x2.')
    if args.session_started is None:args.session_started=time.monotonic()
    status={'status':'paused','completed_jobs':0,'required_jobs':10,'active_job':None,'extension_manifest_sha256':args.expected_extension_sha}
    processes=[];streams=[]
    try:
        for dose in (2,4):
            log=(out/f'worker_dose{dose}.log').open('w',encoding='utf-8');streams.append(log)
            command=[sys.executable,str(Path(__file__).resolve()),'--inputs',args.inputs,'--extension',args.extension,'--expected-extension-sha',args.expected_extension_sha,
                '--input-root',args.input_root,'--work',args.work,'--session-hours',str(args.session_hours),'--session-started',str(args.session_started),'--worker-dose',str(dose)]
            processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
        previous=-1
        while True:
            count=sum((out/name(job)/'dose_complete.json').is_file() for job in JOBS)
            status['completed_jobs']=count;write_json(out/'dose_progress.json',status)
            if count!=previous:print('DOSE CONDITIONS COMPLETE:',count,'/10. Worker details are in worker_dose2.log and worker_dose4.log.',flush=True);previous=count
            if all(p.poll() is not None for p in processes):break
            if any(p.poll() not in (None,0) for p in processes):
                for p in processes:
                    if p.poll() is None:p.terminate()
                raise RuntimeError('A dose worker failed. Retain full output and inspect worker logs; do not restart with changed settings.')
            time.sleep(5)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('Dose worker error; see worker logs.')
        completed=0
        for job in JOBS:
            folder=out/name(job)
            if (folder/'dose_complete.json').exists():inspect(folder,identity(args.inputs,args.extension,job));completed+=1
        status['completed_jobs']=completed
        status['workers']=[json.loads((out/f'worker_dose{d}_progress.json').read_text()) for d in (2,4)]
        if completed==10:status['status']='complete'
    finally:
        for p in processes:
            if p.poll() is None:p.terminate();p.wait()
        for s in streams:s.close()
        write_json(out/'dose_progress.json',status);bundle(out)
    print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--extension',required=True);p.add_argument('--expected-extension-sha',required=True)
    p.add_argument('--input-root',default='/kaggle/input');p.add_argument('--work',default='/kaggle/working');p.add_argument('--session-hours',type=float,default=10.5)
    p.add_argument('--session-started',type=float);p.add_argument('--resume',action='store_true');p.add_argument('--check-only',action='store_true');p.add_argument('--worker-dose',type=int,choices=(2,4));main(p.parse_args())
