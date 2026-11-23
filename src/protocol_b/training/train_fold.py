"""Train one corrected Protocol B model/fold with full epoch-boundary resume."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '42')
from pathlib import Path
import argparse
import gc
import importlib
import importlib.util
import json
import platform
import shutil
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import torch
import timm
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler
from corrected_core import (MODELS, atomic_json, atomic_array, find_inputs,
                            find_dataset, prepare_fold, sha256)
from resume_engine import CheckpointManager, atomic_torch_save, seed_everything, run_epochs


def environment_record():
    value={'python':platform.python_version(),'torch':str(torch.__version__),
           'cuda':torch.version.cuda,'cuda_device_count':torch.cuda.device_count(),
           'cudnn':torch.backends.cudnn.version() if torch.cuda.is_available() else None,
           'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           'compute_capability':list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}
    for name in ['torchvision','timm','numpy','pandas','scipy','sklearn','PIL','huggingface_hub','safetensors']:
        value[name]=str(importlib.import_module(name).__version__)
    return value


def load_worker(inputs):
    path=Path(inputs)/'provenance/original_worker_v34.py'
    if str(path.parent) not in sys.path:
        sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('original_worker_v34',path)
    worker=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=worker
    spec.loader.exec_module(worker)
    return worker


def build_network(config, fresh, seed, device):
    kwargs={'pretrained':bool(fresh),'num_classes':6}
    if config['timm_name'].startswith(('swin','beit')):
        kwargs['img_size']=384
    source_hash=None
    if fresh:
        from huggingface_hub import hf_hub_download
        path=hf_hub_download(repo_id=config['hf_repo'],filename=config['hf_file'],revision=config['hf_revision'])
        source_hash=sha256(path)
        kwargs['pretrained_cfg_overlay']={'file':path,'source':''}
    seed_everything(seed)
    return timm.create_model(config['timm_name'],**kwargs).to(device),source_hash


def optimizer_factory(worker):
    def create(model,phase):
        worker.set_backbone_requires_grad(model,phase=='full')
        parameters=model.parameters() if phase=='full' else filter(lambda p:p.requires_grad,model.parameters())
        optimizer=torch.optim.AdamW(parameters,lr=1e-4 if phase=='full' else 1e-3,weight_decay=.01)
        scheduler=CosineLRScheduler(optimizer,t_initial=22 if phase=='full' else 25,
                                     lr_min=1e-6,warmup_t=2,warmup_lr_init=1e-6)
        return optimizer,scheduler
    return create


def make_loaders(worker,frames,generator):
    train_tf,val_tf=worker.get_transforms()
    settings={'batch_size':16,'num_workers':2,'pin_memory':torch.cuda.is_available(),'persistent_workers':False}
    train=DataLoader(worker.MpoxDataset(frames['train'],train_tf),shuffle=True,drop_last=True,generator=generator,**settings)
    rest={key:DataLoader(worker.MpoxDataset(frames[key],val_tf),shuffle=False,**settings)
          for key in ['selection','calibration','test']}
    return train,rest


def verify_complete(directory,identity_sha):
    marker=Path(directory)/'fold_complete.json'
    if not marker.is_file():return False
    value=json.loads(marker.read_text())
    if value.get('schema')!='pb_known_duplicate_corrected_v1' or value.get('identity_sha256')!=identity_sha or value.get('epochs_completed')!=25:
        raise ValueError('Completion marker uses a different corrected job identity.')
    for name,record in value['prediction_files'].items():
        if not (Path(directory)/name).is_file() or sha256(Path(directory)/name)!=record['sha256']:
            return False
    for record in value['weights']:
        if not (Path(directory)/record['file']).is_file() or sha256(Path(directory)/record['file'])!=record['sha256']:
            return False
    return True


def copy_previous_outputs(manager,fold_dir):
    """Carry the chosen valid full checkpoint and small inference work into outputs."""
    if manager.last_path is None or manager.last_path.parent.resolve()==fold_dir.resolve():return
    source=manager.last_path.parent
    keep=[manager.last_path,manager.last_path.with_suffix('.json')]
    keep += [p for p in source.iterdir() if p.is_file() and p.name not in [v.name for v in keep]
             and not p.name.endswith('.tmp') and ('_resume_' not in p.name)
             and p.suffix in ['.json','.npy','.npz','.csv','.pt']]
    for p in keep:
        shutil.copyfile(p,fold_dir/p.name)
    manager.last_path=fold_dir/manager.last_path.name


def write_weights(result,fold_dir,stem):
    records=[]
    for rank,snapshot in enumerate(result['snapshots'],1):
        target=fold_dir/f'{stem}_rank{rank}.pt'
        atomic_torch_save(target,snapshot['state_dict'])
        records.append({'rank':rank,'epoch':int(snapshot['epoch']),
                        'selection_logloss':float(snapshot['selection_logloss']),
                        'file':target.name,'sha256':sha256(target),'bytes':target.stat().st_size})
    atomic_json(fold_dir/'retained_weights.json',{'epochs_completed':result['completed_epochs'],
                'training_complete':result['complete'],'weights':records})
    return records


def infer_one(worker,model,loader,fold_dir,stem,rank,split,weight_record,identity_sha):
    cache=fold_dir/f'inference_rank{rank}_{split}.npz'
    marker=cache.with_suffix('.json')
    if marker.is_file() and cache.is_file():
        info=json.loads(marker.read_text())
        if info.get('identity_sha256')==identity_sha and info.get('weight_sha256')==weight_record['sha256'] and info.get('sha256')==sha256(cache):
            with np.load(cache,allow_pickle=False) as arr:
                return {key:arr[key] for key in ['probs','labels','pids']}
    logs,labels,pids=worker.validate(model,loader,use_tta=True)
    p,y,ids=worker.get_patient_level_arrays(logs,labels,pids)
    temporary=cache.with_name(cache.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,probs=p,labels=y,pids=ids.astype(str))
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,cache)
    atomic_json(marker,{'identity_sha256':identity_sha,'weight_sha256':weight_record['sha256'],'sha256':sha256(cache)})
    return {'probs':p,'labels':y,'pids':ids.astype(str)}


def export_predictions(worker,model,frames,loaders,result,weights,fold_dir,identity,manager,deadline):
    outputs={'calibration':[],'test':[]}
    for rank,record in enumerate(weights,1):
        model.load_state_dict(torch.load(fold_dir/record['file'],map_location='cpu',weights_only=True),strict=True)
        for split in outputs:
            cache=fold_dir/f'inference_rank{rank}_{split}.npz'
            if time.monotonic()>deadline-900 and not cache.exists():
                atomic_json(fold_dir/'status.json',{'status':'trained_inference_pending','completed_epochs':25,
                            'instruction':'Save full outputs and resume; completed training and cached inference are retained.'})
                print('PAUSED: all 25 epochs are saved; resume to finish final inference.',flush=True)
                return False
            part=infer_one(worker,model,loaders[split],fold_dir,manager.stem,rank,split,record,manager.identity_sha256)
            expected=np.sort(frames[split].patient_id.unique())
            if not np.array_equal(part['pids'],expected):raise ValueError('Inference known-group order differs.')
            outputs[split].append(part)
    files={}
    for split,parts in outputs.items():
        if not np.array_equal(parts[0]['labels'],parts[1]['labels']) or not np.array_equal(parts[0]['pids'],parts[1]['pids']):
            raise ValueError('Retained checkpoint predictions are misaligned.')
        values={'probs':np.mean([p['probs'] for p in parts],axis=0),
                'labels':parts[0]['labels'],'pids':parts[0]['pids'].astype(str)}
        token='val' if split=='calibration' else 'test'
        for kind,array in values.items():
            name=f'{manager.stem}_{token}_{kind}.npy'
            atomic_array(fold_dir/name,array)
            files[name]={'sha256':sha256(fold_dir/name),'bytes':(fold_dir/name).stat().st_size}
    atomic_json(fold_dir/'epoch_history.json',result['history'])
    metadata={'schema':'pb_known_duplicate_corrected_v1','status':'complete','model':identity['model'],
              'fold':identity['fold'],'epochs_completed':25,'identity':identity,
              'identity_sha256':manager.identity_sha256,'input_manifest_sha256':identity['input_manifest_sha256'],
              'prediction_files':files,'weights':weights,'summary':frames['summary'],
              'evaluation_unit':'Known-duplicate-aware group','history_sha256':sha256(fold_dir/'epoch_history.json')}
    atomic_json(fold_dir/'fold_complete.json',metadata)
    atomic_json(fold_dir/'status.json',{'status':'complete','completed_epochs':25,'model':identity['model'],'fold':identity['fold']})
    print(f"COMPLETE: {identity['model']} fold {identity['fold']}; 25 epochs, two weights and final predictions saved.",flush=True)
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--fold',type=int,choices=range(1,6),required=True)
    p.add_argument('--input-root',type=Path,default=Path('/kaggle/input'))
    p.add_argument('--dataset-root',type=Path)
    p.add_argument('--output-root',type=Path,default=Path('/kaggle/working/pb_corrected'))
    p.add_argument('--expected-manifest-sha',required=True)
    p.add_argument('--max-hours',type=float,default=4.25)
    p.add_argument('--require-resume',action='store_true')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args(argv)
    started=time.monotonic()
    if not 0<args.max_hours<=8:raise ValueError('Choose a time budget above zero and at most eight hours.')
    deadline=started+args.max_hours*3600
    inputs=find_inputs(args.input_root,args.expected_manifest_sha)
    manifest=json.loads((inputs/'pb_corrected_manifest.json').read_text())
    dataset=args.dataset_root or find_dataset(args.input_root,manifest['datalog_sha256'])
    frames=prepare_fold(inputs,dataset,args.fold,args.expected_manifest_sha)
    fold_dir=args.output_root/args.model/f'fold{args.fold}'
    fold_dir.mkdir(parents=True,exist_ok=True)
    atomic_json(fold_dir/'preflight.json',frames['summary'])
    print(json.dumps(frames['summary'],indent=2),flush=True)
    print('PREFLIGHT PASS: corrected known groups are disjoint across all four roles; augmentation sources and image hashes match.',flush=True)
    if args.check_only:return 0
    if not torch.cuda.is_available():raise RuntimeError('Select GPU T4 x2 for training; use no GPU for CPU preflight/merge.')
    if torch.cuda.device_count()!=1:raise RuntimeError('Use run_jobs.py to assign one visible GPU per training process.')
    if timm.__version__!='1.0.29':raise RuntimeError('Run the setup cell to pin timm 1.0.29.')
    worker=load_worker(inputs)
    worker.CFG.DEVICE=torch.device('cuda:0')
    worker.set_full_determinism()
    torch.use_deterministic_algorithms(True)
    root=Path(__file__).parent
    identity={'protocol':'pb_known_duplicate_corrected_v1','model':args.model,'fold':args.fold,
              'input_manifest_sha256':args.expected_manifest_sha,'data_signature':frames['summary']['data_signature'],
              'training_seed':42+100*MODELS.index(args.model)+args.fold,
              'training':manifest['training'],'pretrained_source':manifest['models'][args.model],
              'environment':environment_record(),
              'code':{name:sha256(root/name) for name in ['corrected_core.py','resume_engine.py','train_fold.py']},
              'original_worker_sha256':sha256(inputs/'provenance/original_worker_v34.py')}
    manager=CheckpointManager(fold_dir,args.model,args.fold,identity)
    state=manager.load_auto([fold_dir,args.input_root])
    if state is None and args.require_resume:
        raise FileNotFoundError('Resume was requested but no valid full checkpoint was attached. Attach the previous saved NOTEBOOK OUTPUT, including .pt and .json files. The small results ZIP cannot resume training.')
    copy_previous_outputs(manager,fold_dir)
    atomic_json(fold_dir/'job_identity.json',identity)
    if verify_complete(fold_dir,manager.identity_sha256):
        print('Already complete: verified saved results; no training repeated.',flush=True)
        return 0
    model,pretrained_sha=build_network(manifest['models'][args.model],state is None,identity['training_seed'],worker.CFG.DEVICE)
    if pretrained_sha:atomic_json(fold_dir/'pretrained_source.json',dict(manifest['models'][args.model],file_sha256=pretrained_sha))
    generator=worker.get_generator()
    train_loader,loaders=make_loaders(worker,frames,generator)
    scaler=GradScaler('cuda',enabled=True,growth_interval=1000)
    mixup=worker.Mixup(mixup_alpha=.1,cutmix_alpha=0.,label_smoothing=.1,num_classes=6)
    soft=worker.SoftTargetCrossEntropy()
    hard=torch.nn.CrossEntropyLoss(weight=torch.tensor(frames['class_weights']).to(worker.CFG.DEVICE),label_smoothing=.1)
    def train_epoch(net,optimizer,scale,epoch):
        net.train();total=0.;count=0
        for x,y,_ in train_loader:
            x,y=x.to(worker.CFG.DEVICE),y.to(worker.CFG.DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda'):
                if epoch<10:
                    xx,yy=mixup(x,y);loss=soft(net(xx),yy)
                else:loss=hard(net(x),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss; last complete epoch remains available.')
            scale.scale(loss).backward();scale.step(optimizer);scale.update()
            total+=float(loss.detach())*len(y);count+=len(y)
        worker.cuda_sync()
        return total/count
    def selection_loss(net):
        logs,labels,pids=worker.validate(net,loaders['selection'],use_tta=False)
        prob,y,ids=worker.get_patient_level_arrays(logs,labels,pids)
        return worker.evaluate_patient_level(prob,y,verbose=False)[-1]
    result=run_epochs(model,optimizer_factory(worker),scaler,generator,train_epoch,selection_loss,manager,
                      resume_state=state,session_seconds=max(0,deadline-time.monotonic()),
                      status_path=fold_dir/'status.json')
    del state
    weights=write_weights(result,fold_dir,manager.stem)
    atomic_json(fold_dir/'epoch_history.json',result['history'])
    frames['assignment'].drop(columns='filepath').to_csv(fold_dir/'corrected_assignment.csv',index=False)
    frames['augmented'].drop(columns='filepath').to_csv(fold_dir/'augmented_used.csv',index=False)
    if not result['complete']:
        print('PAUSED SAFELY: save full notebook outputs; resume from the last completed epoch.',flush=True)
        return 0
    export_predictions(worker,model,frames,loaders,result,weights,fold_dir,identity,manager,deadline)
    gc.collect()
    return 0


if __name__=='__main__':raise SystemExit(main())
