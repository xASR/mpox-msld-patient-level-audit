"""Finite job lists, epoch-boundary resume, and immutable supplemental results."""
from pathlib import Path
import argparse
import gc as python_gc
import json
import shutil
import time
import tempfile
import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from sklearn.utils.class_weight import compute_class_weight
from supplemental_common import (sha, digest, write_json, verify_inputs, bundle, jobs_for, job_name)
from checkpoint_engine import CheckpointManager, run_epochs, atomic_torch_save, seed_everything
from protocol_a_core import verify_design, find_images, inspect_job
import protocol_a_train as training


def scoped_resume(search, destination, model, fold, identity, required=False):
    """Do not mix different repeated seeds that share a model/fold filename."""
    expected = digest(identity); destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    found = False
    for mark in Path(search).rglob(f'{model}_fold{fold}_resume_?.json'):
        rec = json.loads(mark.read_text())
        if rec.get('identity_sha256') != expected: continue
        src = mark.with_suffix('.pt')
        if not src.is_file() or src.stat().st_size != rec['bytes'] or sha(src) != rec['sha256']: continue
        # Keep every distinct slot/epoch candidate; CheckpointManager detects conflicts.
        target = destination / (sha(src)[:16] + '_' + str(rec['completed_epochs']))
        target.mkdir(exist_ok=True)
        shutil.copyfile(src, target / src.name); shutil.copyfile(mark, target / mark.name)
        found = True
    if required and not found and not any(destination.rglob('*.pt')):
        raise FileNotFoundError('REQUIRE_RESUME is on, but no matching full state was attached. Attach the interrupted supplemental saved output, including .pt/.json; RESULTS_ONLY.zip cannot resume training. Use the same GPU/environment.')
    return destination


class FixedAncestryDataset(Dataset):
    """A deterministic, offline 384px augmentation pool; no test originals inserted."""
    def __init__(self, frame, images):
        self.frame = frame.reset_index(drop=True); self.images = images
        self.transform = T.Compose([T.Resize((384,384)), T.ToTensor(),
                                    T.Normalize([.485,.456,.406], [.229,.224,.225])])
        self.augment = T.Compose([T.RandomRotation(30),
            T.RandomAffine(degrees=0, translate=(.1,.1), scale=(.8,1.2), shear=15),
            T.ColorJitter(brightness=.15, contrast=.2, saturation=.15, hue=.1), T.Resize((384,384))])

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        r = self.frame.iloc[index]
        image = Image.open(self.images[r.filename]).convert('RGB')
        if int(r.variant) > 0:
            # torchvision PIL transforms draw from torch; avoid changing the loader RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(r.augmentation_seed)); image = self.augment(image)
        return self.transform(image), int(r.label), r.patient_cluster


def ablation_frames(inputs, seed):
    root = Path(inputs) / 'ablation' / f'seed_{seed}'
    clean = pd.read_csv(root/'clean.csv'); contaminated = pd.read_csv(root/'contaminated.csv')
    test = pd.read_csv(root/'test.csv')
    if len(clean) != len(contaminated) or not np.array_equal(clean.label, contaminated.label):
        raise ValueError('Ablation conditions must match training N and ordered class labels.')
    if list(clean.slot) != list(range(len(clean))) or not np.array_equal(clean.slot, contaminated.slot): raise ValueError('Training slots changed.')
    if set(clean.patient_cluster) & set(test.patient_cluster): raise ValueError('Clean training contains a reserved test group.')
    inserted = contaminated.intentional_contaminant.astype(str).str.lower().eq('true')
    if not inserted.any() or not set(contaminated.loc[inserted, 'filename']) <= set(test.filename): raise ValueError('Contamination intervention differs.')
    if (contaminated.loc[inserted, 'variant'] == 0).any(): raise ValueError('Never insert an unaugmented test original.')
    unchanged = ~inserted
    pd.testing.assert_frame_equal(clean.loc[unchanged].reset_index(drop=True), contaminated.loc[unchanged].reset_index(drop=True))
    if clean.patient_cluster.nunique() != 416 or test.patient_cluster.nunique() != 104: raise ValueError('Expected the same 416/104 premerged groups as the original Table 5.')
    return clean, contaminated, test


def ablation_identity(inputs, job, environment):
    root = Path(inputs)
    # Both conditions have one shared training seed and initial state.
    return {'kind': 'ablation384', 'seed': job['seed'], 'condition': job['condition'],
        'manifest_sha256': sha(root/'supplemental_manifest.json'), 'training_seed': 9000+job['seed'],
        'environment': environment,
        'code': {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'recipe': json.loads((root/'analysis_plan.json').read_text())['ablation384']}


def run_ablation(inputs, job, images, out, search, seconds_left, require_resume):
    clean, contaminated, test = ablation_frames(inputs, job['seed'])
    frame = clean if job['condition'] == 'clean' else contaminated
    identity = ablation_identity(inputs, job, training.environment_record())
    out.mkdir(parents=True, exist_ok=True)
    scoped = scoped_resume(search, Path(tempfile.gettempdir())/'supplemental_resume'/digest(identity), 'ablation384', job['seed'], identity, require_resume)
    manager = CheckpointManager(out, 'ablation384', job['seed'], identity)
    resume = manager.load_auto([out, scoped])
    cfg = json.loads((Path(inputs)/'a_design/seed_42/protocol_a_manifest.json').read_text())['models']['convnext_tiny']
    model, source_sha = training.network(cfg, 9000+job['seed'], resume is None, torch.device('cuda:0'))
    if resume is None:
        write_json(out/'initial_state.json', {'parameter_sha256': tensor_state_sha(model.state_dict()),
            'training_seed': 9000+job['seed'], 'pretrained_file_sha256': source_sha})
    else:
        # Its initial-state identity must also survive a pause.
        originals = [p.parent/'initial_state.json' for p in Path(search).rglob('ablation_complete.json')]
        originals += list(Path(search).rglob('initial_state.json'))
        for p in originals:
            try:
                rec=json.loads(p.read_text())
                if rec.get('training_seed') == 9000+job['seed']:
                    write_json(out/'initial_state.json', rec); break
            except (ValueError, OSError): continue
        if not (out/'initial_state.json').exists(): raise ValueError('Full interrupted output must include initial_state.json.')
    generator = torch.Generator().manual_seed(19000+job['seed'])
    loader = DataLoader(FixedAncestryDataset(frame, images), batch_size=16, shuffle=True,
                        num_workers=2, drop_last=True, generator=generator, pin_memory=True)
    test_loader = DataLoader(FixedAncestryDataset(test, images), batch_size=16, shuffle=False, num_workers=2)
    weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.arange(6), y=clean.label), dtype=torch.float32, device='cuda:0')
    soft_loss = SoftTargetCrossEntropy(); hard_loss = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=.1)
    mixup = Mixup(mixup_alpha=.2, cutmix_alpha=1., label_smoothing=.1, num_classes=6)
    worker = training.original_worker(Path(inputs)/'a_design/seed_42')
    # The imported legacy worker seeds globals at import. Restore the declared
    # paired-job seed before any stochastic training; resume restores its own RNG.
    seed_everything(9000+job['seed'])
    scaler = GradScaler('cuda', enabled=True)
    def epoch(net, optimizer, scaler, ep):
        return worker.train_one_epoch(net, loader, optimizer, scaler, soft_loss, hard_loss, mixup, ep)
    result = run_epochs(model, training.optimizer_factory(worker), scaler, generator, epoch,
        lambda net: 0., manager, resume_state=resume, session_seconds=max(0,seconds_left-600),
        status_path=out/'job_status.json', minimum_epoch_budget=300)
    # The generic engine has an unused selection-loss field of zero. Neither test
    # images nor test loss are evaluated before the final 25th epoch.
    if not result['complete']: return False
    model.eval(); probs=[]
    with torch.no_grad():
        for x, _, _ in test_loader: probs.append(model(x.cuda()).softmax(1).cpu().numpy())
    p = np.concatenate(probs)
    np.savez_compressed(out/'test_predictions.npz', filenames=test.filename.to_numpy(dtype=str),
        groups=test.patient_cluster.to_numpy(dtype=str), labels=test.label.to_numpy(), probs=p)
    atomic_torch_save(out/'final_weights.pt', model.state_dict())
    write_json(out/'history.json', result['history'])
    write_json(out/'ablation_complete.json', {'schema':'supplemental_ablation384_v1','status':'complete',
        'seed':job['seed'],'condition':job['condition'],'identity':identity,'identity_sha256':digest(identity),
        'epochs':25, 'initial_state':json.loads((out/'initial_state.json').read_text()),
        'training_entries':len(frame), 'steps_per_epoch':len(loader),
        'test_originals_used_for_training_or_selection':False,
        'test_relatives_intentionally_exposed':job['condition']=='contaminated',
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.iterdir()
                 if p.is_file() and p.name in ('test_predictions.npz','final_weights.pt','history.json','initial_state.json')}})
    # Remove only named rolling checkpoints after verified final export.
    for p in out.glob(f'ablation384_fold{job["seed"]}_resume_*'):
        if p.suffix in ('.pt','.json'): p.unlink()
    del model, resume, result, loader, test_loader, weights, mixup, scaler
    python_gc.collect(); torch.cuda.empty_cache()
    return True


def tensor_state_sha(state):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        x=value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


def run(args):
    inputs=Path(args.inputs); verify_inputs(inputs,args.expected_sha)
    jobs=jobs_for(args.account)
    if not 0 <= args.start < len(jobs) or args.max_jobs < 1: raise ValueError('Invalid job range.')
    chosen=jobs[args.start:args.start+args.max_jobs]
    out=Path(args.work)/f'SUPP_{args.account}_{args.start:03d}'; out.mkdir(parents=True,exist_ok=True)
    clinical=pd.read_csv(inputs/'a_design/seed_42/clinical_images.csv')
    paths=find_images(args.input_root,clinical,True)
    for j in chosen:
        if j['kind']=='repeat':
            design=inputs/f'a_design/seed_{j["seed"]}'
            verify_design(design,sha(design/'protocol_a_manifest.json'))
        else: ablation_frames(inputs,j['seed'])
    write_json(out/'requested_jobs.json',{'plan_sha256':sha(inputs/'analysis_plan.json'), 'account':args.account,
        'start':args.start, 'max_jobs':args.max_jobs, 'jobs':chosen})
    print('PREFLIGHT PASS. Jobs:',[job_name(j) for j in chosen],flush=True)
    if args.check_only: return
    if not torch.cuda.is_available(): raise RuntimeError('Select T4 x2, Internet ON for training.')
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.monotonic(); status=[]; next_job=args.start
    for i,job in enumerate(chosen,args.start):
        folder=out/job_name(job); folder.mkdir(exist_ok=True)
        left=args.session_hours*3600-(time.monotonic()-started)
        if left < 1200: break
        if job['kind']=='repeat':
            design=inputs/f'a_design/seed_{job["seed"]}'
            manifest=json.loads((design/'protocol_a_manifest.json').read_text())
            identity=training.identity_for(design,manifest,job['model'],job['fold'])
            scoped=scoped_resume(args.input_root,Path(tempfile.gettempdir())/'supplemental_resume'/digest(identity),job['model'],job['fold'],identity,
                                 args.require_resume and i==args.start)
            result=training.train_job(design,manifest,paths,job['model'],job['fold'],folder,scoped,left)
            complete=result['complete']
        else:
            complete=run_ablation(inputs,job,paths,folder,args.input_root,left,args.require_resume and i==args.start)
        status.append({'index':i, 'job':job,'complete':complete})
        next_job=i+1 if complete else i
        write_json(out/'batch_status.json',{'requested_complete':len(status)==len(chosen) and all(r['complete'] for r in status),
            'next_start_job':next_job, 'account_finished':next_job==len(jobs), 'jobs':status})
        bundle(out)
        if not complete: break
    if not status:
        write_json(out/'batch_status.json',{'requested_complete':False,'next_start_job':args.start,'account_finished':False,'jobs':[]})
    print('NEXT START_JOB =',next_job,'of',len(jobs),flush=True)
    print('Keep FULL saved outputs for selected weights and any paused optimizer states. Download RESULTS_ONLY.zip for final CPU merge.',flush=True)
    bundle(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--expected-sha',required=True)
    p.add_argument('--input-root',type=Path,default=Path('/kaggle/input'));p.add_argument('--work',default='/kaggle/working')
    p.add_argument('--account',choices=['alve','friend1','friend2'],required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--max-jobs',type=int,default=10)
    p.add_argument('--session-hours',type=float,default=10.5);p.add_argument('--require-resume',action='store_true');p.add_argument('--check-only',action='store_true')
    run(p.parse_args())
