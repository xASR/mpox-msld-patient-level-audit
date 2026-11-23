"""One resumable, independently evaluated model/fold of revised Protocol A."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import gc
import importlib
import importlib.util
import json
import platform
import time
import numpy as np
import pandas as pd
import torch
import timm
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_class_weight
from completion_core import atomic_json, sha256
from protocol_a_core import aggregate_images, save_group, inspect_job, digest
from checkpoint_engine import CheckpointManager, atomic_torch_save, seed_everything, run_epochs


def environment_record():
    result = {'python': platform.python_version(), 'torch': str(torch.__version__),
              'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
              'cuda_device_count': torch.cuda.device_count(),
              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for name in ('torchvision', 'timm', 'numpy', 'pandas', 'scipy', 'sklearn', 'PIL'):
        result[name] = str(importlib.import_module(name).__version__)
    return result


def identity_for(inputs, manifest, model, fold):
    names = ('protocol_a_train.py', 'protocol_a_core.py', 'checkpoint_engine.py', 'completion_core.py')
    model_index = list(manifest['models']).index(model)
    return {'model': model, 'fold': fold,
            'input_manifest_sha256': sha256(Path(inputs) / 'protocol_a_manifest.json'),
            'split_sha256': sha256(Path(inputs) / f'splits/fold{fold}.csv'),
            'training': manifest['training'], 'pretrained_source': manifest['models'][model],
            'seed': 42 + 100 * model_index + fold, 'environment': environment_record(),
            'training_code': {name: sha256(Path(__file__).parent / name) for name in names}}


def original_worker(inputs):
    import sys
    path = Path(inputs) / 'original_protocol_a_worker.py'
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('original_protocol_a_worker', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def network(config, seed, fresh, device):
    arguments = {'num_classes': 6, 'pretrained': fresh}
    if config['timm_name'].startswith(('swin', 'beit')):
        arguments['img_size'] = 384
    source_sha = None
    if fresh:
        from huggingface_hub import hf_hub_download
        source = hf_hub_download(config['hf_repo'], filename=config['hf_file'], revision=config['hf_revision'])
        source_sha = sha256(source)
        arguments['pretrained_cfg_overlay'] = {'file': source, 'source': ''}
    seed_everything(seed)
    return timm.create_model(config['timm_name'], **arguments).to(device), source_sha


def make_frames(inputs, fold, paths):
    assignment = pd.read_csv(Path(inputs) / f'splits/fold{fold}.csv')
    frame = assignment.copy()
    frame['filename_patient_id'] = frame.patient_id
    frame['patient_id'] = frame.patient_cluster
    frame['filepath'] = [str(paths[name]) for name in frame.filename]
    return assignment, {role: frame.loc[frame.role == role].reset_index(drop=True)
                        for role in ('train', 'selection', 'calibration', 'test')}


def make_loaders(worker, frames, generator, workers=2):
    train_tf, test_tf = worker.get_transforms()
    common = {'batch_size': 16, 'num_workers': workers, 'pin_memory': torch.cuda.is_available(), 'persistent_workers': False}
    return {role: DataLoader(worker.MpoxDataset(frame, train_tf if role == 'train' else test_tf),
                              shuffle=role == 'train', drop_last=role == 'train',
                              generator=generator if role == 'train' else None, **common)
            for role, frame in frames.items()}


def optimizer_factory(worker):
    def create(model, phase):
        worker.set_backbone_requires_grad(model, phase == 'full')
        parameters = model.parameters() if phase == 'full' else filter(lambda p: p.requires_grad, model.parameters())
        opt = torch.optim.AdamW(parameters, lr=1e-4 if phase == 'full' else 1e-3, weight_decay=1e-2)
        scheduler = CosineLRScheduler(opt, t_initial=22 if phase == 'full' else 25,
                                      lr_min=1e-6, warmup_t=2, warmup_lr_init=1e-6)
        return opt, scheduler
    return create


def export_job(worker, model, frames, loaders, result, folder, identity, assignment):
    if not result['complete'] or result['completed_epochs'] != 25 or len(result['history']) != 25:
        raise ValueError('All 25 epochs are required before exporting final predictions.')
    if len(result['snapshots']) != 2:
        raise ValueError('Two selection-half checkpoints are required.')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{identity['model']}_fold{identity['fold']}"
    image_predictions = {'calibration': [], 'test': []}
    retained, artifacts = [], []
    for rank, snapshot in enumerate(result['snapshots'], 1):
        path = folder / f'{stem}_rank{rank}.pt'
        atomic_torch_save(path, snapshot['state_dict'])
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        retained.append({'rank': rank, 'epoch': int(snapshot['epoch']),
                         'selection_logloss': float(snapshot['selection_logloss']), 'file': path.name})
        artifacts.append(path)
        for role in image_predictions:
            p, y, ids = worker.validate(model, loaders[role])
            if not np.array_equal(y, frames[role].label.to_numpy()) or not np.array_equal(ids.astype(str), frames[role].patient_cluster.to_numpy().astype(str)):
                raise ValueError('Inference rows differ from the fixed image manifest.')
            image_predictions[role].append(p)
    files = {}
    for role, split in (('calibration', 'val'), ('test', 'test')):
        frame = frames[role]
        # Equal snapshot mean at image level, then the original A confidence-weighted group mean.
        image_probs = np.mean(image_predictions[role], axis=0)
        group = aggregate_images(image_probs, frame.label.values, frame.patient_cluster.values)
        files.update(save_group(folder, f'{stem}_{split}', group))
        path = folder / f'{stem}_{split}_images.npz'
        np.savez_compressed(path, filenames=frame.filename.to_numpy().astype(str),
                            patient_ids=frame.filename_patient_id.to_numpy().astype(str),
                            patient_clusters=frame.patient_cluster.to_numpy().astype(str),
                            labels=frame.label.to_numpy(), probs=image_probs)
        artifacts.append(path)
    atomic_json(folder / 'epoch_history.json', result['history'])
    assignment.to_csv(folder / 'assignment.csv', index=False)
    artifacts += [folder / 'epoch_history.json', folder / 'assignment.csv']
    for path in artifacts:
        files[path.name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    meta = {'format_version': 1, 'status': 'complete', 'epochs_completed': 25,
            'model': identity['model'], 'fold': identity['fold'], 'identity': identity,
            'identity_sha256': digest(identity), 'weights': retained, 'files': files,
            'test_used_for_selection_or_calibration': False,
            'aggregation': 'Mean TTA logits then softmax; equal image-probability mean across two checkpoints; confidence-weighted mean across images in each patient cluster'}
    atomic_json(folder / 'job_complete.json', meta)
    atomic_json(folder / 'job_status.json', {'status': 'complete', 'completed_epochs': 25})
    return meta


def train_job(inputs, manifest, paths, model_key, fold, folder, input_root, seconds_left):
    if not torch.cuda.is_available():
        raise RuntimeError('Select GPU T4 x2 for the Protocol A training notebook.')
    if timm.__version__ != '1.0.29':
        raise ValueError('Run setup first; timm 1.0.29 is required.')
    started = time.monotonic()
    identity = identity_for(inputs, manifest, model_key, fold)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / 'job_complete.json'
    if marker.exists():
        return {'complete': True, 'metadata': inspect_job(folder, inputs, identity['input_manifest_sha256'], identity)}
    manager = CheckpointManager(folder, model_key, fold, identity)
    resume = manager.load_auto([folder, Path(input_root)])
    worker = original_worker(inputs)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    model, source_sha = network(manifest['models'][model_key], identity['seed'], resume is None, torch.device('cuda:0'))
    if source_sha:
        atomic_json(folder / 'pretrained_source.json', {'config': manifest['models'][model_key], 'file_sha256': source_sha})
    assignment, frames = make_frames(inputs, fold, paths)
    generator = torch.Generator().manual_seed(identity['seed'] + 10000)
    loaders = make_loaders(worker, frames, generator)
    labels = frames['train'].label.to_numpy()
    class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.arange(6), y=labels),
                                 dtype=torch.float32, device='cuda:0')
    mixup = Mixup(mixup_alpha=.2, cutmix_alpha=1.0, label_smoothing=.1, num_classes=6)
    soft_loss = SoftTargetCrossEntropy()
    hard_loss = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=.1)
    scaler = GradScaler('cuda', enabled=True)

    def train_epoch(model, optimizer, scaler, epoch):
        return worker.train_one_epoch(model, loaders['train'], optimizer, scaler, soft_loss, hard_loss, mixup, epoch)

    def selection_loss(model):
        p, y, ids = worker.validate(model, loaders['selection'])
        group = aggregate_images(p, y, ids)
        return log_loss(group['labels'], group['probs'], labels=list(range(6)))

    # Leave time for final two-checkpoint inference and saving outputs.
    budget = max(0, seconds_left - (time.monotonic() - started) - 600)
    result = run_epochs(model, optimizer_factory(worker), scaler, generator, train_epoch,
                        selection_loss, manager, resume_state=resume, session_seconds=budget,
                        status_path=folder / 'job_status.json', minimum_epoch_budget=180)
    if not result['complete']:
        return {'complete': False, 'completed_epochs': result['completed_epochs']}
    if seconds_left - (time.monotonic() - started) < 180:
        atomic_json(folder / 'job_status.json', {'status': 'trained_inference_pending', 'completed_epochs': 25})
        return {'complete': False, 'completed_epochs': 25}
    meta = export_job(worker, model, frames, loaders, result, folder, identity, assignment)
    inspect_job(folder, inputs, identity['input_manifest_sha256'], identity)
    # Completed jobs retain both selected weights; full optimizer states are needed only for the active job.
    for slot in ('a', 'b'):
        for suffix in ('.pt', '.json'):
            path = folder / f'{model_key}_fold{fold}_resume_{slot}{suffix}'
            if path.is_file():
                path.unlink()
    del result, model, loaders, scaler, class_weights, mixup, soft_loss, hard_loss, resume
    gc.collect()
    torch.cuda.empty_cache()
    print(f'COMPLETE: Protocol A {model_key} fold {fold}; 25 epochs, both weights and final predictions saved.', flush=True)
    return {'complete': True, 'metadata': meta}
