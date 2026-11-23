"""Epoch-boundary checkpoints with model, optimizer, scheduler, AMP, and RNG state."""
from pathlib import Path
import copy
import hashlib
import json
import os
import random
import time

import numpy as np
import torch

from completion_core import atomic_json, sha256


def identity_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng(generator):
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': {'name': state[0], 'keys': state[1].tolist(), 'position': int(state[2]),
                      'has_gauss': int(state[3]), 'cached_gaussian': float(state[4])},
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'train_generator': generator.get_state()}


def restore_rng(value, generator):
    random.setstate(value['python'])
    n = value['numpy']
    np.random.set_state((n['name'], np.asarray(n['keys'], dtype=np.uint32), n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(value['torch_cpu'].cpu())
    if value['torch_cuda']:
        if not torch.cuda.is_available() or len(value['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from the checkpoint.')
        torch.cuda.set_rng_state_all(value['torch_cuda'])
    generator.set_state(value['train_generator'].cpu())


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


class CheckpointManager:
    """Two alternating slots preserve an older usable checkpoint after a torn write."""
    def __init__(self, directory, model, fold, identity):
        self.directory = Path(directory)
        self.model, self.fold, self.identity = model, int(fold), identity
        self.identity_sha256 = identity_digest(identity)
        self.stem = f'{model}_fold{fold}'
        self.last_path = None

    def save(self, state):
        epoch = int(state['next_epoch'])
        slot = 'a' if epoch % 2 == 0 else 'b'
        path = self.directory / f'{self.stem}_resume_{slot}.pt'
        payload = dict(state, format_version=1, identity=self.identity,
                       identity_sha256=self.identity_sha256)
        atomic_torch_save(path, payload)
        atomic_json(path.with_suffix('.json'), {'format_version': 1, 'model': self.model,
                    'fold': self.fold, 'completed_epochs': epoch,
                    'identity_sha256': self.identity_sha256,
                    'bytes': path.stat().st_size, 'sha256': sha256(path)})
        self.last_path = path
        print(f'Checkpoint saved after {epoch} completed epochs: {path.name}', flush=True)
        return path

    def load_auto(self, search_roots):
        candidates, incompatible, malformed, seen = [], [], [], set()
        for root in search_roots:
            root = Path(root)
            if not root.exists():
                continue
            for sidecar in root.rglob(f'{self.stem}_resume_?.json'):
                canonical = str(sidecar.resolve())
                if canonical in seen:
                    continue
                seen.add(canonical)
                try:
                    meta = json.loads(sidecar.read_text(encoding='utf-8'))
                    if meta['model'] != self.model or int(meta['fold']) != self.fold or meta['format_version'] != 1:
                        raise ValueError('Checkpoint job metadata differs.')
                    if meta['identity_sha256'] != self.identity_sha256:
                        incompatible.append(str(sidecar))
                    else:
                        candidates.append((int(meta['completed_epochs']), sidecar, meta))
                except (OSError, KeyError, ValueError) as error:
                    malformed.append(f'{sidecar}: {error}')
        if incompatible:
            raise ValueError('A checkpoint for this fold uses different data, settings, code, or environment. Restore its original Kaggle environment and input package before resuming.\n' + '\n'.join(incompatible))
        if not candidates:
            if malformed:
                raise ValueError('Checkpoint sidecars were found but could not be read.\n' + '\n'.join(malformed))
            return None
        # Validate each candidate without loading several large tensor files into RAM.
        valid = []
        for epoch, sidecar, meta in sorted(candidates, key=lambda x: x[0], reverse=True):
            path = sidecar.with_suffix('.pt')
            if path.is_file() and path.stat().st_size == meta['bytes'] and sha256(path) == meta['sha256']:
                valid.append((epoch, path, meta))
            else:
                print(f'Ignoring incomplete checkpoint slot: {path}', flush=True)
        if not valid:
            raise ValueError('No checkpoint passed its byte-count and SHA-256 checks. Attach the full saved notebook outputs, including both .pt and .json files.')
        latest = max(v[0] for v in valid)
        top = [v for v in valid if v[0] == latest]
        if len({v[2]['sha256'] for v in top}) != 1:
            raise ValueError('Different checkpoints exist at the same epoch. Keep only the intended saved output version attached.')
        _, path, _ = top[0]
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['identity_sha256'] != self.identity_sha256 or state['identity'] != self.identity:
            raise ValueError('Checkpoint identity differs from its sidecar.')
        if int(state['next_epoch']) != latest:
            raise ValueError('Checkpoint epoch differs from its sidecar.')
        self.last_path = path
        print(f'Resuming from {latest} completed epochs: {path}', flush=True)
        return state


def run_epochs(model, make_optimizer_scheduler, scaler, generator, train_epoch,
               selection_loss, manager, total_epochs=25, freeze_epochs=3,
               resume_state=None, session_seconds=None, max_epochs_this_session=None,
               status_path=None, minimum_epoch_budget=900):
    """Never checkpoint or publish a partially trained epoch as complete."""
    started = time.monotonic()
    next_epoch = int(resume_state['next_epoch']) if resume_state else 0
    if next_epoch < 0 or next_epoch > total_epochs:
        raise ValueError('Checkpoint epoch is outside the configured training schedule.')
    phase = resume_state['phase'] if resume_state else 'head'
    expected_phase = 'head' if next_epoch <= freeze_epochs else 'full'
    if phase != expected_phase:
        raise ValueError('Saved optimizer phase differs from the freeze schedule.')
    optimizer, scheduler = make_optimizer_scheduler(model, phase)
    if resume_state:
        model.load_state_dict(resume_state['model'], strict=True)
        # Scheduler construction must happen before loading the optimizer's LR state.
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        scaler.load_state_dict(resume_state['scaler'])
        history = list(resume_state['history'])
        snapshots = list(resume_state['snapshots'])
        if [int(row['epoch']) for row in history] != list(range(1, next_epoch + 1)):
            raise ValueError('Checkpoint training history contains gaps or repeated epochs.')
        restore_rng(resume_state['rng'], generator)
    else:
        history, snapshots = [], []

    def save_state():
        value = {'next_epoch': next_epoch, 'phase': phase,
                 'model': cpu_copy(model.state_dict()),
                 'optimizer': cpu_copy(optimizer.state_dict()),
                 'scheduler': cpu_copy(scheduler.state_dict()),
                 'scaler': cpu_copy(scaler.state_dict()),
                 'history': copy.deepcopy(history), 'snapshots': cpu_copy(snapshots),
                 'rng': capture_rng(generator)}
        manager.save(value)
        if status_path:
            atomic_json(status_path, {'status': 'trained_inference_pending' if next_epoch == total_epochs else 'training',
                        'completed_epochs': next_epoch, 'required_epochs': total_epochs,
                        'checkpoint': manager.last_path.name})

    if resume_state is None:
        save_state()
    completed_here = 0
    for epoch in range(next_epoch, total_epochs):
        if max_epochs_this_session is not None and completed_here >= max_epochs_this_session:
            break
        estimate = max([row['epoch_seconds'] for row in history[-3:]] or [0])
        needed = max(minimum_epoch_budget, estimate * 2) + 300
        if session_seconds is not None and session_seconds - (time.monotonic() - started) < needed:
            print('Stopping before another epoch to leave time for output saving.', flush=True)
            break
        if epoch == freeze_epochs:
            if phase != 'head':
                raise ValueError('Backbone-unfreezing transition would be applied twice.')
            phase = 'full'
            optimizer, scheduler = make_optimizer_scheduler(model, phase)
        epoch_started = time.monotonic()
        mean_train_loss = train_epoch(model, optimizer, scaler, epoch)
        scheduler.step(epoch if epoch < freeze_epochs else epoch - freeze_epochs)
        score = float(selection_loss(model))
        if not np.isfinite(score):
            raise ValueError('Selection log-loss is not finite; the last complete checkpoint remains available.')
        snapshots.append({'selection_logloss': score, 'epoch': epoch + 1,
                          'state_dict': cpu_copy(model.state_dict())})
        snapshots.sort(key=lambda value: value['selection_logloss'])
        snapshots = snapshots[:2]
        history.append({'epoch': epoch + 1, 'phase': phase,
                        'mean_training_loss': None if mean_train_loss is None else float(mean_train_loss),
                        'selection_logloss': score,
                        'learning_rates': [float(g['lr']) for g in optimizer.param_groups],
                        'epoch_seconds': float(time.monotonic() - epoch_started)})
        next_epoch = epoch + 1
        completed_here += 1
        save_state()
        print(f'Epoch {next_epoch}/{total_epochs}: selection log-loss {score:.6f}; retained epochs {[s["epoch"] for s in snapshots]}', flush=True)
    complete = next_epoch == total_epochs
    if not complete and status_path:
        atomic_json(status_path, {'status': 'paused', 'completed_epochs': next_epoch,
                    'required_epochs': total_epochs,
                    'instruction': 'Save notebook outputs. Attach those outputs to the same fold notebook in a new session; automatic resume will continue from the last saved epoch.'})
    return {'complete': complete, 'completed_epochs': next_epoch,
            'history': history, 'snapshots': snapshots}
