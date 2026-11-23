"""Frozen data checks and patient-cluster aggregation for revised Protocol A."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import numpy as np
import pandas as pd
from completion_core import sha256, atomic_json, atomic_array, extract_small_zip


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def find_study(root, expected_a_sha):
    root = Path(root)
    candidates = list(root.rglob('protocol_a_manifest.json'))
    if not candidates:
        for archive in sorted(root.rglob('MSLD_Final_Run_Inputs.zip')):
            candidates += list(extract_small_zip(archive).rglob('protocol_a_manifest.json'))
    candidates = [p for p in candidates if sha256(p) == expected_a_sha
                  and (p.parent / 'original_protocol_a_worker.py').is_file()
                  and (p.parent / 'splits/fold1.csv').is_file()]
    if not candidates:
        raise FileNotFoundError('Attach the private dataset made from MSLD_Final_Run_Inputs.zip. Its frozen manifest must match this notebook.')
    return sorted(candidates)[0].parent


def verify_design(inputs, expected_sha):
    inputs = Path(inputs)
    marker = inputs / 'protocol_a_manifest.json'
    if sha256(marker) != expected_sha:
        raise ValueError('Protocol A manifest differs from the notebook.')
    manifest = json.loads(marker.read_text())
    for name, record in manifest['files'].items():
        path = (inputs / name).resolve()
        if not path.is_relative_to(inputs.resolve()) or not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Frozen design file missing or changed: {name}')
    clinical = pd.read_csv(inputs / 'clinical_images.csv')
    if len(clinical) != 755 or clinical.filename.nunique() != 755 or clinical.patient_id.nunique() != 524 or clinical.patient_cluster.nunique() != 520:
        raise ValueError('Expected 755 clinical images, 524 filename identifiers and 520 known-duplicate-aware groups.')
    if clinical.filename.str.contains('keywords', case=False).any():
        raise ValueError('Keywords.jpg must never enter the clinical manifest.')
    outer_groups = []
    for fold in range(1, 6):
        a = pd.read_csv(inputs / f'splits/fold{fold}.csv')
        pd.testing.assert_frame_equal(a[clinical.columns], clinical)
        if set(a.role) != {'train', 'selection', 'calibration', 'test'}:
            raise ValueError('The four fixed roles are required.')
        if not a.groupby('patient_cluster').role.nunique().eq(1).all() or not a.groupby('patient_id').role.nunique().eq(1).all():
            raise ValueError('A patient or confirmed duplicate group crosses partitions.')
        for _, part in a.groupby('role'):
            if set(part.label) != set(range(6)):
                raise ValueError('Each partition must contain all six classes.')
        outer_groups += sorted(a.loc[a.role == 'test', 'patient_cluster'].unique())
    if len(outer_groups) != len(set(outer_groups)) or set(outer_groups) != set(clinical.patient_cluster):
        raise ValueError('Every group must be tested once across the five outer folds.')
    return manifest, clinical


def find_images(root, clinical, check_hashes=True):
    candidates = []
    for folder in Path(root).rglob('Original Images'):
        files = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        found = {}
        for p in sorted(files):
            found.setdefault(p.name.upper(), p)
        if set(found) == set(clinical.filename.str.upper()):
            candidates.append((len(folder.parts), folder, found))
    if not candidates:
        raise FileNotFoundError('Attach the original MSLD v2.0 dataset by joydippaul, including Original Images.')
    _, folder, paths = sorted(candidates, key=lambda x: (x[0], str(x[1])))[0]
    if check_hashes:
        def check(row):
            p = paths[row.filename.upper()]
            if sha256(p) != row.sha256:
                raise ValueError(f'Clinical image content differs: {row.filename}')
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(check, clinical.itertuples(index=False)))
    return {row.filename: paths[row.filename.upper()] for row in clinical.itertuples(index=False)}


def preflight(input_root, expected_sha, verify_content=True):
    inputs = find_study(input_root, expected_sha)
    manifest, clinical = verify_design(inputs, expected_sha)
    paths = find_images(input_root, clinical, verify_content)
    return inputs, manifest, clinical, paths


def aggregate_images(probs, labels, clusters):
    """The original A confidence-weighted image mean, with fixed group labels."""
    p = np.asarray(probs, dtype=np.float64)
    y, ids = np.asarray(labels), np.asarray(clusters).astype(str)
    if p.shape != (len(y), 6) or len(y) != len(ids) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid per-image predictions.')
    p /= p.sum(axis=1, keepdims=True)
    unique, vectors, targets = np.sort(np.unique(ids)), [], []
    for pid in unique:
        mask = ids == pid
        if len(set(y[mask])) != 1:
            raise ValueError('Group has inconsistent labels.')
        images = p[mask]
        weights = images.max(axis=1)
        weights /= weights.sum()
        vectors.append((weights[:, None] * images).sum(axis=0))
        targets.append(y[mask][0])
    return {'probs': np.stack(vectors), 'labels': np.asarray(targets, dtype=np.int64), 'pids': unique}


def validate_group(group):
    p, y, ids = (np.asarray(group[k]) for k in ('probs', 'labels', 'pids'))
    if p.shape != (len(y), 6) or len(ids) != len(y) or len(set(ids)) != len(ids) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError('Invalid patient-cluster predictions.')
    if not np.isin(y, range(6)).all():
        raise ValueError('Invalid class labels.')
    return group


def read_group(folder, prefix):
    return validate_group({k: np.load(Path(folder) / f'{prefix}_{k}.npy', allow_pickle=False) for k in ('probs', 'labels', 'pids')})


def save_group(folder, prefix, group):
    validate_group(group)
    records = {}
    for kind, array in group.items():
        name = f'{prefix}_{kind}.npy'
        atomic_array(Path(folder) / name, array)
        records[name] = {'bytes': (Path(folder) / name).stat().st_size, 'sha256': sha256(Path(folder) / name)}
    return records


def inspect_job(folder, inputs, expected_a_sha, expected_identity=None, require_weights=True):
    folder = Path(folder)
    meta = json.loads((folder / 'job_complete.json').read_text())
    if meta.get('status') != 'complete' or meta.get('epochs_completed') != 25 or meta['identity']['input_manifest_sha256'] != expected_a_sha:
        raise ValueError('A job is incomplete or belongs to another design.')
    if meta['identity_sha256'] != digest(meta['identity']) or (expected_identity is not None and meta['identity'] != expected_identity):
        raise ValueError('Saved job identity differs from this run.')
    model, fold = meta['model'], int(meta['fold'])
    required = {f'{model}_fold{fold}_{split}_{kind}.npy'
                for split in ('test', 'val') for kind in ('probs', 'labels', 'pids')}
    required |= {'epoch_history.json', 'assignment.csv',
                 f'{model}_fold{fold}_rank1.pt', f'{model}_fold{fold}_rank2.pt'}
    if not required.issubset(meta['files']) or len(meta['weights']) != 2:
        raise ValueError('Completed job is missing required prediction, history or weight records.')
    for name, record in meta['files'].items():
        p = (folder / name).resolve()
        if not require_weights and p.suffix == '.pt' and not p.is_file():
            # The small final analysis ZIP intentionally excludes selected weight tensors.
            continue
        if not p.is_relative_to(folder.resolve()) or not p.is_file() or p.stat().st_size != record['bytes'] or sha256(p) != record['sha256']:
            raise ValueError(f'Completed job artifact missing or changed: {p}')
    history = json.loads((folder / 'epoch_history.json').read_text())
    if [h['epoch'] for h in history] != list(range(1, 26)) or any(h['phase'] != ('head' if h['epoch'] <= 3 else 'full') for h in history):
        raise ValueError('Incomplete history or different freeze schedule.')
    ranked = sorted(history, key=lambda h: h['selection_logloss'])[:2]
    if [h['epoch'] for h in ranked] != [w['epoch'] for w in meta['weights']]:
        raise ValueError('Selected checkpoints differ from selection history.')
    assignment = pd.read_csv(Path(inputs) / f'splits/fold{fold}.csv')
    for split, role in (('test', 'test'), ('val', 'calibration')):
        group = read_group(folder, f'{model}_fold{fold}_{split}')
        expected = assignment.loc[assignment.role == role].groupby('patient_cluster').label.first().to_dict()
        if set(group['pids']) != set(expected) or not all(int(expected[pid]) == int(y) for pid, y in zip(group['pids'], group['labels'])):
            raise ValueError('Completed group IDs or labels differ from the fixed split.')
    return meta
