"""Portable data checks and pooling for the Protocol B completion runs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd

MODELS = ('swin_small', 'beit_base')
KINDS = ('probs', 'labels', 'pids')
SPLITS = ('test', 'val')

# Kaggle may unpack/recompress nested .gz files during dataset ingestion. These
# pins verify the exact original decompressed CSV bytes for the v1 input package.
# Prediction arrays and every other input retain their original byte-hash checks.
GZIP_REFERENCE_CONTENT = {
    'augmented_fold4.csv.gz': {
        'archive_sha256': '903a87b954d1d16a592796d0831a25a2d407c27fc710dff10eabcdd0c1d8884a',
        'content_sha256': 'dacf38f8efca15764e39d5a795c7f761fc70ad5a99c531546473f5df02fc6734',
        'content_bytes': 2217208,
    },
    'augmented_fold5.csv.gz': {
        'archive_sha256': 'f59285406632f00d1e040cc67e7ca3f536b710c462471ad082b050adc9642c7b',
        'content_sha256': '0bfc5fed4530001bac81a29497944fb45ff7c058ed75c5c9025b0d5b58d2dec8',
        'content_bytes': 2282103,
    },
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def atomic_array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as f:
        np.save(f, value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def inside(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f'Path leaves its input folder: {relative}')
    return result


def extract_small_zip(archive, cache_root=None):
    archive = Path(archive)
    cache = Path(cache_root) if cache_root else Path.cwd() / '.protocol_b_archive_cache'
    destination = cache / sha256(archive)
    with ZipFile(archive) as zipped:
        if sum(info.file_size for info in zipped.infolist()) > 100 * 1024 * 1024:
            raise ValueError('Expected the small inputs/results ZIP, not a full model-output archive.')
        for info in zipped.infolist():
            relative = PurePosixPath(info.filename.replace('\\', '/'))
            if relative.is_absolute() or '..' in relative.parts or any(':' in p for p in relative.parts) or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Unsafe path inside ZIP archive.')
            target = inside(destination, relative.as_posix())
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(info) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
    return destination


def find_inputs(search_root, expected_sha256=None):
    root = Path(search_root)
    candidates = sorted(root.rglob('completion_input_manifest.json'))
    if root.name == 'completion_input_manifest.json':
        candidates = [root]
    if not candidates:
        for archive in sorted(root.rglob('protocol_b_completion_inputs.zip')):
            extracted = extract_small_zip(archive)
            candidates.extend(sorted(extracted.rglob('completion_input_manifest.json')))
        if not candidates:
            raise FileNotFoundError('Attach protocol_b_completion_inputs.zip as a Kaggle dataset first.')
    if expected_sha256:
        candidates = [p for p in candidates if sha256(p) == expected_sha256]
        if not candidates:
            raise ValueError('The attached completion input package differs from this notebook.')
    digests = {sha256(p) for p in candidates}
    if len(digests) != 1:
        raise ValueError('Several different completion input packages are attached; keep one version.')
    return candidates[0].parent


def verified_input_path(root, relative, record):
    """Return a verified path and its explicit pandas compression setting."""
    root = Path(root)
    path = inside(root, relative)
    if path.is_file() and path.stat().st_size == record['bytes'] and sha256(path) == record['sha256']:
        return path, 'gzip' if relative.endswith('.gz') else None
    content = GZIP_REFERENCE_CONTENT.get(relative)
    if content and record['sha256'] == content['archive_sha256']:
        plain_name = Path(relative).name[:-3]
        candidates = [path, path.with_suffix('')]
        candidates += sorted(root.rglob(plain_name))
        seen = set()
        for candidate in candidates:
            if not candidate.is_file() or str(candidate.resolve()) in seen:
                continue
            seen.add(str(candidate.resolve()))
            if not candidate.resolve().is_relative_to(root.resolve()):
                raise ValueError('An extracted reference file leaves its input folder.')
            try:
                with candidate.open('rb') as source:
                    compressed = source.read(2) == b'\x1f\x8b'
                opener = gzip.open if compressed else open
                with opener(candidate, 'rb') as source:
                    payload = source.read(content['content_bytes'] + 1)
            except (OSError, EOFError):
                continue
            if len(payload) == content['content_bytes'] and hashlib.sha256(payload).hexdigest() == content['content_sha256']:
                return candidate, 'gzip' if compressed else None
        available = [p.relative_to(root).as_posix() for p in candidates if p.is_file()]
        raise ValueError(f'Input missing or changed: {relative}. Exact decompressed CSV content did not match. Candidate files: {available}')
    raise ValueError(f'Input missing or changed: {relative}')


def verify_inputs(input_dir, expected_sha256=None):
    root = Path(input_dir)
    marker = root / 'completion_input_manifest.json'
    if expected_sha256 and sha256(marker) != expected_sha256:
        raise ValueError('Completion input manifest SHA-256 differs.')
    manifest = json.loads(marker.read_text(encoding='utf-8'))
    if manifest['format_version'] != 1:
        raise ValueError('Unsupported completion input format.')
    for relative, record in manifest['files'].items():
        p, compression = verified_input_path(root, relative, record)
        if relative in GZIP_REFERENCE_CONTENT and (p.name != Path(relative).name or p.stat().st_size != record['bytes']):
            print(f'Verified exact reference CSV contents after Kaggle archive processing: {p.name}', flush=True)
    return manifest


def find_dataset(search_root, expected_excel_sha):
    choices = []
    for p in sorted(Path(search_root).rglob('datalog.xlsx')):
        if (p.parent / 'Original Images').is_dir() and (p.parent / 'Augmented Images').is_dir():
            if sha256(p) == expected_excel_sha:
                choices.append(p.parent)
    if len(choices) != 1:
        raise ValueError(f'Expected one unchanged MSLD v2.0 dataset with datalog.xlsx; found {len(choices)}. Attach the original MSLD v2.0 dataset only once.')
    return choices[0]


def load_predictions(directory, prefix):
    result = {k: np.load(Path(directory) / f'{prefix}_{k}.npy', allow_pickle=(k == 'pids')) for k in KINDS}
    p, y, ids = (result[k] for k in KINDS)
    if p.shape != (len(y), 6) or y.shape != (len(ids),):
        raise ValueError(f'Prediction dimensions differ: {prefix}')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-5):
        raise ValueError(f'Invalid probabilities: {prefix}')
    if not np.isin(y, np.arange(6)).all() or len(set(ids)) != len(ids):
        raise ValueError(f'Invalid labels or repeated IDs: {prefix}')
    return result


def pool_predictions(groups):
    # Same arithmetic as the V34 worker; repeated patients get one mean vector.
    p = np.concatenate([g['probs'] for g in groups])
    y = np.concatenate([g['labels'] for g in groups])
    ids = np.concatenate([g['pids'] for g in groups])
    frame = pd.DataFrame({'pid': ids, 'label': y, 'probs': list(p)})
    if not (frame.groupby('pid')['label'].nunique() == 1).all():
        raise ValueError('A patient has inconsistent labels between folds.')
    if len(frame) != frame['pid'].nunique():
        frame = frame.groupby('pid').agg({'label': 'first', 'probs': lambda x: np.mean(np.stack(x), axis=0)}).reset_index()
    frame = frame.sort_values('pid').reset_index(drop=True)
    return {'probs': np.stack(frame['probs'].values), 'labels': frame['label'].values, 'pids': frame['pid'].values}


def clean_stem(value):
    return Path(str(value).replace('\\', '/')).stem.strip().upper()


def verify_sheet(dataset, assignment, fold):
    sheets = pd.read_excel(Path(dataset) / 'datalog.xlsx', sheet_name=None, engine='openpyxl')
    sheet = next((sheets[k] for k in (f'Sheet{fold}', f'Fold{fold}', f'fold_{fold}', f'sheet_{fold}', f'Sheet {fold}', f'Fold {fold}') if k in sheets), None)
    if sheet is None:
        raise ValueError(f'Official sheet missing for fold {fold}; fallback splitting is disabled.')
    for token, roles in (('train', ['train_original']), ('val', ['val_selection', 'val_calibration']), ('test', ['test'])):
        columns = [c for c in sheet.columns if token in str(c).lower()]
        if len(columns) != 1:
            raise ValueError(f'Official {token} column is ambiguous for fold {fold}.')
        expected = {clean_stem(v) for v in sheet[columns[0]].dropna()}
        actual = {clean_stem(v) for v in assignment.loc[assignment.role.isin(roles), 'filename']}
        if expected != actual:
            raise ValueError(f'Official {token} assignments differ for fold {fold}.')


def prepare_fold(input_dir, dataset_dir, model, fold, expected_manifest_sha=None, verify_content=True):
    if model not in MODELS or fold not in (4, 5):
        raise ValueError('Only Swin-Small or BEiT-Base folds 4 and 5 may be trained by this package.')
    inputs, dataset = Path(input_dir), Path(dataset_dir)
    manifest = verify_inputs(inputs, expected_manifest_sha)
    if sha256(dataset / 'datalog.xlsx') != manifest['datalog_sha256']:
        raise ValueError('The dataset spreadsheet differs from the original release.')
    originals = pd.read_csv(inputs / 'original_images.csv')
    assignment = pd.read_csv(inputs / f'reference_assignments/fold{fold}.csv')
    augmented_name = f'augmented_fold{fold}.csv.gz'
    augmented_path, compression = verified_input_path(inputs, augmented_name, manifest['files'][augmented_name])
    augmented = pd.read_csv(augmented_path, compression=compression)
    verify_sheet(dataset, assignment, fold)
    original_paths = {}
    original_files = []
    for path in sorted((dataset / 'Original Images').rglob('*')):
        if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
            original_files.append(path)
            original_paths.setdefault(path.name.upper(), path)
    if set(original_paths) != {v.upper() for v in originals.filename}:
        raise ValueError('The 755 clinical original filenames differ from the reference release.')
    if len(original_files) != 3775 or len(originals) != 755 or originals.patient_id.nunique() != 524:
        raise ValueError('Expected 3775 original fold copies, 755 distinct clinical images, and 524 identifiers.')
    aug_count = sum(1 for p in (dataset / 'Augmented Images').rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
    if aug_count != 37044:
        raise ValueError(f'Expected 37044 released augmented files; found {aug_count}.')
    expected_by_name = originals.set_index(originals.filename.str.upper()).to_dict('index')
    for row in assignment.to_dict('records'):
        expected = expected_by_name[row['filename'].upper()]
        if int(row['label']) != int(expected['label']) or row['patient_id'] != expected['patient_id']:
            raise ValueError('Original image identity or class differs.')
    assignment['filepath'] = [str(original_paths[name.upper()]) for name in assignment.filename]
    augmented['filepath'] = [str(inside(dataset, rel)) for rel in augmented.relative_path]
    checks = [(original_paths[r['filename'].upper()], r['sha256']) for r in originals.to_dict('records')]
    checks += [(Path(r['filepath']), r['sha256']) for r in augmented.to_dict('records')]
    if verify_content:
        def verify_one(item):
            path, expected = item
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f'Dataset image missing or changed: {path}')
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(verify_one, checks))
    train_original = assignment.loc[assignment.role == 'train_original'].reset_index(drop=True)
    val = assignment.loc[assignment.role.isin(['val_selection', 'val_calibration'])].reset_index(drop=True)
    test = assignment.loc[assignment.role == 'test'].reset_index(drop=True)
    train_ids, val_ids, test_ids = (set(f.patient_id) for f in (train_original, val, test))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError('Patient identifiers overlap within a fold.')
    if not set(augmented.patient_id).issubset(train_ids):
        raise ValueError('An augmented training file belongs to a held-out identifier.')
    train_labels = train_original.groupby('patient_id').label.first().to_dict()
    if not all(int(row.label) == int(train_labels[row.patient_id]) for row in augmented.itertuples()):
        raise ValueError('An augmented image has an inconsistent class.')
    train = pd.concat([train_original, augmented], ignore_index=True)
    val_pids = np.sort(val.patient_id.unique())
    shuffled = np.random.RandomState(42 + fold - 1).permutation(len(val_pids))
    half = len(shuffled) // 2
    selection_indices, calibration_indices = shuffled[:half], shuffled[half:]
    for indices, role in ((selection_indices, 'val_selection'), (calibration_indices, 'val_calibration')):
        if set(val_pids[indices]) != set(assignment.loc[assignment.role == role, 'patient_id']):
            raise ValueError('Selection/calibration patient halves differ from the original run.')
    counts = np.array([(train_original.label.values == c).sum() for c in range(6)])
    if (counts == 0).any():
        raise ValueError('A class is missing from the original training images.')
    weights = (1 - .999) / (1 - np.power(.999, counts))
    weights = (weights / weights.sum() * 6).astype(np.float32)
    signature_data = {'input_manifest_sha256': sha256(inputs / 'completion_input_manifest.json'),
                      'model': model, 'fold': fold,
                      'assignment_sha256': sha256(inputs / f'reference_assignments/fold{fold}.csv'),
                      'selected_augmented_sha256': manifest['files'][augmented_name]['sha256']}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    summary = {'model': model, 'fold': fold, 'original_images': 755, 'patient_identifiers': 524,
               'n_train_original': len(train_original), 'n_augmented_used': len(augmented),
               'n_val': len(val), 'n_test': len(test), 'n_selection_patients': len(selection_indices),
               'n_calibration_patients': len(calibration_indices), 'n_test_patients': len(test_ids),
               'dataset_image_hashes_checked': len(checks) if verify_content else 0,
               'data_signature': signature, 'training_seed': 42 + fold - 1,
               'reference_assignments_and_content_match': verify_content,
               'old_cross_fold_rng_state_restored': False,
               'seed_note': 'Fresh independent replacement fold; seed 42+zero-based-fold-index. The original cross-fold RNG stream was not saved.'}
    return {'manifest': manifest, 'train': train, 'train_original': train_original,
            'val': val, 'test': test, 'assignment': assignment, 'augmented': augmented,
            'selection_indices': selection_indices, 'calibration_indices': calibration_indices,
            'val_pids': val_pids, 'class_weights': weights, 'summary': summary}


def verify_frozen_reconstruction(input_dir, model):
    root = Path(input_dir) / 'original_reference' / model
    for split in SPLITS:
        pooled = pool_predictions([load_predictions(root, f'{model}_fold{f}_{split}') for f in (1, 2, 3, 4)])
        old = load_predictions(root, f'{model}_{split}')
        if not all(np.array_equal(pooled[k], old[k]) for k in KINDS):
            raise ValueError(f'Recovered {model} folds do not reproduce original {split} predictions.')
    return True
