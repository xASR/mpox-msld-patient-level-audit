"""Verified inputs for duplicate-corrected official-fold Protocol B."""
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile
import hashlib
import json
import os
import shutil
import stat
import numpy as np
import pandas as pd

MODELS = ('efficientnetv2_s', 'convnext_tiny', 'swin_small', 'beit_base')
CLASSES = ['Healthy', 'Cowpox', 'Monkeypox', 'Measles', 'Chickenpox', 'HFMD']
MARKER = 'pb_corrected_manifest.json'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f'Path leaves its root: {relative}')
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        np.save(stream, value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def safe_extract(archive, destination, max_bytes=100*1024*1024):
    destination = Path(destination)
    with ZipFile(archive) as zipped:
        if sum(info.file_size for info in zipped.infolist()) > max_bytes:
            raise ValueError('Expected a small inputs/results ZIP, not a full weight archive.')
        for info in zipped.infolist():
            rel = PurePosixPath(info.filename.replace('\\', '/'))
            if rel.is_absolute() or '..' in rel.parts or any(':' in p for p in rel.parts) or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Unsafe path in ZIP.')
            target = inside(destination, rel.as_posix())
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(info) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
    return destination


def find_inputs(search_root, expected_sha):
    search_root = Path(search_root)
    candidates = sorted(search_root.rglob(MARKER))
    if not candidates:
        for archive in sorted(search_root.rglob('PB_Corrected_Inputs.zip')):
            target = Path('/tmp/pb_corrected_inputs') / sha256(archive)
            safe_extract(archive, target)
            candidates.extend(target.rglob(MARKER))
    candidates = [p for p in candidates if sha256(p) == expected_sha]
    if not candidates:
        raise FileNotFoundError('Attach the new PB_Corrected_Inputs.zip dataset matching this notebook. Old completion/final-run inputs are not compatible.')
    return sorted(candidates)[0].parent


def verify_inputs(root, expected_sha):
    root = Path(root)
    if sha256(root/MARKER) != expected_sha:
        raise ValueError('Input manifest changed.')
    manifest = json.loads((root/MARKER).read_text(encoding='utf-8'))
    if manifest['schema'] != 'pb_known_duplicate_corrected_v1':
        raise ValueError('Wrong protocol inputs.')
    for name, record in manifest['files'].items():
        path = inside(root, name)
        if not path.is_file() or sha256(path) != record['sha256']:
            raise ValueError(f'Frozen input changed: {name}')
    return manifest


def find_dataset(search_root, expected_sha):
    choices = [p.parent for p in sorted(Path(search_root).rglob('datalog.xlsx'))
               if (p.parent/'Original Images').is_dir() and (p.parent/'Augmented Images').is_dir()
               and sha256(p)==expected_sha]
    if len(choices)!=1:
        raise ValueError(f'Expected one original MSLD v2.0 dataset with its unchanged spreadsheet; found {len(choices)}.')
    return choices[0]


def check_group_separation(assignment, augmented):
    roles = ['train_original', 'val_selection', 'val_calibration', 'test']
    used = assignment[assignment.role.isin(roles)]
    for column in ['patient_id', 'patient_cluster', 'sha256', 'rgb_sha256']:
        n = used.groupby(column).role.nunique()
        if (n>1).any():
            raise ValueError(f'{column} crosses corrected roles: {n[n>1].index.tolist()}')
    train = used[used.role.eq('train_original')]
    held = used[~used.role.eq('train_original')]
    if not set(augmented.patient_id).issubset(set(train.patient_id)):
        raise ValueError('An augmented source identifier is held out.')
    if not set(augmented.patient_cluster).issubset(set(train.patient_cluster)):
        raise ValueError('An augmented source group is held out.')
    if set(augmented.sha256) & set(held.sha256):
        raise ValueError('An augmented training raster matches a held-out original file.')
    original_to_group = used.set_index('filename').patient_cluster.to_dict()
    for row in augmented.itertuples():
        if original_to_group.get(row.source_filename) != row.patient_cluster:
            raise ValueError('An augmented source is not a retained training original.')
    return True


def prepare_fold(inputs, dataset, fold, expected_sha, verify_content=True):
    inputs, dataset = Path(inputs), Path(dataset)
    manifest = verify_inputs(inputs, expected_sha)
    if sha256(dataset/'datalog.xlsx') != manifest['datalog_sha256']:
        raise ValueError('The original dataset spreadsheet differs.')
    clinical = pd.read_csv(inputs/'clinical_images.csv')
    assignment = pd.read_csv(inputs/f'splits/fold{fold}.csv')
    augmented = pd.read_csv(inputs/f'augmented/fold{fold}.csv')
    if len(clinical)!=755 or clinical.patient_id.nunique()!=524 or clinical.patient_cluster.nunique()!=520:
        raise ValueError('Expected the frozen 755-image, 524-identifier, 520-known-group inventory.')
    if set(assignment.filename) != set(clinical.filename) or len(assignment)!=755:
        raise ValueError('Corrected assignment inventory differs.')
    sheet = pd.read_excel(dataset/'datalog.xlsx', sheet_name=f'Sheet{fold}')
    official_test = set(sheet['test'].dropna().astype(str).str.strip().str.upper())
    if official_test != set(assignment.loc[assignment.role.eq('test'),'filename'].str.upper()):
        raise ValueError('Corrected run must preserve the official original-image test cohort.')
    check_group_separation(assignment, augmented)
    original_paths = {}
    for path in sorted((dataset/'Original Images').rglob('*')):
        if path.is_file() and path.suffix.lower() in ('.jpg','.jpeg','.png'):
            original_paths.setdefault(path.name.upper(),path)
    if set(original_paths)!=set(clinical.filename.str.upper()):
        raise ValueError('Clinical original filenames differ from the frozen inventory.')
    assignment['filepath'] = [str(original_paths[name.upper()]) for name in assignment.filename]
    augmented['filepath'] = [str(inside(dataset, rel)) for rel in augmented.relative_path]
    checks = {str(original_paths[row.filename.upper()]):row.sha256 for row in clinical.itertuples()}
    checks.update(dict(zip(augmented.filepath,augmented.sha256)))
    if verify_content:
        def verify(item):
            name, digest = item
            if not Path(name).is_file() or sha256(name)!=digest:
                raise ValueError(f'Dataset content differs: {name}')
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(verify,checks.items()))
    frames = {}
    for key, roles in [('train_original',['train_original']), ('selection',['val_selection']),
                       ('calibration',['val_calibration']), ('test',['test'])]:
        frame = assignment[assignment.role.isin(roles)].copy().reset_index(drop=True)
        if set(frame.label)!=set(range(6)):
            raise ValueError(f'A class is missing from {key}.')
        # The worker aggregates by this column. Keep raw filename identifiers too.
        frame['filename_identifier'] = frame.patient_id
        frame['patient_id'] = frame.patient_cluster
        frames[key] = frame
    augmented['filename_identifier'] = augmented.patient_id
    augmented['patient_id'] = augmented.patient_cluster
    frames['train'] = pd.concat([frames['train_original'],augmented],ignore_index=True)
    counts = np.bincount(frames['train_original'].label, minlength=6)
    weights = (1-.999)/(1-np.power(.999,counts))
    frames['class_weights'] = (weights/weights.sum()*6).astype(np.float32)
    frames['assignment'], frames['augmented'], frames['manifest'] = assignment, augmented, manifest
    summary = {'schema':manifest['schema'],'fold':fold,'clinical_images':755,'filename_identifiers':524,
               'known_groups':520,'test_images':len(frames['test']),
               'test_groups':frames['test'].patient_id.nunique(),
               'selection_groups':frames['selection'].patient_id.nunique(),
               'calibration_groups':frames['calibration'].patient_id.nunique(),
               'train_original_images':len(frames['train_original']),
               'train_augmented_entries':len(augmented),
               'excluded_original_images':int(assignment.role.str.startswith('excluded_').sum()),
               'known_group_role_overlap':0,'image_files_hash_checked':len(checks) if verify_content else 0,
               'data_signature':hashlib.sha256((sha256(inputs/f'splits/fold{fold}.csv')+
                                                  sha256(inputs/f'augmented/fold{fold}.csv')).encode()).hexdigest(),
               'scope':'Separation of known relationships, not independently verified clinical identity or proof that no unknown relatives exist.'}
    frames['summary'] = summary
    return frames


def pool_predictions(parts):
    rows = []
    for part in parts:
        for pid,label,prob in zip(part['pids'],part['labels'],part['probs']):
            rows.append({'pid':str(pid),'label':int(label),'prob':np.asarray(prob,dtype=float)})
    frame = pd.DataFrame(rows)
    if not (frame.groupby('pid').label.nunique()==1).all():
        raise ValueError('Inconsistent labels within a known group.')
    grouped = frame.groupby('pid',sort=True).agg(label=('label','first'),prob=('prob',lambda s:np.mean(np.stack(s),axis=0)))
    return {'pids':grouped.index.to_numpy(dtype=str),'labels':grouped.label.to_numpy(dtype=np.int64),
            'probs':np.stack(grouped.prob)}
