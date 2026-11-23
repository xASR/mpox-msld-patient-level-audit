"""Train the revised Protocol A, resume safely, then export publication outputs."""
from pathlib import Path
import argparse
import json
import shutil
import time
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from completion_core import atomic_json, atomic_array, sha256
from protocol_a_core import preflight, inspect_job, read_group, find_study, verify_design
from study_statistics import DISPLAY, fit_calibration, temperature, analyze_predictions


def restore_completed(input_root, output, inputs, manifest, expected_sha):
    from protocol_a_train import identity_for
    seen = {}
    for marker in sorted(Path(input_root).rglob('job_complete.json')):
        try:
            data = json.loads(marker.read_text())
        except (OSError, ValueError):
            continue
        if data.get('identity', {}).get('input_manifest_sha256') != expected_sha:
            continue
        model, fold = data['model'], int(data['fold'])
        if model not in manifest['models'] or fold not in range(1, 6):
            raise ValueError('Unexpected saved Protocol A job.')
        meta = inspect_job(marker.parent, inputs, expected_sha, identity_for(inputs, manifest, model, fold))
        key = (model, fold)
        fingerprint = {name: record['sha256'] for name, record in meta['files'].items()}
        if key in seen and seen[key] != fingerprint:
            raise ValueError('Two different completed versions of the same Protocol A fold are attached.')
        seen[key] = fingerprint
        destination = Path(output) / f'training/{model}/fold{fold}'
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / 'job_complete.json').exists():
            old = inspect_job(destination, inputs, expected_sha, meta['identity'])
            if old['files'] != meta['files']:
                raise ValueError('Working outputs differ from the attached completed fold.')
            continue
        for name in [*meta['files'], 'job_complete.json', 'job_status.json', 'pretrained_source.json']:
            source = marker.parent / name
            if source.is_file():
                shutil.copy2(source, destination / name)
        print(f'Restored completed {model} fold {fold}; training will be skipped.', flush=True)


def analyze_completed(inputs, manifest, output, expected_sha, bootstrap=2000):
    output = Path(output)
    rows, ensemble_parts, id_parts, label_parts = [], [], [], []
    raw_parts = {m: [] for m in DISPLAY}
    scaled_parts = {m: [] for m in DISPLAY}
    params = {}
    for fold in range(1, 6):
        groups = {}
        for model in DISPLAY:
            folder = output / f'training/{model}/fold{fold}'
            inspect_job(folder, inputs, expected_sha)
            groups[model] = {split: read_group(folder, f'{model}_fold{fold}_{split}') for split in ('val', 'test')}
        reference = groups[next(iter(DISPLAY))]
        for model in DISPLAY:
            for split in ('val', 'test'):
                if not np.array_equal(groups[model][split]['pids'], reference[split]['pids']) or not np.array_equal(groups[model][split]['labels'], reference[split]['labels']):
                    raise ValueError('Model evaluation rows differ.')
        if set(reference['val']['pids']) & set(reference['test']['pids']):
            raise ValueError('Calibration and test groups overlap within an outer fold.')
        temperatures, weights, diagnostics = fit_calibration([groups[m]['val']['probs'] for m in DISPLAY], reference['val']['labels'])
        scaled = []
        for i, model in enumerate(DISPLAY):
            raw = groups[model]['test']['probs']
            calibrated = temperature(raw, temperatures[i])
            raw_parts[model].append(raw)
            scaled_parts[model].append(calibrated)
            scaled.append(calibrated)
            rows.append({'fold': fold, 'model': model, 'temperature': temperatures[i],
                         'weight': float(weights[i]), 'calibration_clusters': len(reference['val']['pids']),
                         'test_clusters': len(reference['test']['pids'])})
        ensemble_parts.append(sum(w * p for w, p in zip(weights, scaled)))
        id_parts.append(reference['test']['pids'])
        label_parts.append(reference['test']['labels'])
        params[str(fold)] = {'temperatures': dict(zip(DISPLAY, temperatures)),
                            'weights': {m: float(w) for m, w in zip(DISPLAY, weights)},
                            'calibration_clusters': len(reference['val']['pids']),
                            'diagnostics': diagnostics}
    ids, labels = np.concatenate(id_parts), np.concatenate(label_parts)
    if len(ids) != len(set(ids)) or len(ids) != 520:
        raise ValueError('Every one of the 520 groups must have exactly one outer-test prediction.')
    order = np.argsort(ids)
    ids, labels = ids[order], labels[order]
    ensemble = np.concatenate(ensemble_parts)[order]
    raw = {m: np.concatenate(parts)[order] for m, parts in raw_parts.items()}
    scaled = {m: np.concatenate(parts)[order] for m, parts in scaled_parts.items()}
    for model in DISPLAY:
        for kind, value in (('probs', raw[model]), ('labels', labels), ('pids', ids)):
            atomic_array(output / f'predictions/{model}/{model}_test_{kind}.npy', value)
        atomic_array(output / f'predictions/{model}/{model}_test_scaled_probs.npy', scaled[model])
    analysis = output / 'analysis'
    analysis.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(analysis / 'calibration_parameters.csv', index=False)
    report = analyze_predictions(ensemble, raw, scaled, labels, ids, analysis, bootstrap)
    report.update(protocol='A', counts=manifest['counts'], n_cal={'by_fold': {f: v['calibration_clusters'] for f, v in params.items()}},
                  calibration_by_fold=params, input_manifest_sha256=expected_sha,
                  test_used_for_selection_or_calibration=False,
                  note='Revised Protocol A; fixed outer test groups; no global fitting on the pooled outer-test predictions.')
    atomic_json(analysis / 'report.json', report)
    shutil.copy2(Path(inputs) / 'clinical_images.csv', output / 'clinical_images.csv')
    shutil.copy2(Path(inputs) / 'split_summary.csv', output / 'split_summary.csv')
    atomic_json(output / 'publication_context.json', {'protocol': 'A', 'evaluation_unit': 'patient cluster',
                'clinical_images': 755, 'filename_patient_ids': 524, 'known_duplicate_aware_groups': 520,
                'analysis_folder': 'analysis', 'input_manifest_sha256': expected_sha,
                'methods': manifest, 'bootstrap_conditions': 'Fitted models and fold-specific calibration kept fixed.'})
    return report


def bundle_results(output):
    output = Path(output)
    archive = output.parent / 'Protocol_A_755_Results_and_Publication.zip'
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix != '.pt' and '__pycache__' not in path.parts:
                z.write(path, path.relative_to(output.parent).as_posix())
    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Final result ZIP failed integrity checking.')
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/protocol_a_755'))
    parser.add_argument('--expected-manifest-sha', required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--max-hours', type=float, default=10.5)
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.input_root.resolve()):
        raise ValueError('Outputs must be outside the input tree.')
    if not 0 < args.max_hours <= 10.5:
        raise ValueError('Use a positive budget no larger than 10.5 hours.')
    started = time.monotonic()
    inputs, manifest, clinical, paths = preflight(args.input_root, args.expected_manifest_sha)
    print('PREFLIGHT PASS: 755 clinical images, 524 filename IDs, 520 known-duplicate-aware groups; all five splits valid; each group tested once.', flush=True)
    print(pd.read_csv(inputs / 'split_summary.csv').to_string(index=False), flush=True)
    if args.check_only:
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'protocol_a_manifest.json', manifest)
    from protocol_a_train import train_job, identity_for
    restore_completed(args.input_root, args.output, inputs, manifest, args.expected_manifest_sha)
    completed = 0
    for model in manifest['models']:
        for fold in range(1, 6):
            folder = args.output / f'training/{model}/fold{fold}'
            if (folder / 'job_complete.json').exists():
                inspect_job(folder, inputs, args.expected_manifest_sha, identity_for(inputs, manifest, model, fold))
                completed += 1
                continue
            remaining = args.max_hours * 3600 - (time.monotonic() - started)
            if remaining < 1200:
                atomic_json(args.output / 'run_status.json', {'status': 'paused', 'completed_jobs': completed,
                            'required_jobs': 20, 'next_model': model, 'next_fold': fold})
                print('SAFE PAUSE: save full notebook outputs, attach them to this same notebook, and run again.', flush=True)
                return 0
            if shutil.disk_usage(args.output).free < 5 * 1024**3:
                atomic_json(args.output / 'run_status.json', {'status': 'paused_low_disk_space', 'completed_jobs': completed, 'required_jobs': 20})
                print('SAFE PAUSE: less than 5 GiB free. Keep all outputs and resume in a new session.', flush=True)
                return 0
            print(f'JOB {completed + 1}/20: {model}, fold {fold}', flush=True)
            result = train_job(inputs, manifest, paths, model, fold, folder, args.input_root, remaining)
            if not result['complete']:
                atomic_json(args.output / 'run_status.json', {'status': 'paused', 'completed_jobs': completed,
                            'required_jobs': 20, 'active_model': model, 'active_fold': fold,
                            'active_completed_epochs': result['completed_epochs']})
                print('SAFE PAUSE: full checkpoints are in this saved output; resume this same notebook.', flush=True)
                return 0
            completed += 1
    atomic_json(args.output / 'run_status.json', {'status': 'training_complete_analysis_pending', 'completed_jobs': 20, 'required_jobs': 20})
    if args.max_hours * 3600 - (time.monotonic() - started) < 600:
        print('All 20 jobs are saved. Resume once to finish the CPU analysis and publication outputs.', flush=True)
        return 0
    report = analyze_completed(inputs, manifest, args.output, args.expected_manifest_sha)
    from publication_outputs import generate_publication
    generate_publication(args.output)
    atomic_json(args.output / 'run_status.json', {'status': 'complete', 'completed_jobs': 20, 'required_jobs': 20,
                                                'test_groups': 520, 'accuracy': report['ens_fixed']['acc']})
    archive = bundle_results(args.output)
    print('COMPLETE: Protocol A revised 755-image run; all 20 jobs, ensemble and publication outputs finished.', flush=True)
    print(json.dumps(report['ens_fixed'], indent=2), flush=True)
    print(f'Download: {archive}', flush=True)
    print('Keep the full saved notebook version separately for all selected model weights.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
