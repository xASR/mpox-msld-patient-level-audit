"""Metrics and fixed-model patient bootstrap for the corrected evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binomtest, norm
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, log_loss, confusion_matrix,
                             classification_report, roc_curve)
from completion_core import atomic_json, atomic_array

NAMES = ['Healthy', 'Cowpox', 'Mpox', 'Measles', 'Chickenpox', 'HFMD']
DISPLAY = {'efficientnetv2_s': 'EfficientNetV2-S', 'convnext_tiny': 'ConvNeXt-Tiny',
           'swin_small': 'Swin-Small', 'beit_base': 'BEiT-Base'}


def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    return p / p.sum(axis=1, keepdims=True)


def temperature(p, value):
    logits = np.log(np.clip(normalize(p), 1e-8, 1)) / float(value)
    logits -= logits.max(axis=1, keepdims=True)
    return normalize(np.exp(logits))


def weights(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def fit_calibration(calibration_probs, labels):
    scaled, temperatures, diagnostics = [], [], []
    for p in calibration_probs:
        result = minimize(lambda t: log_loss(labels, temperature(p, t[0]), labels=list(range(6))),
                          [1.0], method='L-BFGS-B', bounds=[(.1, 5.0)],
                          options={'maxiter': 1000, 'ftol': 1e-12})
        if not result.success or not np.isfinite(result.fun):
            raise ValueError(f'Temperature fit failed: {result.message}')
        temperatures.append(float(result.x[0]))
        scaled.append(temperature(p, result.x[0]))
        diagnostics.append(str(result.message))
    result = minimize(lambda z: log_loss(labels, sum(w * p for w, p in zip(weights(z), scaled)), labels=list(range(6))),
                      np.zeros(len(scaled)), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f'Ensemble weight fit failed: {result.message}')
    return temperatures, weights(result.x), {'temperature_messages': diagnostics,
                                             'weight_message': str(result.message),
                                             'calibration_log_loss': float(result.fun)}


def ece(p, y):
    confidence, correct = p.max(axis=1), p.argmax(axis=1) == y
    value = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            value += abs(confidence[mask].mean() - correct[mask].mean()) * mask.mean()
    return float(value)


def metrics(p, y):
    pred = p.argmax(axis=1)
    return {'acc': float(accuracy_score(y, pred)),
            'prec': float(precision_score(y, pred, average='macro', zero_division=0)),
            'rec': float(recall_score(y, pred, average='macro', zero_division=0)),
            'f1': float(f1_score(y, pred, average='macro')),
            'auc': float(roc_auc_score(y, p, multi_class='ovr', labels=list(range(6)))),
            'ece': ece(p, y), 'll': float(log_loss(y, p, labels=list(range(6))))}


def wilson(correct, n):
    z = norm.ppf(.975)
    p = correct / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [float(center - half), float(center + half)]


def reliability(p, y):
    rows = []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p.max(axis=1) > lo) & (p.max(axis=1) <= hi)
        if mask.any():
            rows.append({'lo': float(lo), 'hi': float(hi), 'n': int(mask.sum()),
                         'conf': float(p[mask].max(axis=1).mean()),
                         'acc': float((p[mask].argmax(axis=1) == y[mask]).mean())})
    return rows


def analyze_predictions(ensemble, raw, scaled, y, ids, output, bootstrap=2000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    n = len(y)
    if len(set(ids)) != n or set(y) != set(range(6)):
        raise ValueError('Need unique evaluation groups and all six classes.')
    report = {'n_test': n, 'evaluation_unit': 'known-duplicate-aware patient cluster',
              'ens_fixed': metrics(ensemble, y),
              'singles_raw': {m: metrics(raw[m], y) for m in DISPLAY},
              'singles_tempscaled': {m: metrics(scaled[m], y) for m in DISPLAY},
              'correct': {m: int((raw[m].argmax(axis=1) == y).sum()) for m in DISPLAY}}
    report['correct']['ensemble'] = int((ensemble.argmax(axis=1) == y).sum())
    report['accuracy_wilson_95_ci'] = wilson(report['correct']['ensemble'], n)
    report['class_support'] = {name: int((y == c).sum()) for c, name in enumerate(NAMES)}
    classification = classification_report(y, ensemble.argmax(axis=1), labels=list(range(6)),
                                           target_names=NAMES, output_dict=True, zero_division=0)
    report['per_class'] = {name: classification[name] for name in NAMES}
    report['per_class_auc'] = {name: float(roc_auc_score(y == c, ensemble[:, c])) for c, name in enumerate(NAMES)}
    matrix = confusion_matrix(y, ensemble.argmax(axis=1), labels=list(range(6)))
    report['confusion'] = matrix.tolist()
    report['mcnemar'] = {}
    for model in DISPLAY:
        a, b = ensemble.argmax(axis=1) == y, raw[model].argmax(axis=1) == y
        ens_only, single_only = int((a & ~b).sum()), int((~a & b).sum())
        p = binomtest(min(ens_only, single_only), ens_only + single_only, .5).pvalue if ens_only + single_only else 1.0
        report['mcnemar'][model] = {'ens_only': ens_only, 'single_only': single_only, 'p': float(p)}
    rng = np.random.default_rng(42)
    samples = {key: [] for key in ('acc', 'f1', 'auc', 'ece')}
    dauc, df1 = {m: [] for m in DISPLAY}, {m: [] for m in DISPLAY}
    for i in range(bootstrap):
        ix = rng.integers(0, n, n)
        if len(np.unique(y[ix])) < 6:
            continue
        mm = metrics(ensemble[ix], y[ix])
        for key in samples:
            samples[key].append(mm[key])
        for model in DISPLAY:
            dauc[model].append(mm['auc'] - roc_auc_score(y[ix], raw[model][ix], multi_class='ovr', labels=list(range(6))))
            df1[model].append(mm['f1'] - f1_score(y[ix], raw[model][ix].argmax(axis=1), average='macro'))
        if (i + 1) % 500 == 0:
            print(f'Bootstrap resamples: {i + 1}/{bootstrap}', flush=True)
    report['boot_n'] = len(samples['acc'])
    report['boot_ci'] = {key: np.percentile(value, [2.5, 97.5]).tolist() for key, value in samples.items()}
    report['boot_dauc'] = {m: [float(np.mean(v)), *np.percentile(v, [2.5, 97.5]).tolist()] for m, v in dauc.items()}
    report['boot_df1'] = {m: [float(np.mean(v)), *np.percentile(v, [2.5, 97.5]).tolist()] for m, v in df1.items()}
    report['bootstrap_scope'] = 'Patient clusters resampled; trained models and fold-specific calibration parameters kept fixed.'
    report['reliability_ens'] = reliability(ensemble, y)
    report['reliability_convnext_raw'] = reliability(raw['convnext_tiny'], y)
    roc = {}
    for c, name in enumerate(NAMES):
        fpr, tpr, _ = roc_curve(y == c, ensemble[:, c])
        roc[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
    atomic_json(output / 'roc_points.json', roc)
    atomic_json(output / 'report.json', report)
    for name, array in (('ens_probs', ensemble), ('labels', y), ('patient_ids', np.asarray(ids).astype(str))):
        atomic_array(output / f'{name}.npy', array)
    pd.DataFrame(matrix, index=NAMES, columns=NAMES).to_csv(output / 'confusion_matrix.csv')
    table = pd.DataFrame({'patient_id': ids, 'label': y, 'prediction': ensemble.argmax(axis=1),
                          'confidence': ensemble.max(axis=1), 'correct': ensemble.argmax(axis=1) == y})
    for c in range(6):
        table[f'probability_class_{c}'] = ensemble[:, c]
    table.to_csv(output / 'per_patient_predictions.csv', index=False)
    np.savez_compressed(output / 'bootstrap_distributions.npz',
                        **{key: np.asarray(v) for key, v in samples.items()},
                        **{f'dauc_{m}': np.asarray(v) for m, v in dauc.items()},
                        **{f'df1_{m}': np.asarray(v) for m, v in df1.items()})
    return report
