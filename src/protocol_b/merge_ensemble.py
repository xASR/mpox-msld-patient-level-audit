"""Protocol B, step 2: combine the four backbones (paper Section 3.6 and 3.7).

Reads the 24 arrays written by train_worker_v34.py (6 per backbone), removes
calibration patients that are test patients of the same backbone in another
fold, joins the four backbones on patient ID, fits one temperature per backbone
and the ensemble weights on the calibration patients (L-BFGS-B), and only then
scores the 186 shared test patients. It also re-runs the merge without the
overlap removal ("leaky") and writes bootstrap intervals, McNemar tests,
per-class results, the confusion matrix and ROC points.

Usage:
    python merge_ensemble.py results/predictions results/protocol_b_merge

The prediction folder is searched recursively for files named
<model>_{val,test}_{probs,labels,pids}.npy, so any sub-folder layout works.
pids arrays are object arrays of strings and need allow_pickle; it is enabled
for those files only.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binomtest
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score, log_loss,
                             precision_score, recall_score, confusion_matrix,
                             classification_report, roc_curve)

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
C = 6
NAMES = ['Healthy', 'Cowpox', 'Mpox', 'Measles', 'Chickenpox', 'HFMD']
MODELS = {"efficientnetv2_s": "EfficientNetV2-S", "convnext_tiny": "ConvNeXt-Tiny",
          "swin_small": "Swin-Small", "beit_base": "BEiT-Base"}


def find_array(model, split, kind):
    """Locate <model>_<split>_<kind>.npy anywhere below DATA."""
    target = f"{model}_{split}_{kind}.npy"
    for root, _, files in os.walk(DATA):
        if target in files:
            return os.path.join(root, target)
    raise FileNotFoundError(f"{target} not found under {DATA}")

def norm(p):
    p = p / (p.sum(1, keepdims=True) + 1e-8)
    return p / p.sum(1, keepdims=True)

def softmax(x):
    x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / (e.sum(-1, keepdims=True) + 1e-8)

def ece(p, y, n_bins=10):
    b = np.linspace(0, 1, n_bins + 1); conf = p.max(1); acc = p.argmax(1) == y; e = 0.0
    for i in range(n_bins):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.mean() > 0: e += abs(conf[m].mean() - acc[m].mean()) * m.mean()
    return e

def metrics(p, y):
    pr = p.argmax(1)
    return dict(acc=accuracy_score(y, pr), prec=precision_score(y, pr, average='macro', zero_division=0),
                rec=recall_score(y, pr, average='macro', zero_division=0), f1=f1_score(y, pr, average='macro'),
                auc=roc_auc_score(y, p, multi_class='ovr', labels=list(range(C))),
                ece=ece(p, y), ll=log_loss(y, p, labels=list(range(C))))

def tune_T(p, y):
    def obj(t):
        s = norm(np.clip(p, 1e-8, 1) ** (1.0 / t[0]))
        return log_loss(y, s, labels=list(range(C)))
    r = minimize(obj, [1.0], bounds=[(0.1, 10.0)], method='L-BFGS-B')
    return r.x[0], r.success

def opt_w(plist, y):
    n = len(plist)
    def obj(z):
        w = softmax(z); return log_loss(y, np.sum([w[i] * plist[i] for i in range(n)], 0), labels=list(range(C)))
    r = minimize(obj, np.ones(n), method='L-BFGS-B')
    return softmax(r.x), r.success

raw = {}
for m, d in MODELS.items():
    arr = {}
    for split in ['val', 'test']:
        for k in ['probs', 'labels', 'pids']:
            arr[f'{split}_{k}'] = np.load(find_array(m, split, k), allow_pickle=(k == 'pids'))
    raw[m] = arr

report = {'shapes': {m: {k: list(v.shape) for k, v in a.items()} for m, a in raw.items()}}
report['nan'] = {m: bool(np.isnan(a['test_probs']).any() or np.isnan(a['val_probs']).any()) for m, a in raw.items()}

def run(drop_overlap):
    vd, td, info = {}, {}, {}
    for m, a in raw.items():
        v = pd.DataFrame({'pid': a['val_pids'], 'label': a['val_labels'].astype(int), 'probs': list(norm(a['val_probs']))})
        t = pd.DataFrame({'pid': a['test_pids'], 'label': a['test_labels'].astype(int), 'probs': list(norm(a['test_probs']))})
        ov = set(v.pid) & set(t.pid)
        info[m] = dict(val=len(v), test=len(t), overlap=len(ov))
        if drop_overlap: v = v[~v.pid.isin(set(t.pid))].reset_index(drop=True)
        vd[m], td[m] = v, t
    sv = set.intersection(*[set(vd[m].pid) for m in MODELS]); st = set.intersection(*[set(td[m].pid) for m in MODELS])
    V, T, ref = [], [], {}
    for m in MODELS:
        v = vd[m][vd[m].pid.isin(sv)].sort_values('pid').reset_index(drop=True)
        t = td[m][td[m].pid.isin(st)].sort_values('pid').reset_index(drop=True)
        if not ref: ref = dict(vy=v.label.values, ty=t.label.values, vp=v.pid.values, tp=t.pid.values)
        assert np.array_equal(v.pid.values, ref['vp']) and np.array_equal(t.pid.values, ref['tp'])
        assert np.array_equal(v.label.values, ref['vy']) and np.array_equal(t.label.values, ref['ty'])
        V.append(norm(np.stack(v.probs.values))); T.append(norm(np.stack(t.probs.values)))
    temps, cV, cT = [], [], []
    for p, q in zip(V, T):
        Tm, ok = tune_T(p, ref['vy']); temps.append((Tm, ok))
        cV.append(norm(np.clip(p, 1e-8, 1) ** (1 / Tm))); cT.append(norm(np.clip(q, 1e-8, 1) ** (1 / Tm)))
    w, okw = opt_w(cV, ref['vy'])
    ens = np.sum([w[i] * cT[i] for i in range(len(w))], 0)
    return dict(info=info, n_cal=len(sv), n_test=len(st), temps=temps, w=w, okw=okw, ens=ens,
                singles=T, singles_cal=cT, y=ref['ty'], pids=ref['tp'], valy=ref['vy'])

fixed, leaky = run(True), run(False)
y = fixed['y']
report['overlap_info'] = fixed['info']
report['n_cal'] = {'fixed': fixed['n_cal'], 'leaky': leaky['n_cal']}
report['n_test'] = fixed['n_test']
report['temps'] = {m: [float(t), bool(ok)] for m, (t, ok) in zip(MODELS, fixed['temps'])}
report['temps_leaky'] = {m: float(t) for m, (t, ok) in zip(MODELS, leaky['temps'])}
report['weights'] = {m: float(w) for m, w in zip(MODELS, fixed['w'])}
report['weights_leaky'] = {m: float(w) for m, w in zip(MODELS, leaky['w'])}
report['ens_fixed'] = metrics(fixed['ens'], y)
report['ens_leaky'] = metrics(leaky['ens'], y)
report['singles_raw'] = {m: metrics(p, y) for m, p in zip(MODELS, fixed['singles'])}
report['singles_tempscaled'] = {m: metrics(p, y) for m, p in zip(MODELS, fixed['singles_cal'])}
report['correct'] = {m: int((p.argmax(1) == y).sum()) for m, p in zip(MODELS, fixed['singles'])}
report['correct']['ensemble'] = int((fixed['ens'].argmax(1) == y).sum())
report['class_support'] = {NAMES[c]: int((y == c).sum()) for c in range(C)}

# per-class report and confusion matrix
pred = fixed['ens'].argmax(1)
cr = classification_report(y, pred, labels=list(range(C)), target_names=NAMES, digits=4, output_dict=True, zero_division=0)
report['per_class'] = {k: cr[k] for k in NAMES}
report['per_class_auc'] = {NAMES[c]: float(roc_auc_score((y == c).astype(int), fixed['ens'][:, c])) for c in range(C)}
cm = confusion_matrix(y, pred, labels=list(range(C)))
report['confusion'] = cm.tolist()

# McNemar exact: ensemble vs each single backbone
ce = pred == y
report['mcnemar'] = {}
for m, p in zip(MODELS, fixed['singles']):
    cs = p.argmax(1) == y
    b = int(((ce) & (~cs)).sum()); c_ = int(((~ce) & (cs)).sum())
    pval = binomtest(min(b, c_), b + c_, 0.5).pvalue if b + c_ > 0 else 1.0
    report['mcnemar'][m] = dict(ens_only=b, single_only=c_, p=float(pval))

# patient bootstrap CIs
rng = np.random.default_rng(42); B = 2000; N = len(y)
keys = ['acc', 'f1', 'auc', 'ece']
boot = {k: [] for k in keys}; dauc = {m: [] for m in MODELS}; df1 = {m: [] for m in MODELS}
for _ in range(B):
    idx = rng.integers(0, N, N)
    if len(np.unique(y[idx])) < C: continue
    mm = metrics(fixed['ens'][idx], y[idx])
    for k in keys: boot[k].append(mm[k])
    for m, p in zip(MODELS, fixed['singles']):
        dauc[m].append(mm['auc'] - roc_auc_score(y[idx], p[idx], multi_class='ovr', labels=list(range(C))))
        df1[m].append(mm['f1'] - f1_score(y[idx], p[idx].argmax(1), average='macro'))
report['boot_n'] = len(boot['acc'])
report['boot_ci'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
report['boot_dauc'] = {m: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in dauc.items()}
report['boot_df1'] = {m: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in df1.items()}

# reliability bins (ensemble, plus the best single raw for comparison)
def rel(p):
    b = np.linspace(0, 1, 11); conf = p.max(1); acc = p.argmax(1) == y; rows = []
    for i in range(10):
        msk = (conf > b[i]) & (conf <= b[i + 1])
        if msk.sum(): rows.append(dict(lo=float(b[i]), hi=float(b[i + 1]), n=int(msk.sum()),
                                       conf=float(conf[msk].mean()), acc=float(acc[msk].mean())))
    return rows
report['reliability_ens'] = rel(fixed['ens'])
report['reliability_convnext_raw'] = rel(fixed['singles'][1])

# ROC points per class (ensemble) for plotting
roc = {}
for c in range(C):
    fpr, tpr, _ = roc_curve((y == c).astype(int), fixed['ens'][:, c])
    roc[NAMES[c]] = dict(fpr=fpr.tolist(), tpr=tpr.tolist())
json.dump(roc, open(os.path.join(OUT, 'roc_points.json'), 'w'))
np.save(os.path.join(OUT, 'ens_probs.npy'), fixed['ens']); np.save(os.path.join(OUT, 'labels.npy'), y)
pd.DataFrame(cm, index=NAMES, columns=NAMES).to_csv(os.path.join(OUT, 'confusion_matrix.csv'))
json.dump(report, open(os.path.join(OUT, 'report.json'), 'w'), indent=1, default=float)
print(json.dumps({k: report[k] for k in ['n_cal', 'n_test', 'temps', 'weights', 'weights_leaky', 'ens_fixed', 'ens_leaky', 'correct', 'overlap_info', 'class_support', 'mcnemar', 'boot_ci', 'boot_n', 'boot_dauc', 'boot_df1', 'per_class_auc', 'confusion', 'temps_leaky']}, indent=1, default=float))
for k in NAMES: print(k, {kk: round(v, 4) for kk, v in report['per_class'][k].items()})
print('raw singles', {m: {k: round(v, 4) for k, v in d.items()} for m, d in report['singles_raw'].items()})
print('T-scaled singles', {m: {k: round(v, 4) for k, v in d.items()} for m, d in report['singles_tempscaled'].items()})
