# ==============================================================================
# MSLD v2.0 - Leakage ablation, seeded repeats (Code 9, pipelines 1A and 1B)
#
# Same setup as Code 9: ViT-Tiny (ImageNet-pretrained) at 128 px, offline
# "MATLAB-style" augmentation (original + 9 copies), Adam lr 1e-3, CE loss,
# batch 32, 100 epochs, no scheduler.
#   1A  augment, then split   : 7,550 images -> random 80:20 split by image
#   1B  split, then augment   : patient-grouped fold held out first, only the
#                               training images are expanded ten-fold
# What is new: every run is seeded (split, augmentation, weights, batch order),
# each seed runs 1A and 1B back to back, results are saved after every run,
# and a restarted session skips runs that are already in the CSV.
#
# Kaggle: attach the MSLD v2.0 dataset, GPU on, internet on (to download the
# ViT-Tiny weights), then "Save Version" (Save & Run All). About 13 min per run
# on a T4 (measured), so 5 seeds x 2 pipelines take a little over 2 hours.
# Output: /kaggle/working/ablation_runs.csv and ablation_summary.txt
# ==============================================================================
import os, re, glob, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats

SEEDS     = [0, 1, 2, 3, 4]      # cut to [0, 1, 2] if you are short on time
EPOCHS    = 100
BATCH     = 32
LR        = 1e-3
MULT      = 10                   # original + 9 augmented copies
TEST_FRAC = 0.20
ROOT      = "/kaggle/input"
OUT_DIR   = "/kaggle/working"
RUNS_CSV  = os.path.join(OUT_DIR, "ablation_runs.csv")
MAX_HOURS = 11.3                 # stay inside Kaggle's 12-hour limit
START     = time.time()
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_MAP = {'CHICKENPOX': 0, 'COWPOX': 1, 'HFMD': 2, 'HEALTHY': 3, 'MEASLES': 4, 'MONKEYPOX': 5}


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- data (as Code 9)
def load_originals():
    files = [p for p in glob.glob(f"{ROOT}/**/*", recursive=True)
             if p.lower().endswith((".jpg", ".jpeg", ".png"))
             and not re.search(r"AUGMENTED|FOLDS_AUG|MASK", p, re.I)]
    rows, seen = [], set()
    for p in sorted(files):                       # sorted: same order on every machine
        folder = os.path.basename(os.path.dirname(p)).upper()
        name = os.path.basename(p)
        if folder not in CLASS_MAP or f"{folder}_{name}" in seen:
            continue
        seen.add(f"{folder}_{name}")
        parts = name.replace('-', '_').split('_')
        rows.append({'filepath': p, 'label': CLASS_MAP[folder],
                     'pid': f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else name})
    df = pd.DataFrame(rows).reset_index(drop=True)
    df['img_id'] = np.arange(len(df))
    return df


AUG = T.Compose([
    T.RandomRotation(degrees=30),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
    T.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.15, hue=0.1),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])
BASE = T.Compose([T.Resize((128, 128)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])


def expand(df, mult, seed):
    """Original + (mult-1) augmented copies per image. Returns tensors and the
    source image id / patient id of every row (used for the overlap check)."""
    torch.manual_seed(seed); random.seed(seed)
    xs, ys, src, pids = [], [], [], []
    for r in df.itertuples():
        with Image.open(r.filepath) as im:
            img = im.convert('RGB')
            xs.append(BASE(img)); ys.append(r.label); src.append(r.img_id); pids.append(r.pid)
            for _ in range(mult - 1):
                xs.append(AUG(img)); ys.append(r.label); src.append(r.img_id); pids.append(r.pid)
    return torch.stack(xs), torch.tensor(ys), np.array(src), np.array(pids)


# ---------------------------------------------------------------- training
def train_eval(Xtr, ytr, Xte, yte, seed):
    seed_all(seed)
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True,
                              num_classes=6, img_size=128).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, generator=g)
    for ep in range(EPOCHS):
        model.train()
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 256):
            preds.append(model(Xte[i:i + 256].to(DEVICE)).argmax(1).cpu())
    preds = torch.cat(preds).numpy(); yt = yte.numpy()
    del model, opt; torch.cuda.empty_cache()
    return int((preds == yt).sum()), accuracy_score(yt, preds), f1_score(yt, preds, average='macro')


def run_1A(df, seed):
    X, y, src, pid = expand(df, MULT, seed)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_FRAC, random_state=seed, stratify=y.numpy())
    # leakage evidence: test rows whose source photo / patient also has rows in training
    src_overlap = float(np.isin(src[te], src[tr]).mean())
    pid_overlap = float(np.isin(pid[te], pid[tr]).mean())
    c, acc, f1 = train_eval(X[tr], y[tr], X[te], y[te], seed)
    return dict(n_train=len(tr), n_test=len(te), correct=c, acc=acc, macro_f1=f1,
                test_photo_in_train=src_overlap, test_patient_in_train=pid_overlap)


def run_1B(df, seed):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr_i, te_i = next(sgkf.split(df, df['label'], df['pid']))
    dtr, dte = df.iloc[tr_i], df.iloc[te_i]
    assert not set(dtr['pid']) & set(dte['pid']), "patient overlap in 1B"
    Xtr, ytr, _, _ = expand(dtr, MULT, seed)
    Xte, yte, _, _ = expand(dte, 1, seed)
    c, acc, f1 = train_eval(Xtr, ytr, Xte, yte, seed)
    return dict(n_train=len(ytr), n_test=len(yte), correct=c, acc=acc, macro_f1=f1,
                test_photo_in_train=0.0, test_patient_in_train=0.0)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    df = load_originals()
    print(f"Loaded {len(df)} original images, {df['pid'].nunique()} patients. Device: {DEVICE}")
    assert len(df) == 755, "expected 755 original images; check the dataset path"

    done = pd.read_csv(RUNS_CSV) if os.path.exists(RUNS_CSV) else pd.DataFrame()
    last_minutes = 0.0
    for seed in SEEDS:
        for name, fn in [("1A_augment_then_split", run_1A), ("1B_split_then_augment", run_1B)]:
            if len(done) and ((done['pipeline'] == name) & (done['seed'] == seed)).any():
                print(f"skip {name} seed {seed} (already in CSV)"); continue
            hours_left = MAX_HOURS - (time.time() - START) / 3600
            if hours_left * 60 < last_minutes * 1.15:
                print(f"Stopping: {hours_left:.2f} h left, last run took {last_minutes:.1f} min"); break
            t0 = time.time()
            res = fn(df, seed)
            last_minutes = (time.time() - t0) / 60
            row = dict(pipeline=name, seed=seed, minutes=round(last_minutes, 1), **res)
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(RUNS_CSV, index=False)          # saved after every run
            print(f"{name} seed {seed}: acc {res['acc']*100:.2f}% "
                  f"({res['correct']}/{res['n_test']}), macro F1 {res['macro_f1']:.4f}, "
                  f"{last_minutes:.1f} min")
        else:
            continue
        break

    # ------------------------------------------------------------ summary
    lines = []
    for name, g in done.groupby('pipeline'):
        a = g['acc'].values * 100
        n = len(a); m = a.mean(); sd = a.std(ddof=1) if n > 1 else float('nan')
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float('nan')
        lines.append(f"{name}: n={n} runs, accuracy {m:.2f}% +/- {sd:.2f} (95% CI {m-half:.2f} to {m+half:.2f}); "
                     f"runs: {', '.join(f'{v:.2f}' for v in a)}")
        if name.startswith("1A"):
            lines.append(f"   test images whose photo has copies in training: {g['test_photo_in_train'].mean()*100:.1f}%; "
                         f"whose patient is in training: {g['test_patient_in_train'].mean()*100:.1f}%")
    piv = done.pivot(index='seed', columns='pipeline', values='acc').dropna()
    if len(piv) >= 2:
        gap = (piv['1A_augment_then_split'] - piv['1B_split_then_augment']).values * 100
        n = len(gap); half = stats.t.ppf(0.975, n - 1) * gap.std(ddof=1) / np.sqrt(n)
        lines.append(f"Paired gap 1A - 1B over {n} seeds: {gap.mean():.2f} points "
                     f"(95% CI {gap.mean()-half:.2f} to {gap.mean()+half:.2f})")
    summary = "\n".join(lines)
    print("\n" + "=" * 70 + "\n" + summary + "\n" + "=" * 70)
    with open(os.path.join(OUT_DIR, "ablation_summary.txt"), "w") as f:
        f.write(summary + "\n")
