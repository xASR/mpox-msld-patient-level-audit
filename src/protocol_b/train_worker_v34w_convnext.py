# ==============================================================================
# V34-W (2026-11): the V34 worker below, unchanged in training, data, seeds and
# checkpoint selection. Lines marked [V34-W] only (1) select ConvNeXt-Tiny,
# (2) refuse to run unless the inputs match the original run, (3) record the
# epoch of each kept snapshot, and (4) save the two kept snapshots' weights, the
# fold assignment and run metadata. None of these consume random numbers.
# ==============================================================================
# MSLD v2.0 - INDIVIDUAL MODEL WORKER SCRIPT (V34 - Kaggle Final)
# Target: Train a single model and explicitly save its .npy arrays.
# Environment: Kaggle (P100 or T4x2 GPU — Single GPU utilised)
#
# Audit history:
#   V27 -> V28: Fixed Bugs 1-7 (Leakage, DataLoaders, Optimizers, Alignment)
#   V28 -> V29: Fixed early-exit DataLoader leak; updated regex anchor.
#   V29 -> V30: num_workers=0 for Colab /dev/shm deadlock; per-fold checkpoints.
#   V30 -> V31: Atomic saves; Drive mirror (Colab); dead code removal.
#   V31 Kaggle: Migrated to Kaggle. Restored num_workers=2 + pin_memory=True.
#   V32 Kaggle: Fixed 7 bugs (timm API, datalog guards, disjoint assert, seed
#               rotation, NaN/Inf guard, sheet-name fallback, dead param).
#   V33 Kaggle: ROOT CAUSE FIX — "0 images loaded" (directory-based labels,
#               auto-path discovery, mask exclusion, basename clean_name,
#               recursive datalog search, diagnostic output).
#   V34 Kaggle: Final pre-commit audit fixes:
#     Bug 1: str|None return annotation → Optional[str] (Python 3.9 compat).
#     Bug 2: _label_from_directory greedy prefix match guarded with .isalpha()
#            to prevent dataset slug 'MPOX-SKIN-LESION-...' matching MPOX prefix
#            and misclassifying non-Monkeypox images as label 2.
#     Bug 3: fold_data['train'] hard-coded key → case-insensitive column search,
#            consistent with val_col/test_col handling.
#     Bug 4: optimizer, scheduler, scaler, loss functions explicitly deleted at
#            both fold-end paths to prevent CUDA VRAM retention between folds.
# ==============================================================================

import os, random, gc, glob, re, time
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import t
from typing import Optional

# ⏱️ GLOBAL TIME GUARD (11.5 h — safely inside Kaggle's 12-hour limit)
START_TIME  = time.time()
MAX_RUNTIME = 11.5 * 60 * 60

def time_exceeded():
    return (time.time() - START_TIME) > MAX_RUNTIME

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"]          = "42"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
import torchvision.transforms as T

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, log_loss,
    precision_score, recall_score, classification_report,
)

# ===================== CONFIG =====================
class CFG:
    SEED                = 42
    IMG_SIZE            = 384
    BATCH_SIZE          = 16
    EPOCHS              = 25
    FREEZE_EPOCHS       = 3
    MIXUP_DISABLE_EPOCH = 10
    NUM_CLASSES         = 6
    CLASS_NAMES         = ['Healthy', 'Cowpox', 'Monkeypox', 'Measles', 'Chickenpox', 'HFMD']
    CLASS_MAP = {
        'HEALTHY': 0, 'COWPOX': 1, 'MONKEYPOX': 2,
        'MEASLES': 3, 'CHICKENPOX': 4, 'HFMD': 5
    }
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 🔴 KEEP ONLY ONE MODEL UNCOMMENTED PER KAGGLE NOTEBOOK
    MODELS = {
        #"efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
        "convnext_tiny":    "convnext_tiny.in12k_ft_in1k_384",   # [V34-W] ConvNeXt-Tiny only
        #"swin_small":       "swin_small_patch4_window7_224",
        # "beit_base":        "beit_base_patch16_384",
    }

    TRANSFORMER_FAMILIES = ['swin', 'vit', 'deit', 'coat', 'maxvit', 'pvt',
                            'beit', 'xcit', 'twins', 'cait', 'levit', 'pit']

    SAVE_DIR    = "/kaggle/working/outputs"
    NUM_WORKERS = 2


# ==============================================================================
# LABEL DETECTION — DIRECTORY BASED (PRIMARY) + FILENAME (FALLBACK)
# ==============================================================================
_DIR_EXACT_LABEL = {
    'MONKEYPOX': 2,  'MPOX': 2,
    'CHICKENPOX': 4, 'CHICKEN POX': 4, 'CHICKEN_POX': 4,
    'MEASLES': 3,
    'COWPOX': 1,
    'HFMD': 5,
    'HEALTHY': 0,    'NORMAL': 0,
}

_DIR_PREFIX_LABEL = [
    ('MONKEY', 2), ('MPOX', 2),
    ('CHICKEN', 4),
    ('MEASLE', 3),
    ('COWPOX', 1),
    ('HFMD', 5),
    ('HEALTHY', 0), ('NORMAL', 0),
]

def _label_from_directory(img_path: str) -> Optional[int]:
    """
    Walk parent directory components (closest to file first) and return the
    label index from the first component that matches a known class name.

    Bug 2 Fix: Prefix matching is guarded with .isalpha(). A valid class
    directory name ('MONKEYPOX', 'CHICKENPOX', ...) is pure alphabetic.
    The Kaggle dataset slug ('MPOX-SKIN-LESION-DATASET-VERSION-20-MSLD-V20')
    contains hyphens and digits — .isalpha() returns False — so it can never
    trigger a false prefix match even if the iteration reaches the root.
    """
    parts = re.split(r'[\\/]', img_path.upper())
    for part in reversed(parts[:-1]):   # skip the filename itself
        part = part.strip()
        if part in _DIR_EXACT_LABEL:
            return _DIR_EXACT_LABEL[part]
        # Only apply prefix matching to pure-alpha parts (class dir names).
        # This prevents 'MPOX-SKIN-LESION-...' from matching the 'MPOX' prefix.
        if part.isalpha():
            for prefix, lbl in _DIR_PREFIX_LABEL:
                if part.startswith(prefix):
                    return lbl
    return None

def _label_from_filename(name_upper: str) -> Optional[int]:
    """Filename-prefix fallback. Used only when directory detection fails."""
    if name_upper.startswith(('MKP', 'MPOX', 'MONKEYPOX')):    return 2
    if name_upper.startswith(('CHP', 'CHICKENPOX', 'CHICKEN')): return 4
    if name_upper.startswith(('MSL', 'MEASLES', 'MEASLE')):     return 3
    if name_upper.startswith(('CWP', 'COWPOX')):                return 1
    if name_upper.startswith('HFMD') or 'HAND' in name_upper:   return 5
    if name_upper.startswith(('HEALTHY', 'NORMAL')):             return 0
    return None


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_full_determinism():
    torch.backends.cudnn.deterministic    = True
    torch.backends.cudnn.benchmark        = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False
    try: torch.use_deterministic_algorithms(True)
    except Exception: pass

def cuda_sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

seed_all(CFG.SEED)
set_full_determinism()

def get_generator():
    g = torch.Generator()
    g.manual_seed(CFG.SEED)
    return g


# ==============================================================================
# ATOMIC SAVE
# ==============================================================================
def atomic_save(arr: np.ndarray, path: str) -> None:
    tmp = path + ".tmp.npy"   # <--- THIS STOPS NUMPY FROM RENAMING IT
    np.save(tmp, arr)
    os.replace(tmp, path)

def save_fold_checkpoint(display_name: str, fold_num: int, checkpoint: dict) -> None:
    os.makedirs(CFG.SAVE_DIR, exist_ok=True)  # <--- ADD THIS LINE HERE
    for key, arr in checkpoint.items():
        local_path = os.path.join(CFG.SAVE_DIR,
                                  f"{display_name}_fold{fold_num}_{key}.npy")
        atomic_save(arr, local_path)
    print(f"  💾 Fold {fold_num} checkpoint atomically stored → {CFG.SAVE_DIR}")

# ==============================================================================
# DATASET
# ==============================================================================
class MpoxDataset(Dataset):
    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(row.filepath) as img:
            tensor_img = self.transform(img.convert('RGB'))
        return tensor_img, int(row.label), str(row.patient_id)

def get_transforms():
    train_tf = T.Compose([
        T.RandomResizedCrop(CFG.IMG_SIZE, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.GaussianBlur(3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = T.Compose([
        T.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

def get_patient_id(filename: str) -> str:
    name  = os.path.splitext(filename)[0]
    name  = re.sub(r'[_\s-]*(aug(mented)?|mask).*$', '', name, flags=re.IGNORECASE)
    match = re.search(r'([A-Za-z]+)[-_\s]*\(?(\d+)', name)
    return f"{match.group(1).upper()}_{match.group(2)}" if match else name.upper()


# ==============================================================================
# AUTO-DISCOVER DATASET ROOT
# Bug 1 Fix: Return annotation changed from str|None (Python 3.10+) to
# Optional[str] (Python 3.9 compatible).
# ==============================================================================
def find_dataset_root() -> Optional[str]:
    """
    Scan /kaggle/input/ for the MSLD v2.0 dataset directory.
    Returns the first candidate path that contains at least one image file.
    """
    kaggle_input = "/kaggle/input"
    candidates   = []

    if os.path.isdir(kaggle_input):
        for name in sorted(os.listdir(kaggle_input)):
            if any(kw in name.lower()
                   for kw in ('mpox', 'msld', 'monkeypox', 'skin-lesion')):
                candidates.append(os.path.join(kaggle_input, name))

    # Hardcoded fallbacks for known Kaggle dataset slugs
    candidates += [
        "/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20",
        "/kaggle/input/mpox-skin-lesion-dataset-version20",
        "/kaggle/input/msld-v20",
    ]

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    seen       = set()
    for base in candidates:
        if base in seen or not os.path.isdir(base):
            continue
        seen.add(base)
        # Use iglob + next() for early termination — avoids scanning 50K+ files
        first_image = next(
            (f for f in glob.iglob(os.path.join(base, "**", "*"), recursive=True)
             if os.path.isfile(f) and os.path.splitext(f)[1].lower() in valid_exts),
            None
        )
        if first_image is not None:
            print(f"  ✅ Dataset root found: {base}")
            return base
        else:
            print(f"  ⚠️  Candidate exists but contains no images: {base}")

    return None


# ==============================================================================
# PARSE MSLD v2.0 DATASET
# ==============================================================================
def parse_msld_dataset(base_path: str) -> pd.DataFrame:
    valid_exts  = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    all_files   = glob.glob(os.path.join(base_path, "**", "*"), recursive=True)
    image_paths = [
        f for f in all_files
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in valid_exts
    ]

    print(f"\n  Raw image files found under '{base_path}': {len(image_paths)}")

    if len(image_paths) == 0:
        print("  ❌ No image files found. Diagnostic info:")
        if os.path.isdir(base_path):
            top = os.listdir(base_path)
            print(f"     Top-level contents ({len(top)} items): {top[:20]}")
            for item in top[:5]:
                sub = os.path.join(base_path, item)
                if os.path.isdir(sub):
                    print(f"       └─ {item}/ → {os.listdir(sub)[:10]}")
        else:
            print(f"     Path does not exist: {base_path}")
        return pd.DataFrame()

    data             = []
    seen_originals   = set()
    skipped_mask     = 0
    skipped_no_label = 0
    src_counts       = {'dir': 0, 'filename': 0}

    for img_path in image_paths:
        filename   = os.path.basename(img_path)
        name_upper = filename.upper()

        # Exclude segmentation mask files entirely
        if 'MASK' in img_path.upper():
            skipped_mask += 1
            continue

        is_augmented = bool(re.search(r'FOLDS_AUG|AUGMENTED', img_path, re.IGNORECASE))

        label = _label_from_directory(img_path)
        if label is not None:
            src_counts['dir'] += 1
        else:
            label = _label_from_filename(name_upper)
            if label is not None:
                src_counts['filename'] += 1

        if label is None:
            skipped_no_label += 1
            continue

        # Deduplicate original images; augmented variants are all retained
        if not is_augmented:
            dedup_key = f"{filename.strip().lower()}_{label}"
            if dedup_key in seen_originals:
                continue
            seen_originals.add(dedup_key)

        data.append({
            'filepath':     img_path,
            'filename':     filename,
            'label':        label,
            'label_name':   CFG.CLASS_NAMES[label],
            'patient_id':   get_patient_id(filename),
            'is_augmented': is_augmented,
        })

    df = pd.DataFrame(data)
    print(f"  Label source  — directory: {src_counts['dir']}, "
          f"filename fallback: {src_counts['filename']}")
    print(f"  Skipped       — mask files: {skipped_mask}, "
          f"no label detected: {skipped_no_label}")
    if len(df):
        orig = df[~df['is_augmented']]
        aug  = df[df['is_augmented']]
        print(f"  Loaded        — {len(orig)} original images "
              f"({orig['patient_id'].nunique()} unique patients), "
              f"{len(aug)} augmented images")
    return df


# ==============================================================================
# MATH / METRIC HELPERS
# ==============================================================================
def stable_softmax(x: np.ndarray) -> np.ndarray:
    x     = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-8)

def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences    = np.max(probs, axis=1)
    predictions    = np.argmax(probs, axis=1)
    accuracies     = predictions == labels
    ece = 0.0
    for i in range(n_bins):
        in_bin      = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            ece += np.abs(confidences[in_bin].mean() - accuracies[in_bin].mean()) * prop_in_bin
    return ece

def logit_averaging(log_probs) -> np.ndarray:
    return stable_softmax(np.mean(np.stack(log_probs), axis=0))

def get_patient_level_arrays(log_probs, labels, pids):
    df = pd.DataFrame({'pid': pids, 'label': labels, 'logits': list(log_probs)})
    lc = df.groupby('pid')['label'].nunique()
    assert (lc == 1).all(), \
        f"CRITICAL: Label inconsistency for PIDs: {lc[lc > 1].index.tolist()}"
    agg = (df.groupby('pid')
             .agg(label=('label', 'first'), probs=('logits', logit_averaging))
             .reset_index()
             .sort_values('pid')
             .reset_index(drop=True))
    return np.stack(agg['probs'].values), agg['label'].values, agg['pid'].values

def global_sort_arrays(probs, labels, pids):
    df = pd.DataFrame({'pid': pids, 'label': labels})
    df['probs'] = list(probs)
    
    # If the datalog placed the same patient in multiple folds, average their predictions
    if len(df['pid']) != df['pid'].nunique():
        df = df.groupby('pid').agg({
            'label': 'first',
            'probs': lambda x: np.mean(np.stack(x), axis=0)
        }).reset_index()
        
    df = df.sort_values('pid').reset_index(drop=True)
    return np.stack(df['probs'].values), df['label'].values, df['pid'].values
    
def evaluate_patient_level(p_agg, y_true, verbose=True):
    preds = p_agg.argmax(axis=1)
    acc   = accuracy_score(y_true, preds)
    prec  = precision_score(y_true, preds, average='macro', zero_division=0)
    rec   = recall_score(y_true, preds, average='macro', zero_division=0)
    f1    = f1_score(y_true, preds, average='macro')

    # Crash Path 1 Fix: Handle missing classes in tiny subsets
    try:
        auc = roc_auc_score(y_true, p_agg, multi_class='ovr') \
              if len(np.unique(y_true)) > 1 else float('nan')
    except ValueError:
        auc = float('nan')

    ece = calculate_ece(p_agg, y_true)

    # Crash Path 2 Fix: Anchor LabelBinarizer to all 6 classes
    try:
        ll = log_loss(y_true, p_agg, labels=list(range(CFG.NUM_CLASSES)))
    except ValueError:
        ll = float('nan')

    if verbose:
        print(f"      ↳ Acc:{acc:.4f} | Prec:{prec:.4f} | Rec:{rec:.4f} | "
              f"F1:{f1:.4f} | AUC:{auc:.4f} | ECE:{ece:.4f} | LogLoss:{ll:.4f}")
    return f1, acc, prec, rec, auc, ece, ll

def report_with_ci(scores):
    scores = np.array(scores)
    mean   = scores.mean()
    std    = scores.std(ddof=1)
    ci95   = t.ppf(0.975, df=len(scores)-1) * (std / np.sqrt(len(scores)))
    return mean, std, ci95

def save_classification_outputs(p_agg, y_true, tag, save_dir):
    preds    = p_agg.argmax(axis=1)
    report   = classification_report(y_true, preds, target_names=CFG.CLASS_NAMES,
                                     digits=4, output_dict=False)
    out_path = os.path.join(save_dir, f"{tag}_classification_report.txt")
    with open(out_path, "w") as f:
        f.write(f"Classification Report — {tag}\n{'='*60}\n{report}")
    print(f"\n  📋 Classification Report → {out_path}")


def set_backbone_requires_grad(model, requires_grad: bool):
    cfg = model.default_cfg
    if hasattr(cfg, 'get'):
        head_name = cfg.get('classifier', None) or 'fc'
    else:
        head_name = getattr(cfg, 'classifier', None) or 'fc'
    for name, param in model.named_parameters():
        if not ('head' in name or head_name in name):
            param.requires_grad = requires_grad

def build_model(timm_name: str):
    kwargs = {'pretrained': True, 'num_classes': CFG.NUM_CLASSES}
    if any(fam in timm_name for fam in CFG.TRANSFORMER_FAMILIES):
        kwargs['img_size'] = CFG.IMG_SIZE
    try:
        model = timm.create_model(timm_name, **kwargs)
    except TypeError:
        kwargs.pop('img_size', None)
        model = timm.create_model(timm_name, **kwargs)
    return model.to(CFG.DEVICE)

@torch.no_grad()
def tta_predict(model, x, use_tta: bool = True):
    if not use_tta:
        return torch.log(F.softmax(model(x), dim=1) + 1e-8)
    views = [x, torch.flip(x, [3]),
             torch.rot90(x, 1, [2, 3]), torch.rot90(x, 3, [2, 3])]
    probs = torch.stack([F.softmax(model(v), dim=1) for v in views], dim=0)
    return torch.log(probs.mean(dim=0) + 1e-8)

def train_one_epoch(model, loader, optimizer, scaler,
                    loss_fn_soft, loss_fn_hard, mixup_fn, ep: int):
    model.train()
    for x, y, _ in loader:
        x, y = x.to(CFG.DEVICE), y.to(CFG.DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda'):
            if ep < CFG.MIXUP_DISABLE_EPOCH:
                x_mix, target = mixup_fn(x, y)
                loss          = loss_fn_soft(model(x_mix), target)
            else:
                loss = loss_fn_hard(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

def validate(model, loader, use_tta: bool = True):
    model.eval()
    all_log_probs, all_labels, all_pids = [], [], []
    with torch.no_grad():
        for x, y, pid in loader:
            probs = tta_predict(model, x.to(CFG.DEVICE), use_tta=use_tta)
            all_log_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_pids.extend(pid)
    cuda_sync()
    return (np.concatenate(all_log_probs, axis=0),
            np.concatenate(all_labels,    axis=0),
            np.array(all_pids))


def train_model(df: pd.DataFrame, datalog, display_name: str, timm_name: str):
    print(f"\n{'='*60}\n  Training: {display_name}  ({timm_name})\n{'='*60}")

    has_datalog = datalog is not None
    sgkf        = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
    original_df = df[df['is_augmented'] == False].reset_index(drop=True)
    original_df['filename_no_ext'] = original_df['filename'].apply(
        lambda f: os.path.splitext(os.path.basename(f))[0].strip().upper())

    oof_test_probs,      oof_test_labels,      oof_test_pids      = [], [], []
    oof_val_calib_probs, oof_val_calib_labels, oof_val_calib_pids = [], [], []
    fold_f1_scores, snapshot_masks = [], {}

    for fold_idx in range(5):
        fold_num = fold_idx + 1
        print(f"\n  ── Fold {fold_num}/5", flush=True)

        if time_exceeded():
            print("  ⏱️  TIME LIMIT REACHED — stopping further folds.")
            break

        datalog_success = False
        if has_datalog:
            candidate_keys = [
                f"Sheet{fold_num}", f"Fold{fold_num}", f"fold_{fold_num}",
                f"sheet_{fold_num}", f"Sheet {fold_num}", f"Fold {fold_num}",
            ]
            fold_data = None
            for key in candidate_keys:
                if key in datalog:
                    fold_data = datalog[key]
                    break

            if fold_data is None:
                print(f"  ⚠️  No matching sheet for fold {fold_num}. "
                      f"Available: {list(datalog.keys())}. Falling back to SGKF.")
            else:
                # Bug 3 Fix: Case-insensitive column search for ALL three splits,
                # not just val/test. Excel columns are frequently title-cased.
                def clean_name(x: str) -> str:
                    return os.path.splitext(os.path.basename(str(x)))[0].strip().upper()

                train_col = next((c for c in fold_data.columns
                                  if 'train' in c.lower()), None)
                val_col   = next((c for c in fold_data.columns
                                  if 'val'   in c.lower()), None)
                test_col  = next((c for c in fold_data.columns
                                  if 'test'  in c.lower()), None)

                if any(col is None for col in [train_col, val_col, test_col]):
                    missing = [n for n, c in [('train', train_col),
                                               ('val',   val_col),
                                               ('test',  test_col)] if c is None]
                    print(f"  ⚠️  Fold {fold_num} sheet missing columns {missing}. "
                          f"Found: {list(fold_data.columns)}. Falling back to SGKF.")
                else:
                    train_files = set(clean_name(f)
                                      for f in fold_data[train_col].dropna())
                    val_files   = set(clean_name(f)
                                      for f in fold_data[val_col].dropna())
                    test_files  = set(clean_name(f)
                                      for f in fold_data[test_col].dropna())

                    train_orig = original_df[
                        original_df['filename_no_ext'].isin(train_files)].copy()
                    val_df     = original_df[
                        original_df['filename_no_ext'].isin(val_files)
                    ].reset_index(drop=True)
                    test_df    = original_df[
                        original_df['filename_no_ext'].isin(test_files)
                    ].reset_index(drop=True)

                    if len(train_orig) > 0 and len(val_df) > 0 and len(test_df) > 0:
                        datalog_success = True
                        print(f"  [Datalog] train={len(train_orig)}, "
                              f"val={len(val_df)}, test={len(test_df)} images")
                    else:
                        print(f"  ⚠️  Fold {fold_num} datalog gave empty split "
                              f"(train={len(train_orig)}, val={len(val_df)}, "
                              f"test={len(test_df)}). Falling back to SGKF.")

        if not datalog_success:
            splits         = list(sgkf.split(original_df, original_df['label'],
                                             original_df['patient_id']))
            tr_idx, te_idx = splits[fold_idx]
            train_orig     = original_df.iloc[tr_idx].copy()
            test_df        = original_df.iloc[te_idx].reset_index(drop=True)

            inner_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                              random_state=CFG.SEED)
            val_split  = list(inner_sgkf.split(
                train_orig, train_orig['label'],
                train_orig['patient_id']))[fold_idx % 5][1]

            val_df     = train_orig.iloc[val_split].reset_index(drop=True)
            train_orig = train_orig.drop(
                train_orig.index[val_split]).reset_index(drop=True)
            print(f"  [SGKF] train={len(train_orig)}, "
                  f"val={len(val_df)}, test={len(test_df)} images")

        # Per-fold val/test patient disjoint assertion
        fold_overlap = set(val_df['patient_id']) & set(test_df['patient_id'])
        assert len(fold_overlap) == 0, \
            f"CRITICAL: Fold {fold_num} val/test patient overlap! " \
            f"PIDs: {list(fold_overlap)[:5]}"

        train_pids = set(train_orig['patient_id'])
        aug_df     = df[(df['is_augmented'] == True) &
                        (df['patient_id'].isin(train_pids))]
        aug_df     = (aug_df.sort_values('filepath')
                            .reset_index(drop=True)
                            .sample(frac=0.50, random_state=CFG.SEED + fold_idx))
        train_df   = pd.concat([train_orig, aug_df]).reset_index(drop=True)

        y_train_np      = train_orig['label'].values.astype(int)
        present_classes = np.unique(y_train_np)
        beta            = 0.999
        class_counts    = np.array([np.sum(y_train_np == c) for c in present_classes])
        eff_num         = 1.0 - np.power(beta, class_counts)
        weights         = (1.0 - beta) / eff_num
        weights         = weights / np.sum(weights) * CFG.NUM_CLASSES
        full_weights    = np.zeros(CFG.NUM_CLASSES, dtype=np.float32)
        for i, c in enumerate(present_classes):
            full_weights[c] = weights[i]

        train_tf, val_tf = get_transforms()
        dl_tr   = DataLoader(MpoxDataset(train_df, train_tf),
                             batch_size=CFG.BATCH_SIZE, shuffle=True,
                             num_workers=CFG.NUM_WORKERS, drop_last=True,
                             pin_memory=True, generator=get_generator())
        dl_va   = DataLoader(MpoxDataset(val_df, val_tf),
                             batch_size=CFG.BATCH_SIZE, shuffle=False,
                             num_workers=CFG.NUM_WORKERS, pin_memory=True)
        dl_test = DataLoader(MpoxDataset(test_df, val_tf),
                             batch_size=CFG.BATCH_SIZE, shuffle=False,
                             num_workers=CFG.NUM_WORKERS, pin_memory=True)

        model     = build_model(timm_name)
        set_backbone_requires_grad(model, False)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=1e-3, weight_decay=1e-2)
        scaler    = GradScaler('cuda', enabled=True, growth_interval=1000)
        scheduler = CosineLRScheduler(optimizer, t_initial=CFG.EPOCHS,
                                      lr_min=1e-6, warmup_t=2, warmup_lr_init=1e-6)
        mixup_fn     = Mixup(mixup_alpha=0.1, cutmix_alpha=0.0,
                             label_smoothing=0.1, num_classes=CFG.NUM_CLASSES)
        loss_fn_soft = SoftTargetCrossEntropy()
        loss_fn_hard = nn.CrossEntropyLoss(
            weight=torch.tensor(full_weights).to(CFG.DEVICE),
            label_smoothing=0.1)

        val_pids_sorted = np.sort(val_df['patient_id'].unique())
        if fold_num not in snapshot_masks:
            rng  = np.random.RandomState(CFG.SEED + fold_idx)
            idx  = rng.permutation(len(val_pids_sorted))
            half = len(idx) // 2
            snapshot_masks[fold_num] = {'sel': idx[:half], 'calib': idx[half:]}

        sel_idx   = snapshot_masks[fold_num]['sel']
        calib_idx = snapshot_masks[fold_num]['calib']
        val_snapshots = []

        # ── Training loop ──────────────────────────────────────────────────
        for ep in range(CFG.EPOCHS):
            if time_exceeded(): break

            if ep == CFG.FREEZE_EPOCHS:
                set_backbone_requires_grad(model, True)
                del optimizer, scheduler
                torch.cuda.empty_cache()
                optimizer = optim.AdamW(model.parameters(),
                                        lr=1e-4, weight_decay=1e-2)
                scheduler = CosineLRScheduler(
                    optimizer,
                    t_initial=(CFG.EPOCHS - CFG.FREEZE_EPOCHS),
                    lr_min=1e-6, warmup_t=2, warmup_lr_init=1e-6)

            train_one_epoch(model, dl_tr, optimizer, scaler,
                            loss_fn_soft, loss_fn_hard, mixup_fn, ep)
            scheduler.step(ep if ep < CFG.FREEZE_EPOCHS else ep - CFG.FREEZE_EPOCHS)

            v_log, v_lbl, v_pid = validate(model, dl_va, use_tta=False)
            v_p, v_pl, v_pp     = get_patient_level_arrays(v_log, v_lbl, v_pid)

            assert np.array_equal(v_pp, val_pids_sorted), \
                f"CRITICAL: PID alignment broken fold {fold_num} ep {ep}!"

            _, _, _, _, _, _, sel_ll = evaluate_patient_level(
                v_p[sel_idx], v_pl[sel_idx], verbose=False)

            sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            val_snapshots.append((sel_ll, sd, ep + 1))   # [V34-W] + epoch number (1-based)
            val_snapshots.sort(key=lambda s: s[0])
            val_snapshots = val_snapshots[:2]
            gc.collect()

        # ── Early exit: explicit cleanup of ALL GPU-holding objects ────────
        if len(val_snapshots) == 0:
            # Bug 4 Fix: optimizer/scheduler/scaler deleted here too.
            del dl_tr, dl_va, dl_test
            del model, optimizer, scheduler, scaler
            del loss_fn_soft, loss_fn_hard, mixup_fn
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # ── Post-training TTA evaluation ───────────────────────────────────
        val_calib_evals, test_evals = [], []
        for _, sd, _ep in val_snapshots:              # [V34-W] tuple now carries the epoch
            model.load_state_dict(sd)

            v_log, v_lbl, v_pid = validate(model, dl_va, use_tta=True)
            v_p, v_pl, v_pp     = get_patient_level_arrays(v_log, v_lbl, v_pid)
            assert np.array_equal(v_pp, val_pids_sorted), \
                f"CRITICAL: PID alignment broken fold {fold_num} (TTA val)!"
            val_calib_evals.append(v_p[calib_idx])

            t_log, t_lbl, t_pid = validate(model, dl_test, use_tta=True)
            t_p, t_pl, t_pp     = get_patient_level_arrays(t_log, t_lbl, t_pid)
            test_evals.append(t_p)

        fold_val_calib = np.mean(val_calib_evals, axis=0)
        fold_test      = np.mean(test_evals,      axis=0)

        fold_f1 = evaluate_patient_level(
            fold_val_calib, v_pl[calib_idx], verbose=False)[0]
        print(f"  🎯 Fold {fold_num} Disjoint Calib-F1 = {fold_f1:.4f}")

        fold_f1_scores.append(fold_f1)
        oof_val_calib_probs.append(fold_val_calib)
        oof_val_calib_labels.append(v_pl[calib_idx])
        oof_val_calib_pids.append(v_pp[calib_idx])
        oof_test_probs.append(fold_test)
        oof_test_labels.append(t_pl)
        oof_test_pids.append(t_pp)

        save_fold_checkpoint(display_name, fold_num, {
            'test_probs':  fold_test,        'test_labels': t_pl,
            'test_pids':   t_pp,             'val_probs':   fold_val_calib,
            'val_labels':  v_pl[calib_idx],  'val_pids':    v_pp[calib_idx],
        })

        # [V34-W] keep the two selected snapshots' weights and the fold definition
        import json
        wdir = os.path.join(CFG.SAVE_DIR, "weights")
        os.makedirs(wdir, exist_ok=True)
        snap_meta = []
        for rank, (s_ll, s_sd, s_ep) in enumerate(val_snapshots, 1):
            wpath = os.path.join(wdir, f"{display_name}_fold{fold_num}_rank{rank}.pt")
            torch.save(s_sd, wpath + ".tmp")
            os.replace(wpath + ".tmp", wpath)
            snap_meta.append({"rank": rank, "epoch": int(s_ep),
                              "selection_logloss": float(s_ll), "file": os.path.basename(wpath)})
        sel_pids = set(val_pids_sorted[sel_idx])
        roles = pd.concat([
            train_orig.assign(role="train_original"),
            val_df.assign(role=np.where(val_df['patient_id'].isin(sel_pids),
                                        "val_selection", "val_calibration")),
            test_df.assign(role="test"),
        ])[['filename', 'patient_id', 'label', 'role']]
        roles.to_csv(os.path.join(wdir, f"{display_name}_fold{fold_num}_assignment.csv"), index=False)
        aug_df[['filepath', 'patient_id', 'label']].to_csv(
            os.path.join(wdir, f"{display_name}_fold{fold_num}_augmented_used.csv.gz"), index=False)
        with open(os.path.join(wdir, f"{display_name}_fold{fold_num}_meta.json"), "w") as f:
            json.dump({
                "model": display_name, "timm_name": timm_name, "fold": fold_num,
                "fold_source": "datalog" if datalog_success else "SGKF fallback",
                "n_train_original": int(len(train_orig)), "n_augmented_used": int(len(aug_df)),
                "n_val": int(len(val_df)), "n_test": int(len(test_df)),
                "n_val_selection_patients": int(len(sel_idx)),
                "n_val_calibration_patients": int(len(calib_idx)),
                "snapshots": snap_meta, "calibration_half_f1": float(fold_f1),
                "class_names": CFG.CLASS_NAMES, "img_size": CFG.IMG_SIZE,
                "normalize_mean": [0.485, 0.456, 0.406], "normalize_std": [0.229, 0.224, 0.225],
                "eval_resize": "Resize((img_size, img_size)), no crop",
                "tta_views": ["identity", "hflip", "rot90", "rot270"],
                "tta_combine": "mean of softmax probabilities, then log",
                "patient_aggregation": "softmax(mean of image log-probabilities)",
                "snapshot_combine": "mean of the two snapshots' patient probabilities",
            }, f, indent=2)
        print(f"  💾 [V34-W] Fold {fold_num} weights (epochs "
              f"{[s['epoch'] for s in snap_meta]}) and assignment saved → {wdir}")

        # Bug 4 Fix: Explicitly delete ALL GPU-holding objects so CUDA VRAM is
        # freed immediately via reference counting, not at GC's discretion.
        # optimizer holds param tensor refs → must be deleted before/with model.
        del dl_tr, dl_va, dl_test
        del model, optimizer, scheduler, scaler
        del loss_fn_soft, loss_fn_hard, mixup_fn
        del val_snapshots, test_evals
        torch.cuda.empty_cache()
        gc.collect()

    if len(fold_f1_scores) == 0:
        return None

    mean_f1, std_f1, ci_f1 = report_with_ci(fold_f1_scores)
    print(f"\n  📊 {display_name} Cross-Fold Calib-F1: "
          f"{mean_f1:.4f} ± {std_f1:.4f}  (95% CI ±{ci_f1:.4f})")

    t_p, t_pl, t_pp = global_sort_arrays(
        np.concatenate(oof_test_probs,  axis=0),
        np.concatenate(oof_test_labels, axis=0),
        np.concatenate(oof_test_pids,   axis=0))

    v_p, v_pl, v_pp = global_sort_arrays(
        np.concatenate(oof_val_calib_probs,  axis=0),
        np.concatenate(oof_val_calib_labels, axis=0),
        np.concatenate(oof_val_calib_pids,   axis=0))

    #assert len(np.unique(t_pp)) == len(t_pp), \
        #"CRITICAL: Duplicate patient IDs in final test set!"
    #assert len(np.unique(v_pp)) == len(v_pp), \
        #"CRITICAL: Duplicate patient IDs in final val set!"

    final_arrays = {
        f"{display_name}_test_probs":  t_p,
        f"{display_name}_test_labels": t_pl,
        f"{display_name}_test_pids":   t_pp,
        f"{display_name}_val_probs":   v_p,
        f"{display_name}_val_labels":  v_pl,
        f"{display_name}_val_pids":    v_pp,
    }

# NaN/Inf guard — validate all arrays before any file is written
    for arr_name, arr in final_arrays.items():
        if 'pids' in arr_name:
            continue  # NumPy cannot check strings for NaN/Inf
            
        assert not np.any(np.isnan(arr)), \
            f"CRITICAL: NaN in '{arr_name}' before saving!"
        assert not np.any(np.isinf(arr)), \
            f"CRITICAL: Inf in '{arr_name}' before saving!"

    os.makedirs(CFG.SAVE_DIR, exist_ok=True) # <-- THIS LINE FIXES THE CRASH
    for arr_name, arr in final_arrays.items():
        atomic_save(arr, os.path.join(CFG.SAVE_DIR, f"{arr_name}.npy"))
        
    print(f"  💾 Final 6 .npy arrays atomically saved for {display_name}.")
    return t_p, t_pl, display_name


def print_dataset_statistics(df: pd.DataFrame):
    orig = df[~df['is_augmented']]
    stats = (orig.groupby('label_name')
                 .agg(Images=('filepath', 'count'),
                      Patients=('patient_id', 'nunique'))
                 .reset_index())
    print(stats.to_string(index=False))
    print(f"\nTotal original images : {len(orig)}"
          f"\nTotal unique patients : {orig['patient_id'].nunique()}"
          f"\nTotal augmented images: {len(df[df['is_augmented']])}\n")


if __name__ == "__main__":
    print("=" * 60)
    print(" MSLD v2.0 — Worker Script V34")
    print("=" * 60)

    os.makedirs(CFG.SAVE_DIR, exist_ok=True)

# Hardcoding the exact path to bypass auto-discovery
    dataset_dir = "/kaggle/input/datasets"  # <-- PASTE YOUR EXACT COPIED PATH HERE

    df = parse_msld_dataset(dataset_dir)
    if len(df) == 0:
        print("\n❌ CRITICAL: 0 images parsed. Check class subdirectory names.")
        print("   Expected: Monkeypox/, Chickenpox/, Measles/, Cowpox/, Healthy/, HFMD/")
        raise SystemExit(1)

    print_dataset_statistics(df)

    # [V34-W] pre-flight: the run must see exactly the files of the original run
    n_orig = int((df['is_augmented'] == False).sum())
    n_aug  = int((df['is_augmented'] == True).sum())
    n_pat  = int(df.loc[df['is_augmented'] == False, 'patient_id'].nunique())
    assert (n_orig, n_aug, n_pat) == (755, 37044, 524), (
        f"Expected 755 original / 37044 augmented images and 524 patients (as in the original "
        f"run) but found {n_orig} / {n_aug} / {n_pat}. Attach ONLY the MSLD v2.0 dataset.")
    import json, platform, sklearn, scipy, torchvision
    run_info = {
        "script": "V34-W (V34 + weight saving)", "python": platform.python_version(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "timm": timm.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "scipy": scipy.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "seed": CFG.SEED, "img_size": CFG.IMG_SIZE, "batch_size": CFG.BATCH_SIZE,
        "epochs": CFG.EPOCHS, "freeze_epochs": CFG.FREEZE_EPOCHS,
        "mixup_disable_epoch": CFG.MIXUP_DISABLE_EPOCH, "class_names": CFG.CLASS_NAMES,
        "models": CFG.MODELS, "dataset_dir": dataset_dir,
        "n_original": n_orig, "n_augmented": n_aug, "n_patients": n_pat,
    }
    with open(os.path.join(CFG.SAVE_DIR, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    print("  [V34-W] run_info:", json.dumps(run_info))

    datalog      = None
    datalog_hits = glob.glob(os.path.join(dataset_dir, "**", "datalog.xlsx"),
                             recursive=True)
    if datalog_hits:
        datalog_path = datalog_hits[0]
        print(f"  📄 Loading datalog: {datalog_path}")
        try:
            datalog = pd.read_excel(datalog_path, sheet_name=None, engine='openpyxl')
            print(f"  ✅ Datalog loaded. Sheets: {list(datalog.keys())}")
        except Exception as e:
            print(f"  ⚠️  datalog.xlsx unreadable ({e}). Using SGKF.")
            datalog = None
    else:
        print("  ⚠️  datalog.xlsx not found. Using SGKF for all folds.")

    for display_name, timm_name in CFG.MODELS.items():
        result = train_model(df, datalog, display_name, timm_name)
        if result:
            t_p, t_pl, name = result
            print(f"\n  📊 {name} — Final Test Set Evaluation:")
            evaluate_patient_level(t_p, t_pl, verbose=True)
            save_classification_outputs(t_p, t_pl, name, CFG.SAVE_DIR)
        else:
            print(f"\n  ❌ {display_name} produced no complete folds. "
                  f"Check time limit and dataset integrity.")