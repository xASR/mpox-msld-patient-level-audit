# ==============================================================================
# OPTIMIZED MEDICAL ML PIPELINE v8 (The Grandmaster / Publication Edition)
# Features: BEiT-Base Upgrade, ViT Warmup Scheduling, Logit-Level TTA, 
#           Confidence-Weighted Patient Aggregation, Class Imbalance Handling, 
#           and Top-K Ensemble Pruning.
# ==============================================================================

import os, random, gc, glob
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import torchvision.transforms as T
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, log_loss, 
    precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

# ===================== CONFIG =====================
class CFG:
    SEED                = 42
    IMG_SIZE            = 384
    BATCH_SIZE          = 16
    EPOCHS              = 25
    FREEZE_EPOCHS       = 3   
    MIXUP_DISABLE_EPOCH = 10  
    NUM_CLASSES         = 6
    CLASS_MAP = {
        'HEALTHY': 0, 'COWPOX': 1, 'MONKEYPOX': 2, 
        'MEASLES': 3, 'CHICKENPOX': 4, 'HFMD': 5
    }
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔴 The Final 4-Model Sequential Flow
    MODELS = {
        "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",    
        "convnext_tiny":    "convnext_tiny.in12k_ft_in1k_384",      
        "swin_small":       "swin_small_patch4_window7_224",   # 🔴 FIXED: Changed 384 back to 224     
        "beit_base":        "beit_base_patch16_384",                
    }

    TRANSFORMER_FAMILIES = [
        'swin', 'vit', 'deit', 'coat', 'maxvit', 'pvt',
        'beit', 'xcit', 'twins', 'cait', 'levit', 'pit'
    ]

    SAVE_DIR = "./outputs"

# ===================== SEED =====================
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(CFG.SEED)

# ===================== DATASET =====================
class MpoxDataset(Dataset):
    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.filepath).convert('RGB')
        return self.transform(img), int(row.label), str(row.patient_id)

# ===================== AUGMENTATIONS =====================
def get_transforms():
    train_tf = T.Compose([
        T.RandomResizedCrop(CFG.IMG_SIZE, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15), 
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
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

# ===================== PATIENT-LEVEL AGGREGATOR =====================
def calculate_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            ece += np.abs(confidences[in_bin].mean() - accuracies[in_bin].mean()) * prop_in_bin
    return ece

def confidence_weighted_avg(probs):
    probs_stack = np.stack(probs)
    conf = np.max(probs_stack, axis=1, keepdims=True)
    weights = conf / (np.sum(conf) + 1e-8)
    return np.sum(weights * probs_stack, axis=0).tolist()

def evaluate_patient_level(probs: np.ndarray, labels: np.ndarray, pids: np.ndarray):
    records = pd.DataFrame({'pid': pids, 'label': labels})
    records['probs'] = list(probs)

    agg = (records
           .groupby('pid')
           .agg(label=('label', 'first'),
                probs=('probs', confidence_weighted_avg))
           .reset_index())

    y_true   = agg['label'].values
    p_agg    = np.stack(agg['probs'].values)
    preds    = p_agg.argmax(axis=1)
    
    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average='macro', zero_division=0)
    rec  = recall_score(y_true, preds, average='macro', zero_division=0)
    f1   = f1_score(y_true, preds, average='macro')
    
    try:
        auc = roc_auc_score(y_true, p_agg, multi_class='ovr')
    except ValueError:
        auc = float('nan')
        
    ece  = calculate_ece(p_agg, y_true)
    
    print(f"      ↳ [Metrics] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | ECE: {ece:.4f}")
    return f1, p_agg, y_true, agg['pid'].values

# ===================== MODEL BUILDER & FREEZING =====================
def build_model(timm_name: str) -> nn.Module:
    kwargs = {'pretrained': True, 'num_classes': CFG.NUM_CLASSES}
    if any(fam in timm_name for fam in CFG.TRANSFORMER_FAMILIES):
        kwargs['img_size'] = CFG.IMG_SIZE
    try:
        model = timm.create_model(timm_name, **kwargs)
    except TypeError:
        kwargs.pop('img_size', None)
        model = timm.create_model(timm_name, **kwargs)
    return model.to(CFG.DEVICE)

def set_backbone_requires_grad(model, requires_grad):
    head_name = model.default_cfg.get('classifier', 'fc')
    for name, param in model.named_parameters():
        if not ('head' in name or head_name in name):
            param.requires_grad = requires_grad

# ===================== TTA =====================
@torch.no_grad()
def tta_predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    views = [
        x,
        torch.flip(x, [3]),                                   
        F.interpolate(                                         
            F.interpolate(x, scale_factor=0.9, mode='bilinear', align_corners=False),
            size=x.shape[-2:], mode='bilinear', align_corners=False),
        F.interpolate(                                         
            F.interpolate(x, scale_factor=1.1, mode='bilinear', align_corners=False),
            size=x.shape[-2:], mode='bilinear', align_corners=False),
    ]
    outs = [model(v) for v in views]
    mean_logits = torch.mean(torch.stack(outs), dim=0)
    return F.softmax(mean_logits, dim=1)

# ===================== TRAIN ONE EPOCH =====================
def train_one_epoch(model, loader, optimizer, scaler, loss_fn_soft, loss_fn_hard, mixup_fn, ep):
    model.train()
    total_loss = 0.0
    for x, y, _ in loader:
        x, y = x.to(CFG.DEVICE), y.to(CFG.DEVICE)
        
        optimizer.zero_grad()
        with autocast('cuda'):
            if ep < CFG.MIXUP_DISABLE_EPOCH:
                x_mix, target = mixup_fn(x, y)
                loss = loss_fn_soft(model(x_mix), target)
            else:
                loss = loss_fn_hard(model(x), y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
    return total_loss / len(loader)

# ===================== VALIDATE =====================
def validate(model, loader):
    model.eval()
    all_probs, all_labels, all_pids = [], [], []
    with torch.no_grad():
        for x, y, pid in loader:
            p = tta_predict(model, x.to(CFG.DEVICE))
            all_probs.append(p.cpu())
            all_labels.append(y)
            all_pids.extend(pid)
    return (torch.cat(all_probs).numpy(),
            torch.cat(all_labels).numpy(),
            np.array(all_pids))

# ===================== TRAIN FULL MODEL =====================
def train_model(df: pd.DataFrame, display_name: str, timm_name: str):
    print(f"\n{'='*60}")
    print(f"  Training: {display_name}  ({timm_name})")
    print(f"{'='*60}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
    oof_probs, oof_labels, oof_pids = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(df, df['label'], df['patient_id'])):
        print(f"  Fold {fold+1}/5 ...", end=' ', flush=True)
        train_df = df.iloc[tr_idx]
        val_df   = df.iloc[va_idx]
        
        classes = np.unique(train_df['label'])
        weights = compute_class_weight('balanced', classes=classes, y=train_df['label'])
        class_weights = torch.tensor(weights, dtype=torch.float32).to(CFG.DEVICE)

        tf_tr, tf_val = get_transforms()
        dl_tr = DataLoader(MpoxDataset(train_df, tf_tr), batch_size=CFG.BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
        dl_va = DataLoader(MpoxDataset(val_df,   tf_val), batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=CFG.NUM_CLASSES)
        
        loss_fn_soft = SoftTargetCrossEntropy()
        loss_fn_hard = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        model = build_model(timm_name)
        
        set_backbone_requires_grad(model, False)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
        scaler = GradScaler()
        
        scheduler = CosineLRScheduler(
            optimizer, t_initial=CFG.EPOCHS, lr_min=1e-6, 
            warmup_t=2, warmup_lr_init=1e-6
        )

        snapshots = []

        for ep in range(CFG.EPOCHS):
            if ep == CFG.FREEZE_EPOCHS:
                print(f"\n      🔓 Epoch {ep+1}: Unfreezing Backbone for Fine-Tuning...")
                set_backbone_requires_grad(model, True)
                optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
                scheduler = CosineLRScheduler(
                    optimizer, t_initial=(CFG.EPOCHS - CFG.FREEZE_EPOCHS), 
                    lr_min=1e-6, warmup_t=2, warmup_lr_init=1e-6
                )
            
            if ep == CFG.MIXUP_DISABLE_EPOCH:
                print(f"\n      🛑 Epoch {ep+1}: Disabling MixUp for Clean Label Fine-Tuning...")

            train_one_epoch(model, dl_tr, optimizer, scaler, loss_fn_soft, loss_fn_hard, mixup_fn, ep)
            
            step_idx = ep if ep < CFG.FREEZE_EPOCHS else ep - CFG.FREEZE_EPOCHS
            scheduler.step(step_idx)
            
            ep_probs, ep_labels, ep_pids = validate(model, dl_va)
            
            print(f"\n      Epoch {ep+1}/{CFG.EPOCHS}")
            ep_f1, _, _, _ = evaluate_patient_level(ep_probs, ep_labels, ep_pids)

            snapshots.append((ep_f1, ep_probs, ep_labels, ep_pids))
            snapshots.sort(key=lambda s: s[0], reverse=True)
            snapshots = snapshots[:3]                   

        f1_scores = [s[0] for s in snapshots]
        weights = softmax(f1_scores)
        fold_probs = np.sum([w * s[1] for w, s in zip(weights, snapshots)], axis=0)
        
        fold_labels = snapshots[0][2]
        fold_pids   = snapshots[0][3]

        best_f1 = snapshots[0][0]
        print(f"\n  🎯 Best Fold Patient-F1 = {best_f1:.4f}")

        oof_probs.append(fold_probs)
        oof_labels.append(fold_labels)
        oof_pids.append(fold_pids)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    oof_probs  = np.concatenate(oof_probs,  axis=0)
    oof_labels = np.concatenate(oof_labels, axis=0)
    oof_pids   = np.concatenate(oof_pids,   axis=0)

    print(f"\n  ✅ {display_name} FINAL OOF EVALUATION:")
    oof_f1, _, _, _ = evaluate_patient_level(oof_probs, oof_labels, oof_pids)
    
    pd.DataFrame({'patient_id': oof_pids, 'label': oof_labels, 'probs': list(oof_probs)}).to_csv(f"{CFG.SAVE_DIR}/oof_{display_name}.csv", index=False)
    return oof_probs, oof_labels, oof_pids

# ===================== WEIGHT OPTIMIZER & TEMP SCALING =====================
def tune_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    def eval_opt(t):
        scaled_logits = np.log(probs + 1e-8) / t[0]
        return log_loss(labels, softmax(scaled_logits, axis=1))
    res = minimize(eval_opt, [1.0], bounds=[(0.1, 5.0)], method='L-BFGS-B')
    return res.x[0]

def optimize_weights(probs_list: list, labels: np.ndarray) -> np.ndarray:
    n = len(probs_list)
    def neg_loss(raw_w):
        w   = softmax(raw_w)                           
        mix = sum(w[i] * probs_list[i] for i in range(n))
        return log_loss(labels, mix)

    result = minimize(neg_loss, x0=np.ones(n), method='L-BFGS-B')
    return softmax(result.x)

# ===================== KAGGLE DATA PARSING =====================
def parse_msld_dataset(base_path):
    all_files = glob.glob(os.path.join(base_path, "**/*.*"), recursive=True)
    valid_exts = ('.jpg', '.jpeg', '.png')
    image_paths = [f for f in all_files if f.lower().endswith(valid_exts)]
    
    data = []
    seen_images = set()
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        name_upper = filename.upper()
        
        # 🔴 BULLETPROOF FILTER: Explicitly reject the Augmented Images folders
        if 'AUGMENTED IMAGES' in img_path.upper() or 'AUGMENTED' in img_path.upper() or 'MASK' in img_path.upper() or 'FOLDS_AUG' in img_path.upper():
            continue
            
        if name_upper.startswith('MKP') or name_upper.startswith('MPOX'): label = 2 
        elif name_upper.startswith('CHP') or name_upper.startswith('CHICKEN'): label = 4 
        elif name_upper.startswith('MSL') or name_upper.startswith('MEASLE'): label = 3 
        elif name_upper.startswith('CWP') or name_upper.startswith('COWPOX'): label = 1 
        elif name_upper.startswith('HFMD') or 'HAND' in name_upper: label = 5 
        elif name_upper.startswith('HEALTHY') or name_upper.startswith('NORMAL'): label = 0 
        else:
            folder_name = os.path.dirname(img_path).upper()
            if 'MPOX' in folder_name or 'MONKEY' in folder_name: label = 2
            elif 'CHICKEN' in folder_name: label = 4
            elif 'MEASLE' in folder_name: label = 3
            elif 'COW' in folder_name: label = 1
            elif 'HFMD' in folder_name or 'HAND' in folder_name: label = 5
            elif 'HEALTH' in folder_name: label = 0
            else: continue
                
        if name_upper in seen_images: 
            continue
        seen_images.add(name_upper)
        
        parts = filename.replace('-', '_').split('_')
        patient_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else filename.split('.')[0]
        
        data.append({'filepath': img_path, 'label': label, 'patient_id': patient_id})
        
    return pd.DataFrame(data)

# ===================== MAIN =====================
def main(df):
    os.makedirs(CFG.SAVE_DIR, exist_ok=True)

    results    = {}   
    labels_ref = None
    pids_ref   = None

    for display_name, timm_name in CFG.MODELS.items():
        oof_probs, oof_labels, oof_pids = train_model(df, display_name, timm_name)
        results[display_name] = (oof_probs, oof_labels, oof_pids, evaluate_patient_level(oof_probs, oof_labels, oof_pids)[0])
        labels_ref = oof_labels   
        pids_ref   = oof_pids

    print("\n" + "="*60)
    print("  Champion Ensemble (Dynamic Weights, Calibrated, Top-K Pruned)")
    print("="*60)

    model_entries = list(results.items())
    model_entries.sort(key=lambda x: x[1][3], reverse=True) # Sort by F1
    top_models = model_entries[:3] # Keep top 3
    
    calibrated_probs_list = []
    
    for name, data in top_models:
        probs, labels, pids, f1 = data
        t_opt = tune_temperature(probs, labels)
        calibrated_probs = softmax(np.log(probs + 1e-8) / t_opt, axis=1)
        calibrated_probs_list.append(calibrated_probs)
        print(f"  [Pruned & Calibrated] {name:20s} F1: {f1:.4f} | Temp: {t_opt:.2f}")

    if len(calibrated_probs_list) > 1:
        weights = optimize_weights(calibrated_probs_list, labels_ref)  
        for (name, _), w in zip(top_models, weights):
            print(f"  {name:20s} Final Ensemble Weight = {w:.4f}")

        champion_probs = sum(weights[i] * calibrated_probs_list[i] for i in range(len(weights)))
        print(f"\n  🏆 FINAL CHAMPION ENSEMBLE EVALUATION:")
        champ_f1, _, _, _ = evaluate_patient_level(champion_probs, labels_ref, pids_ref)

if __name__ == "__main__":
    dataset_dir = "/kaggle/input/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20"
    print(f"Checking path: {dataset_dir}")
    print("Loading dataset...")
    df = parse_msld_dataset(dataset_dir)
    
    if len(df) == 0:
        print("Trying recursive root search...")
        df = parse_msld_dataset("/kaggle/input")
        
    print(f"Loaded {len(df)} unique images.")
    
    if len(df) == 0:
        print("❌ CRITICAL ERROR: 0 images loaded. Check if dataset is attached to the notebook.")
    else:
        main(df)