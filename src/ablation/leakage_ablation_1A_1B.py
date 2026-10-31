"""Leakage ablation, pipelines 1A and 1B side by side (paper Section 3.3, Table 5).

1A: augment all 755 originals x10, then split the 7,550 images 80:20 at random
    (image-level, leaky).
1B: split the 755 originals by patient first (StratifiedGroupKFold, one fold
    held out), then augment only the training images x10 (patient-level).

Same ViT-Tiny (128 px), optimiser (Adam, lr 1e-3), batch size 32 and 100 epochs
for both. The banner printed at the end says "10 EPOCHS"; that is a leftover
label, the loop runs 100 epochs. Archived as run. Reported output:
1A 89.47%, 1B 62.14% (87/140 test images).
"""
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import accuracy_score
import timm
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
print("Initiating Ultimate Side-by-Side Methodology Comparison...\n")

# ---------------------------------------------------------
# 1. Load Original Dataset
# ---------------------------------------------------------
CLASS_MAP = {'CHICKENPOX': 0, 'COWPOX': 1, 'HFMD': 2, 'HEALTHY': 3, 'MEASLES': 4, 'MONKEYPOX': 5}
dataset_dir = "/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20/Original Images/Original Images"

all_files = glob.glob(os.path.join(dataset_dir, "**/*.*"), recursive=True)
data, seen_images = [], set()

for img_path in all_files:
    if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')): continue
    filename = os.path.basename(img_path)
    folder_name = os.path.basename(os.path.dirname(img_path)).upper()
    if folder_name not in CLASS_MAP: continue

    img_sig = f"{folder_name}_{filename}"
    if img_sig in seen_images: continue
    seen_images.add(img_sig)

    parts = filename.replace('-', '_').split('_')
    patient_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else filename
    data.append({'filepath': img_path, 'label': CLASS_MAP[folder_name], 'patient_id': patient_id})

df_original = pd.DataFrame(data)
print(f"Loaded {len(df_original)} original images.\n")

# ---------------------------------------------------------
# 2. Define Augmentations
# ---------------------------------------------------------
matlab_simulator = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.15, hue=0.1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

base_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def expand_dataframe(df_to_expand, multiplier=10):
    """Expands images in memory using the MATLAB simulator."""
    images, labels = [], []
    for idx, row in df_to_expand.iterrows():
        img = Image.open(row['filepath']).convert('RGB')
        images.append(base_transform(img))
        labels.append(row['label'])
        for _ in range(multiplier - 1):
            images.append(matlab_simulator(img))
            labels.append(row['label'])
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)

class TensorDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ---------------------------------------------------------
# 3. Prepare Pipeline A: THE ILLUSION (Augment First, Split Later)
# ---------------------------------------------------------
print("Preparing Pipeline A (The Authors' Method - Leaked)...")
X_all_ill, y_all_ill = expand_dataframe(df_original, multiplier=10)
X_train_ill, X_test_ill, y_train_ill, y_test_ill = train_test_split(
    X_all_ill.numpy(), y_all_ill.numpy(), test_size=0.20, random_state=42, stratify=y_all_ill.numpy()
)

loader_train_ill = DataLoader(TensorDataset(torch.tensor(X_train_ill), torch.tensor(y_train_ill)), batch_size=32, shuffle=True)
loader_test_ill = DataLoader(TensorDataset(torch.tensor(X_test_ill), torch.tensor(y_test_ill)), batch_size=32, shuffle=False)

# ---------------------------------------------------------
# 4. Prepare Pipeline B: THE REALITY (Split First, Augment Train Only)
# ---------------------------------------------------------
print("Preparing Pipeline B (Your Method - Strict Isolation)...")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(df_original, df_original['label'], df_original['patient_id']))

df_train_strict = df_original.iloc[train_idx]
df_test_strict = df_original.iloc[test_idx]

# Expand ONLY the training set so the model has the exact same amount of data to learn from
X_train_str, y_train_str = expand_dataframe(df_train_strict, multiplier=10)
X_test_str, y_test_str = expand_dataframe(df_test_strict, multiplier=1) # No augmentation for test!

loader_train_str = DataLoader(TensorDataset(X_train_str, y_train_str), batch_size=32, shuffle=True)
loader_test_str = DataLoader(TensorDataset(X_test_str, y_test_str), batch_size=32, shuffle=False)

# ---------------------------------------------------------
# 5. Training Engine
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_eval(train_loader, test_loader, model_name):
    print(f"\nTraining {model_name}...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=6, img_size=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train for 10 epochs
    model.train()
    for epoch in range(100):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            preds.extend(torch.argmax(model(inputs.to(device)), 1).cpu().numpy())
            trues.extend(labels.numpy())

    return accuracy_score(trues, preds)

# ---------------------------------------------------------
# 6. Execute the Duel
# ---------------------------------------------------------
acc_illusion = train_and_eval(loader_train_ill, loader_test_ill, "Model A (Leaked Augmentation)")
acc_reality = train_and_eval(loader_train_str, loader_test_str, "Model B (Strict Patient Isolation)")

# ---------------------------------------------------------
# 7. The Final Verdict
# ---------------------------------------------------------
print("\n" + "="*60)
print("🏆 ULTIMATE METHODOLOGY COMPARISON (10 EPOCHS) 🏆")
print("="*60)
print(f"Model A (The Paper's Method):   {acc_illusion * 100:.2f}%  <-- Inflated by leaking clones")
print(f"Model B (Your Strict Method):   {acc_reality * 100:.2f}%  <-- The true generalization capability")
print("="*60)
print("Conclusion: Both models were given ~6,000 training images.")
print("Model A scored higher purely because augmented copies of the test")
print("images were present in its training set. Model B is clinically honest.")
