"""Leakage ablation, pipeline 1A (image-level split after augmentation).

Archived as run for the paper (Section 3.3, Table 5). Each of the 755 original
images is expanded to 10 images (original + 9 MATLAB-style augmented copies),
and the pool of 7,550 images is split 80:20 at random. Augmented copies of the
same photograph, and photographs of the same patient, land on both sides.

Run on Kaggle with one GPU. Edit `dataset_dir` below if your input path differs.
Not seeded beyond random_state=42 for the split, so repeated runs differ:
two runs gave 82.12% and 89.47% (the second inside leakage_ablation_1A_1B.py).
For the seeded, repeated version see leakage_ablation_seeded.py.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import timm
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
print("Initiating Exact Methodology Replication (The 94% Illusion)...")

# ---------------------------------------------------------
# 1. Load Original Dataset (The Base 755 Images)
# ---------------------------------------------------------
CLASS_MAP = {'CHICKENPOX': 0, 'COWPOX': 1, 'HFMD': 2, 'HEALTHY': 3, 'MEASLES': 4, 'MONKEYPOX': 5}
dataset_dir = "/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20/Original Images/Original Images"

all_files = glob.glob(os.path.join(dataset_dir, "**/*.*"), recursive=True)
original_data = []
seen_images = set()

for img_path in all_files:
    if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')): continue
    filename = os.path.basename(img_path)
    folder_name = os.path.basename(os.path.dirname(img_path)).upper()
    if folder_name not in CLASS_MAP: continue

    img_signature = f"{folder_name}_{filename}"
    if img_signature in seen_images: continue
    seen_images.add(img_signature)
    original_data.append({'filepath': img_path, 'label': CLASS_MAP[folder_name]})

df_original = pd.DataFrame(original_data)
print(f"Loaded {len(df_original)} original images.")

# ---------------------------------------------------------
# 2. Simulate the MATLAB Offline Augmentation (1 -> 10 multiplier)
# ---------------------------------------------------------
# Matching their exact methodology parameters:
# Rotation ±30°, Translation ±10%, Shearing ±15%, Scaling 0.8-1.2, Color adjustments
matlab_simulator = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.15, hue=0.1),
    transforms.Resize((128, 128)), # Dimensional Standardization exactly as specified
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [0,1] normalization equivalent
])

base_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("Simulating MATLAB Offline Augmentation (Expanding to ~7500 images)...")
expanded_images = []
expanded_labels = []

# Expand the dataset by generating 9 augmented copies for every 1 original image
for idx, row in tqdm(df_original.iterrows(), total=len(df_original)):
    img = Image.open(row['filepath']).convert('RGB')
    label = row['label']

    # 1. Add the original image
    expanded_images.append(base_transform(img))
    expanded_labels.append(label)

    # 2. Add 9 augmented clones (Total 10 per image -> 755 * 10 = 7550)
    for _ in range(9):
        expanded_images.append(matlab_simulator(img))
        expanded_labels.append(label)

# Convert to tensors
X_all = torch.stack(expanded_images)
y_all = torch.tensor(expanded_labels, dtype=torch.long)
print(f"Expansion complete. Total images: {len(X_all)}")

# ---------------------------------------------------------
# 3. The Fatal Flaw: Splitting AFTER Augmentation (80/20)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_all.numpy(), y_all.numpy(), test_size=0.20, random_state=42, stratify=y_all.numpy()
)
print(f"Data Partitioning: {len(X_train)} training, {len(X_test)} testing.")

class TensorDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# ---------------------------------------------------------
# 4. Train the Vision Transformer (ViT)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nTraining Vision Transformer (ViT) on 128x128 images...")

# Using a standard ViT modified for 128x128 resolution
model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=6, img_size=128).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # They used Adam with LR=0.001

# Train for 10 epochs (It will hit 90%+ rapidly due to the leaked clones)
for epoch in range(100):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/100 - Training Loss: {running_loss/len(train_loader):.4f}")

# ---------------------------------------------------------
# 5. Evaluate and Print the Inflated Metrics
# ---------------------------------------------------------
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        preds = torch.argmax(model(inputs), 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print("\n" + "="*50)
print("🏆 REPLICATED PAPER RESULTS (THE ILLUSION) 🏆")
print("="*50)
print(f"Testing Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
target_names = ['CHICKENPOX', 'COWPOX', 'HFMD', 'HEALTHY', 'MEASLES', 'MONKEYPOX']
print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
