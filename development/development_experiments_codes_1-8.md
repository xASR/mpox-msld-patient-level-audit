**CODE1:**  
import os  
import random  
import numpy as np  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader, Dataset  
from torchvision import transforms, models  
from sklearn.metrics import accuracy\_score, precision\_recall\_fscore\_support, roc\_auc\_score, confusion\_matrix  
from sklearn.utils.class\_weight import compute\_class\_weight  
from sklearn.model\_selection import train\_test\_split  
from PIL import Image  
from collections import defaultdict  
import matplotlib.pyplot as plt  
import seaborn as sns  
import warnings  
import gc

warnings.filterwarnings('ignore')

\# \==========================================  
\# 1\. DETERMINISM & CONFIGURATION  
\# \==========================================  
def set\_seed(seed=42):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual\_seed(seed)  
    torch.cuda.manual\_seed\_all(seed)  
    torch.backends.cudnn.benchmark \= False   
    torch.backends.cudnn.deterministic \= True

set\_seed(42)

BATCH\_SIZE \= 32  
EPOCHS \= 30  
WARMUP\_EPOCHS \= 3     
HEAD\_LR \= 3e-4        
BACKBONE\_LR \= 3e-5    
WEIGHT\_DECAY \= 1e-4  
DROPOUT\_RATE \= 0.5

device \= torch.device("cuda" if torch.cuda.is\_available() else "cpu")  
device\_type \= 'cuda' if torch.cuda.is\_available() else 'cpu'  
use\_amp \= device\_type \== 'cuda'  
print(f"Using compute device: {device} \| AMP Enabled: {use\_amp}")

\# \==========================================  
\# 2\. PATH FINDER & PATIENT GROUPING  
\# \==========================================  
DATA\_DIR \= None  
for root, dirs, files in os.walk('/kaggle/input'):  
    for d in dirs:  
        if d.lower() \== 'fold1':  
            DATA\_DIR \= os.path.join(root, d)  
            break  
    if DATA\_DIR: break

if DATA\_DIR is None:  
    raise FileNotFoundError("Could not find 'fold1' directory in /kaggle/input/")

subdirs \= os.listdir(DATA\_DIR)  
train\_folder \= next((d for d in subdirs if d.lower() \== 'train'), None)  
val\_folder \= next((d for d in subdirs if d.lower() in \['val', 'validation', 'valid'\]), None)  
test\_folder \= next((d for d in subdirs if d.lower() \== 'test'), None)

TRAIN\_DIR \= os.path.join(DATA\_DIR, train\_folder)  
VAL\_DIR \= os.path.join(DATA\_DIR, val\_folder)  
TEST\_DIR \= os.path.join(DATA\_DIR, test\_folder)

print("\\nExtracting and grouping patients...")  
classes \= sorted(os.listdir(TRAIN\_DIR))  
class\_to\_idx \= {c: i for i, c in enumerate(classes)}  
idx\_to\_class \= {i: c for c, i in class\_to\_idx.items()}  
NUM\_CLASSES \= len(classes)

patient\_data \= defaultdict(list)  
\# Pool all images to rebuild the dataset completely from scratch  
for split\_dir in \[TRAIN\_DIR, VAL\_DIR, TEST\_DIR\]:  
    for class\_name in classes:  
        class\_dir \= os.path.join(split\_dir, class\_name)  
        if not os.path.isdir(class\_dir): continue  
        for filename in os.listdir(class\_dir):  
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
                filepath \= os.path.join(class\_dir, filename)  
                try: patient\_id \= f"{class\_name}\_{filename.split('\_')\[1\]}"  
                except IndexError: patient\_id \= f"{class\_name}\_{filename}"  
                patient\_data\[patient\_id\].append((filepath, class\_to\_idx\[class\_name\]))

unique\_patients \= list(patient\_data.keys())  
patient\_labels \= \[patient\_data\[p\]\[0\]\[1\] for p in unique\_patients\]

\# \==========================================  
\# 3\. STRATIFIED 70/10/20 SPLIT  
\# \==========================================  
\# Split 1: 70% Train, 30% Temp (Val \+ Test)  
train\_pat, temp\_pat, train\_lbl, temp\_lbl \= train\_test\_split(  
    unique\_patients, patient\_labels, test\_size=0.30, stratify=patient\_labels, random\_state=42  
)

\# Split 2: From the 30% Temp, split into 1/3 Val (10% total) and 2/3 Test (20% total)  
val\_pat, test\_pat, \_, \_ \= train\_test\_split(  
    temp\_pat, temp\_lbl, test\_size=(2.0/3.0), stratify=temp\_lbl, random\_state=42  
)

train\_files, val\_files, test\_files \= \[\], \[\], \[\]  
for p in train\_pat: train\_files.extend(patient\_data\[p\])  
for p in val\_pat: val\_files.extend(patient\_data\[p\])  
for p in test\_pat: test\_files.extend(patient\_data\[p\])

print(f"Split Complete\! Patients \-\> Train: {len(train\_pat)} \| Val: {len(val\_pat)} \| Test: {len(test\_pat)}")  
print(f"Images Mapping  \-\> Train: {len(train\_files)} \| Val: {len(val\_files)} \| Test: {len(test\_files)}")

class PatientLevelDataset(Dataset):  
    def \_\_init\_\_(self, file\_list, transform=None):  
        self.file\_list, self.transform \= file\_list, transform  
    def \_\_len\_\_(self): return len(self.file\_list)  
    def \_\_getitem\_\_(self, idx):  
        filepath, label \= self.file\_list\[idx\]  
        image \= Image.open(filepath).convert('RGB')  
        if self.transform: image \= self.transform(image)  
        return image, label

\# \==========================================  
\# 4\. PREPROCESSING & AUGMENTATION  
\# \==========================================  
transforms\_train \= transforms.Compose(\[  
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomRotation(15),  
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  
    transforms.ToTensor(),  
    transforms.Normalize(\[0.485, 0.456, 0.406\], \[0.229, 0.224, 0.225\])  
\])

transforms\_eval \= transforms.Compose(\[  
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(\[0.485, 0.456, 0.406\], \[0.229, 0.224, 0.225\])  
\])

train\_loader \= DataLoader(PatientLevelDataset(train\_files, transforms\_train), batch\_size=BATCH\_SIZE, shuffle=True, num\_workers=2, drop\_last=True)  
val\_loader \= DataLoader(PatientLevelDataset(val\_files, transforms\_eval), batch\_size=BATCH\_SIZE, shuffle=False, num\_workers=2)  
test\_loader \= DataLoader(PatientLevelDataset(test\_files, transforms\_eval), batch\_size=BATCH\_SIZE, shuffle=False, num\_workers=2)

train\_targets \= \[item\[1\] for item in train\_files\]  
class\_weights \= compute\_class\_weight(class\_weight='balanced', classes=np.unique(train\_targets), y=train\_targets)  
class\_weights\_tensor \= torch.tensor(class\_weights, dtype=torch.float).to(device)

\# Loss with Label Smoothing  
criterion \= nn.CrossEntropyLoss(weight=class\_weights\_tensor, label\_smoothing=0.05)

\# \==========================================  
\# 5\. MODEL INITIALIZATION (ResNet50)  
\# \==========================================

print("\\nInitializing ResNet50 with ImageNet Weights...")  
model \= models.resnet50(weights=models.ResNet50\_Weights.IMAGENET1K\_V1)  
model.fc \= nn.Sequential(  
    nn.Dropout(DROPOUT\_RATE),   
    nn.Linear(model.fc.in\_features, NUM\_CLASSES)  
)

head\_params \= model.fc.parameters()  
backbone\_params \= \[p for n, p in model.named\_parameters() if 'fc' not in n\]

\# Warmup: Freeze Backbone  
for p in backbone\_params:  
    p.requires\_grad \= False

model \= model.to(device)

optimizer \= optim.Adam(\[  
    {'params': backbone\_params, 'lr': BACKBONE\_LR},   
    {'params': head\_params, 'lr': HEAD\_LR}  
\], weight\_decay=WEIGHT\_DECAY)

scheduler \= torch.optim.lr\_scheduler.CosineAnnealingLR(optimizer, T\_max=EPOCHS)  
scaler \= torch.amp.GradScaler(device\_type, enabled=use\_amp) 

\# \==========================================  
\# 6\. TRAINING LOOP (Saving on Best Val F1)  
\# \==========================================  
print("\\n" \+ "="\*50)  
print("🚀 COMMENCING TRAINING")  
print("="\*50)

best\_val\_f1 \= 0.0  
best\_val\_metrics \= {}

for epoch in range(EPOCHS):  
    if epoch \== WARMUP\_EPOCHS:  
        print("  \[Warmup Complete: Unfreezing Backbone for Full Fine-Tuning\]")  
        for p in backbone\_params:   
            p.requires\_grad \= True

    model.train()  
    running\_loss \= 0.0  
    for images, labels in train\_loader:  
        images, labels \= images.to(device), labels.to(device)  
        optimizer.zero\_grad()  
          
        with torch.amp.autocast(device\_type, enabled=use\_amp):  
            loss \= criterion(model(images), labels)  
              
        scaler.scale(loss).backward()  
        scaler.step(optimizer)  
        scaler.update()  
        running\_loss \+= loss.item() \* images.size(0)  
      
    train\_loss \= running\_loss / len(train\_loader.dataset)  
    scheduler.step()

    \# Validation Phase  
    model.eval()  
    val\_loss \= 0.0  
    all\_val\_labels, all\_val\_preds \= \[\], \[\]  
      
    with torch.no\_grad():  
        for images, labels in val\_loader:  
            images, labels \= images.to(device), labels.to(device)  
            with torch.amp.autocast(device\_type, enabled=use\_amp):  
                outputs \= model(images)  
                loss \= criterion(outputs, labels)  
              
            val\_loss \+= loss.item() \* images.size(0)  
            preds \= torch.argmax(outputs, dim=1)  
            all\_val\_labels.extend(labels.cpu().numpy())  
            all\_val\_preds.extend(preds.cpu().numpy())  
              
    val\_loss /= len(val\_loader.dataset)  
    val\_f1 \= precision\_recall\_fscore\_support(all\_val\_labels, all\_val\_preds, average='macro', zero\_division=0)\[2\]  
    val\_acc \= accuracy\_score(all\_val\_labels, all\_val\_preds)  
      
    print(f"Epoch {epoch+1:02d}/{EPOCHS} \| Train Loss: {train\_loss:.4f} \| Val Loss: {val\_loss:.4f} \| Val F1: {val\_f1:.4f} \| Val Acc: {val\_acc:.4f}")  
      
    \# Save Best Model based on F1-Score  
    if val\_f1 \> best\_val\_f1:  
        best\_val\_f1 \= val\_f1  
        best\_val\_metrics \= {'Epoch': epoch+1, 'Loss': val\_loss, 'Acc': val\_acc, 'F1': val\_f1}  
        torch.save(model.state\_dict(), 'best\_resnet50\_mpox.pth')  
        print(f"   🌟 New best model saved\! (Val F1: {val\_f1:.4f})")

\# \==========================================  
\# 7\. FINAL EVALUATION ON UNTOUCHED TEST SET  
\# \==========================================  
print("\\n" \+ "="\*50)  
print("🔍 EVALUATING BEST MODEL ON UNTOUCHED TEST SET")  
print("="\*50)

model.load\_state\_dict(torch.load('best\_resnet50\_mpox.pth'))  
model.eval()

test\_labels, test\_preds, test\_probs \= \[\], \[\], \[\]

with torch.no\_grad():  
    for images, labels in test\_loader:  
        images, labels \= images.to(device), labels.to(device)  
        with torch.amp.autocast(device\_type, enabled=use\_amp):  
            outputs \= model(images)  
            probs \= torch.softmax(outputs, dim=1)  
          
        preds \= torch.argmax(outputs, dim=1)  
        test\_probs.extend(probs.cpu().float().numpy())  
        test\_preds.extend(preds.cpu().numpy())  
        test\_labels.extend(labels.cpu().numpy())

\# Calculate Final Metrics  
test\_acc \= accuracy\_score(test\_labels, test\_preds)  
test\_prec, test\_rec, test\_f1, \_ \= precision\_recall\_fscore\_support(test\_labels, test\_preds, average='macro', zero\_division=0)

try:  
    test\_auc \= roc\_auc\_score(test\_labels, test\_probs, multi\_class='ovr', average='macro')  
except ValueError:  
    test\_auc \= float('nan')

\# Output 1: Validation Summary  
print("\\n\[1\] VALIDATION PERFORMANCE (Model Selected From)")  
print(f"    Best Epoch : {best\_val\_metrics\['Epoch'\]}")  
print(f"    Val Loss   : {best\_val\_metrics\['Loss'\]:.4f}")  
print(f"    Val Acc    : {best\_val\_metrics\['Acc'\]:.4f}")  
print(f"    Val F1     : {best\_val\_metrics\['F1'\]:.4f}")

\# Output 2: Final Test Metrics  
print("\\n\[2\] FINAL TEST METRICS (Real-World Baseline)")  
print(f"    Accuracy   : {test\_acc \* 100:.2f}%")  
print(f"    Precision  : {test\_prec:.4f}")  
print(f"    Recall     : {test\_rec:.4f}")  
print(f"    F1-Score   : {test\_f1:.4f}")  
print(f"    AUC (OVR)  : {test\_auc:.4f}")

\# Output 3: Comparison  
print("\\n\[3\] COMPARISON SUMMARY")  
print(f"    Validation F1 : {best\_val\_f1:.4f}")  
print(f"    Test F1       : {test\_f1:.4f}")  
diff \= test\_f1 \- best\_val\_f1  
status \= "Stable (Good Generalization)" if abs(diff) \< 0.05 else ("Overfitting" if diff \< 0 else "Underfitting/Variance")  
print(f"    Difference    : {diff:+.4f} \-\> {status}")

\# \==========================================  
\# 8\. CONFUSION MATRIX VISUALIZATION  
\# \==========================================  
cm \= confusion\_matrix(test\_labels, test\_preds)  
class\_names \= \[idx\_to\_class\[i\] for i in range(NUM\_CLASSES)\]

plt.figure(figsize=(10, 8))  
sns.set\_theme(style="white", context="paper", font\_scale=1.2)  
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class\_names, yticklabels=class\_names,   
            cbar=False, annot\_kws={"size": 14, "weight": "bold"})  
plt.title('Test Set Confusion Matrix: ResNet50 (Strict Patient Split)', fontsize=16, pad=20, weight='bold')  
plt.xlabel('Predicted Label', fontsize=14, weight='bold')  
plt.ylabel('True Label', fontsize=14, weight='bold')  
plt.xticks(rotation=45)  
plt.yticks(rotation=0)  
plt.tight\_layout()  
plt.savefig('resnet50\_confusion\_matrix.png', dpi=300)  
plt.show()

print("\\n✅ Run Complete. Confusion Matrix saved as 'resnet50\_confusion\_matrix.png'.")

**CODE 1 Output :**   
Using compute device: cuda \| AMP Enabled: True

Extracting and grouping patients...  
Split Complete\! Patients \-\> Train: 366 \| Val: 52 \| Test: 106  
Images Mapping  \-\> Train: 509 \| Val: 87 \| Test: 159

Initializing ResNet50 with ImageNet Weights...  
Downloading: "[https\://download.pytorch.org/models/resnet50-0676ba61.pth](https://download.pytorch.org/models/resnet50-0676ba61.pth)" to /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth

100%\|██████████\| 97.8M/97.8M \[00:00\<00:00, 193MB/s\] 

\==================================================  
🚀 COMMENCING TRAINING  
\==================================================  
Epoch 01/30 \| Train Loss: 1.7411 \| Val Loss: 1.7406 \| Val F1: 0.2626 \| Val Acc: 0.2644  
   🌟 New best model saved\! (Val F1: 0.2626)  
Epoch 02/30 \| Train Loss: 1.6211 \| Val Loss: 1.6009 \| Val F1: 0.4751 \| Val Acc: 0.5632  
   🌟 New best model saved\! (Val F1: 0.4751)  
Epoch 03/30 \| Train Loss: 1.4952 \| Val Loss: 1.5393 \| Val F1: 0.5227 \| Val Acc: 0.5172  
   🌟 New best model saved\! (Val F1: 0.5227)  
  \[Warmup Complete: Unfreezing Backbone for Full Fine-Tuning\]  
Epoch 04/30 \| Train Loss: 1.3582 \| Val Loss: 1.2911 \| Val F1: 0.6447 \| Val Acc: 0.6322  
   🌟 New best model saved\! (Val F1: 0.6447)  
Epoch 05/30 \| Train Loss: 1.0368 \| Val Loss: 1.1007 \| Val F1: 0.6457 \| Val Acc: 0.6552  
   🌟 New best model saved\! (Val F1: 0.6457)  
Epoch 06/30 \| Train Loss: 0.8202 \| Val Loss: 0.9705 \| Val F1: 0.6748 \| Val Acc: 0.7126  
   🌟 New best model saved\! (Val F1: 0.6748)  
Epoch 07/30 \| Train Loss: 0.6484 \| Val Loss: 0.9344 \| Val F1: 0.6687 \| Val Acc: 0.7011  
Epoch 08/30 \| Train Loss: 0.5809 \| Val Loss: 0.9380 \| Val F1: 0.6587 \| Val Acc: 0.6897  
Epoch 09/30 \| Train Loss: 0.5045 \| Val Loss: 0.9442 \| Val F1: 0.6740 \| Val Acc: 0.7241  
Epoch 10/30 \| Train Loss: 0.4764 \| Val Loss: 0.9389 \| Val F1: 0.6737 \| Val Acc: 0.7241  
Epoch 11/30 \| Train Loss: 0.4518 \| Val Loss: 0.9274 \| Val F1: 0.6851 \| Val Acc: 0.7241  
   🌟 New best model saved\! (Val F1: 0.6851)  
Epoch 12/30 \| Train Loss: 0.4075 \| Val Loss: 0.9438 \| Val F1: 0.6822 \| Val Acc: 0.7356  
Epoch 13/30 \| Train Loss: 0.4225 \| Val Loss: 0.9663 \| Val F1: 0.6856 \| Val Acc: 0.7241  
   🌟 New best model saved\! (Val F1: 0.6856)  
Epoch 14/30 \| Train Loss: 0.4213 \| Val Loss: 0.9347 \| Val F1: 0.6716 \| Val Acc: 0.7126  
Epoch 15/30 \| Train Loss: 0.3818 \| Val Loss: 0.9285 \| Val F1: 0.6805 \| Val Acc: 0.7356  
Epoch 16/30 \| Train Loss: 0.4022 \| Val Loss: 0.9394 \| Val F1: 0.6959 \| Val Acc: 0.7471  
   🌟 New best model saved\! (Val F1: 0.6959)  
Epoch 17/30 \| Train Loss: 0.3699 \| Val Loss: 0.9580 \| Val F1: 0.6898 \| Val Acc: 0.7471  
Epoch 18/30 \| Train Loss: 0.3618 \| Val Loss: 0.9888 \| Val F1: 0.6757 \| Val Acc: 0.7241  
Epoch 19/30 \| Train Loss: 0.3661 \| Val Loss: 0.9922 \| Val F1: 0.6692 \| Val Acc: 0.7126  
Epoch 20/30 \| Train Loss: 0.3584 \| Val Loss: 0.9789 \| Val F1: 0.6817 \| Val Acc: 0.7241  
Epoch 21/30 \| Train Loss: 0.3670 \| Val Loss: 0.9625 \| Val F1: 0.6825 \| Val Acc: 0.7356  
Epoch 22/30 \| Train Loss: 0.3498 \| Val Loss: 0.9615 \| Val F1: 0.6825 \| Val Acc: 0.7356  
Epoch 23/30 \| Train Loss: 0.3642 \| Val Loss: 0.9497 \| Val F1: 0.6886 \| Val Acc: 0.7356  
Epoch 24/30 \| Train Loss: 0.3482 \| Val Loss: 0.9658 \| Val F1: 0.6825 \| Val Acc: 0.7356  
Epoch 25/30 \| Train Loss: 0.3429 \| Val Loss: 0.9607 \| Val F1: 0.6825 \| Val Acc: 0.7356  
Epoch 26/30 \| Train Loss: 0.3598 \| Val Loss: 0.9562 \| Val F1: 0.6757 \| Val Acc: 0.7241  
Epoch 27/30 \| Train Loss: 0.3536 \| Val Loss: 0.9586 \| Val F1: 0.6817 \| Val Acc: 0.7241  
Epoch 28/30 \| Train Loss: 0.3461 \| Val Loss: 0.9599 \| Val F1: 0.6886 \| Val Acc: 0.7356  
Epoch 29/30 \| Train Loss: 0.3564 \| Val Loss: 0.9586 \| Val F1: 0.6886 \| Val Acc: 0.7356  
Epoch 30/30 \| Train Loss: 0.3534 \| Val Loss: 0.9623 \| Val F1: 0.6757 \| Val Acc: 0.7241

\==================================================  
🔍 EVALUATING BEST MODEL ON UNTOUCHED TEST SET  
\==================================================

\[1\] VALIDATION PERFORMANCE (Model Selected From)  
    Best Epoch : 16  
    Val Loss   : 0.9394  
    Val Acc    : 0.7471  
    Val F1     : 0.6959

\[2\] FINAL TEST METRICS (Real-World Baseline)  
    Accuracy   : 79.87%  
    Precision  : 0.7924  
    Recall     : 0.7793  
    F1-Score   : 0.7846  
    AUC (OVR)  : 0.9562

\[3\] COMPARISON SUMMARY  
    Validation F1 : 0.6959  
    Test F1       : 0.7846  
    Difference    : \+0.0887 \-\> Underfitting/Variance

![][image1]  
✅ Run Complete. Confusion Matrix saved as 'resnet50\_confusion\_matrix.png'

**CODE 2:**  
import os  
import random  
import numpy as np  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader, Dataset  
from torchvision import transforms, models  
from sklearn.metrics import accuracy\_score, precision\_recall\_fscore\_support, roc\_auc\_score  
from sklearn.utils.class\_weight import compute\_class\_weight  
from sklearn.model\_selection import StratifiedKFold  
from PIL import Image  
from collections import defaultdict  
import pandas as pd  
import gc

*\# \==========================================*  
*\# 1\. DETERMINISM & CONFIGURATION*  
*\# \==========================================*  
def set\_seed(seed\=42):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual\_seed(seed)  
    torch.cuda.manual\_seed\_all(seed)  
    torch.backends.cudnn.benchmark \= False   
    torch.backends.cudnn.deterministic \= True

set\_seed(42)

MODELS\_TO\_TEST \= \[  
    'alexnet', 'vgg19', 'resnet50', 'resnet101', 'densenet121',   
    'googlenet', 'inception\_v3', 'mobilenet\_v2', 'mobilenet\_v3\_large',   
    'efficientnet\_b0', 'efficientnet\_b4', 'efficientnet\_b7'  
\]

BATCH\_SIZE \= 32  
EPOCHS \= 30  
WARMUP\_EPOCHS \= 3     
HEAD\_LR \= 3e-4        
BACKBONE\_LR \= 3e-5    
WEIGHT\_DECAY \= 1e-4  
DROPOUT\_RATE \= 0.5  
K\_FOLDS \= 5         

device \= torch.device("cuda" if torch.cuda.is\_available() else "cpu")  
device\_type \= 'cuda' if torch.cuda.is\_available() else 'cpu'  
use\_amp \= device\_type \== 'cuda'  
print(f"Using compute device: {device} \| AMP Enabled: {use\_amp}")

*\# \==========================================*  
*\# 2\. PATH FINDER & PATIENT GROUPING*  
*\# \==========================================*  
DATA\_DIR \= None  
for root, dirs, files in os.walk('/kaggle/input'):  
    for d in dirs:  
        if d.lower() \== 'fold1':  
            DATA\_DIR \= os.path.join(root, d)  
            break  
    if DATA\_DIR: break

subdirs \= os.listdir(DATA\_DIR)  
train\_folder \= next((d for d in subdirs if d.lower() \== 'train'), None)  
val\_folder \= next((d for d in subdirs if d.lower() in \['val', 'validation', 'valid'\]), None)  
test\_folder \= next((d for d in subdirs if d.lower() \== 'test'), None)

TRAIN\_DIR \= os.path.join(DATA\_DIR, train\_folder)  
VAL\_DIR \= os.path.join(DATA\_DIR, val\_folder)  
TEST\_DIR \= os.path.join(DATA\_DIR, test\_folder)

print("\\nExtracting all patients for Stratified Cross-Validation...")  
classes \= sorted(os.listdir(TRAIN\_DIR))  
class\_to\_idx \= {c: i for i, c in enumerate(classes)}  
NUM\_CLASSES \= len(classes)

patient\_data \= defaultdict(list)  
for split\_dir in \[TRAIN\_DIR, VAL\_DIR, TEST\_DIR\]:  
    for class\_name in classes:  
        class\_dir \= os.path.join(split\_dir, class\_name)  
        if not os.path.isdir(class\_dir): continue  
        for filename in os.listdir(class\_dir):  
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
                filepath \= os.path.join(class\_dir, filename)  
                try: patient\_id \= f"{class\_name}\_{filename.split('\_')\[1\]}"  
                except IndexError: patient\_id \= f"{class\_name}\_{filename}"  
                patient\_data\[patient\_id\].append((filepath, class\_to\_idx\[class\_name\]))

unique\_patients \= list(patient\_data.keys())  
patient\_labels \= \[\]

for p in unique\_patients:  
    primary\_label \= patient\_data\[p\]\[0\]\[1\]   
    patient\_labels.append(primary\_label)

unique\_patients \= np.array(unique\_patients)  
patient\_labels \= np.array(patient\_labels)

print(f"Total Unique Patients Extracted: {len(unique\_patients)}")

class PatientLevelDataset(Dataset):  
    def \_\_init\_\_(self, file\_list, transform\=None):  
        self.file\_list, self.transform \= file\_list, transform  
    def \_\_len\_\_(self): return len(self.file\_list)  
    def \_\_getitem\_\_(self, idx):  
        filepath, label \= self.file\_list\[idx\]  
        image \= Image.open(filepath).convert('RGB')  
        if self.transform: image \= self.transform(image)  
        return image, label

*\# \==========================================*  
*\# 3\. DYNAMIC MODEL ARCHITECTURE BUILDER*  
*\# \==========================================*  
def build\_model(model\_name):  
    if model\_name \== 'alexnet':  
        model \= models.alexnet(weights\=models.AlexNet\_Weights.IMAGENET1K\_V1)  
        model.classifier\[6\] \= nn.Sequential(nn.Dropout(DROPOUT\_RATE), nn.Linear(model.classifier\[6\].in\_features, NUM\_CLASSES))  
        head\_params, backbone\_params \= model.classifier.parameters(), model.features.parameters()  
          
    elif model\_name \== 'vgg19':  
        model \= models.vgg19(weights\=models.VGG19\_Weights.IMAGENET1K\_V1)  
        model.classifier\[6\] \= nn.Sequential(nn.Dropout(DROPOUT\_RATE), nn.Linear(model.classifier\[6\].in\_features, NUM\_CLASSES))  
        head\_params, backbone\_params \= model.classifier.parameters(), model.features.parameters()

    elif 'resnet' in model\_name:  
        model \= models.resnet50(weights\=models.ResNet50\_Weights.IMAGENET1K\_V1) if model\_name \== 'resnet50' else models.resnet101(weights\=models.ResNet101\_Weights.IMAGENET1K\_V1)  
        model.fc \= nn.Sequential(nn.Dropout(DROPOUT\_RATE), nn.Linear(model.fc.in\_features, NUM\_CLASSES))  
        head\_params \= model.fc.parameters()  
        backbone\_params \= \[p for n, p in model.named\_parameters() if 'fc' not in n\]

    elif model\_name \== 'densenet121':  
        model \= models.densenet121(weights\=models.DenseNet121\_Weights.IMAGENET1K\_V1)  
        model.classifier \= nn.Sequential(nn.Dropout(DROPOUT\_RATE), nn.Linear(model.classifier.in\_features, NUM\_CLASSES))  
        head\_params, backbone\_params \= model.classifier.parameters(), model.features.parameters()

    elif model\_name \== 'googlenet':  
        model \= models.googlenet(weights\=models.GoogLeNet\_Weights.IMAGENET1K\_V1, transform\_input\=False)  
        model.fc \= nn.Sequential(nn.Dropout(DROPOUT\_RATE), nn.Linear(model.fc.in\_features, NUM\_CLASSES))  
        model.aux\_logits \= False   
        head\_params \= model.fc.parameters()  
        backbone\_params \= \[p for n, p in model.named\_parameters() if 'fc' not in n\]

    elif model\_name \== 'inception\_v3':  
        model \= models.inception\_v3(weights\=models.Inception\_V3\_Weights.IMAGENET1K\_V1)  
        model.AuxLogits.fc \= nn.Linear(model.AuxLogits.fc.in\_features, NUM\_CLASSES)  
        model.fc \= nn.Sequential(nn.Dropout(DROPOUT\_RATE), nn.Linear(model.fc.in\_features, NUM\_CLASSES))  
        head\_params \= list(model.fc.parameters()) \+ list(model.AuxLogits.fc.parameters())  
        backbone\_params \= \[p for n, p in model.named\_parameters() if 'fc' not in n\]

    elif 'mobilenet' in model\_name:  
        if model\_name \== 'mobilenet\_v2':  
            model \= models.mobilenet\_v2(weights\=models.MobileNet\_V2\_Weights.IMAGENET1K\_V1)  
            model.classifier\[1\] \= nn.Sequential(nn.Dropout(DROPOUT\_RATE), nn.Linear(model.classifier\[1\].in\_features, NUM\_CLASSES))  
        else:  
            model \= models.mobilenet\_v3\_large(weights\=models.MobileNet\_V3\_Large\_Weights.IMAGENET1K\_V1)  
            model.classifier\[3\] \= nn.Sequential(nn.Dropout(DROPOUT\_RATE), nn.Linear(model.classifier\[3\].in\_features, NUM\_CLASSES))  
        head\_params, backbone\_params \= model.classifier.parameters(), model.features.parameters()

    elif 'efficientnet' in model\_name:  
        if model\_name \== 'efficientnet\_b0': weights \= models.EfficientNet\_B0\_Weights.IMAGENET1K\_V1  
        elif model\_name \== 'efficientnet\_b4': weights \= models.EfficientNet\_B4\_Weights.IMAGENET1K\_V1  
        elif model\_name \== 'efficientnet\_b7': weights \= models.EfficientNet\_B7\_Weights.IMAGENET1K\_V1  
        model \= getattr(models, model\_name)(weights\=weights)  
        model.classifier \= nn.Sequential(nn.Dropout(DROPOUT\_RATE, inplace\=True), nn.Linear(model.classifier\[1\].in\_features, NUM\_CLASSES))  
        head\_params, backbone\_params \= model.classifier.parameters(), model.features.parameters()

    for p in backbone\_params:  
        p.requires\_grad \= False

    return model.to(device), backbone\_params, head\_params

*\# \==========================================*  
*\# 4\. MASTER STRATIFIED K-FOLD EXPERIMENT*  
*\# \==========================================*  
experiment\_results \= \[\]  
skf \= StratifiedKFold(n\_splits\=K\_FOLDS, shuffle\=True, random\_state\=42)

for active\_model in MODELS\_TO\_TEST:  
    print("\\n" \+ "="\*70)  
    print(f"🚀 INITIALIZING RUN: {active\_model.upper()}")  
    print("="\*70)  
      
    image\_size \= 299 if active\_model \== 'inception\_v3' else 224  
      
    transforms\_train \= transforms.Compose(\[  
        transforms.RandomResizedCrop(image\_size, scale\=(0.8, 1.0)),  
        transforms.RandomHorizontalFlip(p\=0.5),  
        transforms.RandomRotation(15),  
        transforms.ColorJitter(brightness\=0.1, contrast\=0.1),  
        transforms.ToTensor(),  
        transforms.Normalize(\[0.485, 0.456, 0.406\], \[0.229, 0.224, 0.225\])  
    \])  
    transforms\_eval \= transforms.Compose(\[  
        transforms.Resize((image\_size, image\_size)),  
        transforms.ToTensor(),  
        transforms.Normalize(\[0.485, 0.456, 0.406\], \[0.229, 0.224, 0.225\])  
    \])  
      
    fold\_metrics \= \[\]

    for fold, (train\_idx, val\_idx) in enumerate(skf.split(unique\_patients, patient\_labels)):  
        print(f"\\n\--- Fold {fold \+ 1}/{K\_FOLDS} \---")  
          
        train\_patients, val\_patients \= unique\_patients\[train\_idx\], unique\_patients\[val\_idx\]  
        train\_files, val\_files \= \[\], \[\]  
        for p in train\_patients: train\_files.extend(patient\_data\[p\])  
        for p in val\_patients: val\_files.extend(patient\_data\[p\])

        train\_loader \= DataLoader(PatientLevelDataset(train\_files, transforms\_train), batch\_size\=BATCH\_SIZE, shuffle\=True, num\_workers\=2, drop\_last\=True)  
        val\_loader \= DataLoader(PatientLevelDataset(val\_files, transforms\_eval), batch\_size\=BATCH\_SIZE, shuffle\=False, num\_workers\=2)

        train\_targets \= \[item\[1\] for item in train\_files\]  
        class\_weights \= compute\_class\_weight(class\_weight\='balanced', classes\=np.unique(train\_targets), y\=train\_targets)  
        class\_weights\_tensor \= torch.tensor(class\_weights, dtype\=torch.float).to(device)  
          
        criterion \= nn.CrossEntropyLoss(weight\=class\_weights\_tensor, label\_smoothing\=0.05)

        model, backbone, head \= build\_model(active\_model)  
        optimizer \= optim.Adam(\[{'params': backbone, 'lr': BACKBONE\_LR}, {'params': head, 'lr': HEAD\_LR}\], weight\_decay\=WEIGHT\_DECAY)  
          
        scheduler \= torch.optim.lr\_scheduler.CosineAnnealingLR(optimizer, T\_max\=EPOCHS)  
        scaler \= torch.amp.GradScaler(device\_type, enabled\=use\_amp) 

        for epoch in range(EPOCHS):  
            if epoch \== WARMUP\_EPOCHS:  
                for p in backbone: p.requires\_grad \= True

            model.train()  
            for images, labels in train\_loader:  
                images, labels \= images.to(device), labels.to(device)  
                optimizer.zero\_grad()  
                  
                with torch.amp.autocast(device\_type, enabled\=use\_amp):  
                    if active\_model \== 'inception\_v3':  
                        outputs, aux\_outputs \= model(images)  
                        loss \= criterion(outputs, labels) \+ 0.4 \* criterion(aux\_outputs, labels)  
                    else:  
                        loss \= criterion(model(images), labels)  
                      
                scaler.scale(loss).backward()  
                scaler.step(optimizer)  
                scaler.update()  
              
            scheduler.step()

        *\# Fold Evaluation*  
        model.eval()  
        all\_labels, all\_probs \= \[\], \[\]  
        with torch.no\_grad():  
            for images, labels in val\_loader:  
                images, labels \= images.to(device), labels.to(device)  
                with torch.amp.autocast(device\_type, enabled\=use\_amp):  
                    probs \= torch.softmax(model(images), dim\=1)  
                  
                all\_probs.extend(probs.cpu().float().numpy())  
                all\_labels.extend(labels.cpu().numpy())

        best\_preds \= np.argmax(all\_probs, axis\=1)  
          
        acc \= accuracy\_score(all\_labels, best\_preds)  
        prec, rec, f1, \_ \= precision\_recall\_fscore\_support(all\_labels, best\_preds, average\='macro', zero\_division\=0)  
        try: auc \= roc\_auc\_score(all\_labels, all\_probs, multi\_class\='ovr', average\='macro')  
        except ValueError: auc \= float('nan')

        print(f"  Acc: {acc:.4f} \| F1: {f1:.4f} \| AUC: {auc:.4f}")  
        fold\_metrics.append(\[acc, prec, rec, f1, auc\])  
          
        *\# Aggressive memory management*  
        del model, optimizer, scaler  
        gc.collect()  
        torch.cuda.empty\_cache()

    avg\_metrics \= np.mean(fold\_metrics, axis\=0)  
    experiment\_results.append({  
        'Model': active\_model.upper(),  
        'Accuracy': avg\_metrics\[0\] \* 100,  
        'Precision': avg\_metrics\[1\],  
        'Recall': avg\_metrics\[2\],  
        'F1-Score': avg\_metrics\[3\],  
        'AUC': avg\_metrics\[4\]  
    })

*\# \==========================================*  
*\# 5\. FINAL RESULTS SUMMARY*  
*\# \==========================================*  
print("\\n" \+ "="\*90)  
print("🏆 MASTER ABLATION STUDY RESULTS (STRATIFIED PATIENT CV)")  
print("="\*90)  
results\_df \= pd.DataFrame(experiment\_results).round(4)  
results\_df \= results\_df.sort\_values(by\='Accuracy', ascending\=False).reset\_index(drop\=True)  
print(results\_df.to\_markdown())  
print("="\*90)  
**CODE 2 Output :**  
`Using compute device: cuda | AMP Enabled: True`

`Extracting all patients for Stratified Cross-Validation...`  
`Total Unique Patients Extracted: 524`

`======================================================================`  
`🚀 INITIALIZING RUN: ALEXNET`  
`======================================================================`

`--- Fold 1/5 ---`  
`Downloading: "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth" to /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`

`100%|██████████| 233M/233M [00:01<00:00, 190MB/s]`

 `Acc: 0.6266 | F1: 0.6225 | AUC: 0.9245`

`--- Fold 2/5 ---`  
  `Acc: 0.6918 | F1: 0.6743 | AUC: 0.9422`

`--- Fold 3/5 ---`  
  `Acc: 0.6748 | F1: 0.6563 | AUC: 0.9113`

`--- Fold 4/5 ---`  
  `Acc: 0.7285 | F1: 0.7174 | AUC: 0.9499`

`--- Fold 5/5 ---`  
  `Acc: 0.6496 | F1: 0.6325 | AUC: 0.9007`

`======================================================================`  
`🚀 INITIALIZING RUN: VGG19`  
`======================================================================`

`--- Fold 1/5 ---`  
`Downloading: "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth" to /root/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth`

`100%|██████████| 548M/548M [00:02<00:00, 245MB/s]`

 `Acc: 0.6646 | F1: 0.6361 | AUC: 0.9146`

`--- Fold 2/5 ---`  
  `Acc: 0.6712 | F1: 0.6364 | AUC: 0.9330`

`--- Fold 3/5 ---`  
  `Acc: 0.6319 | F1: 0.5969 | AUC: 0.8829`

`--- Fold 4/5 ---`  
  `Acc: 0.7285 | F1: 0.7397 | AUC: 0.9528`

`--- Fold 5/5 ---`  
  `Acc: 0.7664 | F1: 0.7547 | AUC: 0.9498`

`======================================================================`  
`🚀 INITIALIZING RUN: RESNET50`  
`======================================================================`

`--- Fold 1/5 ---`  
`Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`

`100%|██████████| 97.8M/97.8M [00:01<00:00, 52.9MB/s]`

 `Acc: 0.7595 | F1: 0.7286 | AUC: 0.9533`

`--- Fold 2/5 ---`  
  `Acc: 0.8151 | F1: 0.7899 | AUC: 0.9592`

`--- Fold 3/5 ---`  
  `Acc: 0.7485 | F1: 0.7382 | AUC: 0.9358`

`--- Fold 4/5 ---`  
  `Acc: 0.7947 | F1: 0.7943 | AUC: 0.9726`

`--- Fold 5/5 ---`  
  `Acc: 0.8321 | F1: 0.8328 | AUC: 0.9683`

`======================================================================`  
`🚀 INITIALIZING RUN: RESNET101`  
`======================================================================`

`--- Fold 1/5 ---`  
`Downloading: "https://download.pytorch.org/models/resnet101-63fe2227.pth" to /root/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth`

`100%|██████████| 171M/171M [00:00<00:00, 239MB/s]`

 `Acc: 0.7722 | F1: 0.7369 | AUC: 0.9570`

`--- Fold 2/5 ---`  
  `Acc: 0.7877 | F1: 0.7726 | AUC: 0.9644`

`--- Fold 3/5 ---`  
  `Acc: 0.7301 | F1: 0.7101 | AUC: 0.9388`

`--- Fold 4/5 ---`  
  `Acc: 0.7748 | F1: 0.7790 | AUC: 0.9688`

`--- Fold 5/5 ---`  
  `Acc: 0.8175 | F1: 0.8253 | AUC: 0.9726`

`======================================================================`  
`🚀 INITIALIZING RUN: DENSENET121`  
`======================================================================`

`--- Fold 1/5 ---`  
`Downloading: "https://download.pytorch.org/models/densenet121-a639ec97.pth" to /root/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth`

`100%|██████████| 30.8M/30.8M [00:00<00:00, 198MB/s]`

 `Acc: 0.5759 | F1: 0.5472 | AUC: 0.9027`

`--- Fold 2/5 ---`  
  `Acc: 0.5479 | F1: 0.5466 | AUC: 0.8774`

`--- Fold 3/5 ---`  
  `Acc: 0.6748 | F1: 0.6632 | AUC: 0.9334`

`--- Fold 4/5 ---`  
  `Acc: 0.6225 | F1: 0.5919 | AUC: 0.9450`

`--- Fold 5/5 ---`  
  `Acc: 0.6715 | F1: 0.6057 | AUC: 0.9334`

`======================================================================`  
`🚀 INITIALIZING RUN: GOOGLENET`  
`======================================================================`

`--- Fold 1/5 ---`  
`Downloading: "https://download.pytorch.org/models/googlenet-1378be20.pth" to /root/.cache/torch/hub/checkpoints/googlenet-1378be20.pth`

`100%|██████████| 49.7M/49.7M [00:00<00:00, 215MB/s]`

 `Acc: 0.6899 | F1: 0.6700 | AUC: 0.9432`

`--- Fold 2/5 ---`  
  `Acc: 0.6849 | F1: 0.6497 | AUC: 0.9364`

`--- Fold 3/5 ---`  
  `Acc: 0.6564 | F1: 0.6554 | AUC: 0.9221`

`--- Fold 4/5 ---`  
  `Acc: 0.7285 | F1: 0.7358 | AUC: 0.9641`

`--- Fold 5/5 ---`  
  `Acc: 0.7737 | F1: 0.7509 | AUC: 0.9629`

`======================================================================`  
`🚀 INITIALIZING RUN: INCEPTION_V3`  
`======================================================================`

`--- Fold 1/5 ---`  
`Downloading: "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth" to /root/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth`

`100%|██████████| 104M/104M [00:00<00:00, 237MB/s]` 

 `Acc: 0.6962 | F1: 0.6621 | AUC: 0.9439`

`--- Fold 2/5 ---`  
  `Acc: 0.8082 | F1: 0.7845 | AUC: 0.9485`

`--- Fold 3/5 ---`  
  `Acc: 0.7546 | F1: 0.7493 | AUC: 0.9489`

`--- Fold 4/5 ---`  
  `Acc: 0.7550 | F1: 0.7466 | AUC: 0.9590`

`--- Fold 5/5 ---`  
  `Acc: 0.8029 | F1: 0.7861 | AUC: 0.9553`

`======================================================================`  
`🚀 INITIALIZING RUN: MOBILENET_V2`  
`======================================================================`

`--- Fold 1/5 ---`  
`Downloading: "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth`

`100%|██████████| 13.6M/13.6M [00:00<00:00, 147MB/s]`

 `Acc: 0.5886 | F1: 0.5633 | AUC: 0.9001`

`--- Fold 2/5 ---`  
  `Acc: 0.6301 | F1: 0.6044 | AUC: 0.9011`

`--- Fold 3/5 ---`  
  `Acc: 0.6442 | F1: 0.6505 | AUC: 0.9186`

`--- Fold 4/5 ---`  
  `Acc: 0.5828 | F1: 0.5785 | AUC: 0.9395`

`--- Fold 5/5 ---`  
  `Acc: 0.6204 | F1: 0.6004 | AUC: 0.9202`

`======================================================================`  
`🚀 INITIALIZING RUN: MOBILENET_V3_LARGE`  
`======================================================================`

`--- Fold 1/5 ---`  
`Downloading: "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_large-8738ca79.pth`

`100%|██████████| 21.1M/21.1M [00:00<00:00, 180MB/s]`

 `Acc: 0.6329 | F1: 0.6185 | AUC: 0.9374`

`--- Fold 2/5 ---`  
  `Acc: 0.7192 | F1: 0.7254 | AUC: 0.9447`

`--- Fold 3/5 ---`  
  `Acc: 0.7055 | F1: 0.6938 | AUC: 0.9371`

`--- Fold 4/5 ---`  
  `Acc: 0.6887 | F1: 0.6813 | AUC: 0.9445`

`--- Fold 5/5 ---`  
  `Acc: 0.7299 | F1: 0.7115 | AUC: 0.9575`

`======================================================================`  
`🚀 INITIALIZING RUN: EFFICIENTNET_B0`  
`======================================================================`

`--- Fold 1/5 ---`  
`Downloading: "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth" to /root/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth`

`100%|██████████| 20.5M/20.5M [00:00<00:00, 145MB/s]`

 `Acc: 0.5886 | F1: 0.5666 | AUC: 0.9249`

`--- Fold 2/5 ---`  
  `Acc: 0.6233 | F1: 0.5928 | AUC: 0.9202`

`--- Fold 3/5 ---`  
  `Acc: 0.6258 | F1: 0.6222 | AUC: 0.9049`

`--- Fold 4/5 ---`  
  `Acc: 0.6556 | F1: 0.6545 | AUC: 0.9186`

`--- Fold 5/5 ---`  
  `Acc: 0.6715 | F1: 0.6275 | AUC: 0.9126`

`======================================================================`  
`🚀 INITIALIZING RUN: EFFICIENTNET_B4`  
`======================================================================`

`--- Fold 1/5 ---`  
`Downloading: "https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth" to /root/.cache/torch/hub/checkpoints/efficientnet_b4_rwightman-23ab8bcd.pth`

`100%|██████████| 74.5M/74.5M [00:00<00:00, 213MB/s]`

 `Acc: 0.6266 | F1: 0.6179 | AUC: 0.9164`

`--- Fold 2/5 ---`  
  `Acc: 0.5685 | F1: 0.5621 | AUC: 0.8864`

`--- Fold 3/5 ---`  
  `Acc: 0.6319 | F1: 0.6381 | AUC: 0.8988`

`--- Fold 4/5 ---`  
  `Acc: 0.6755 | F1: 0.6834 | AUC: 0.9362`

`--- Fold 5/5 ---`  
  `Acc: 0.6423 | F1: 0.6132 | AUC: 0.9250`

`======================================================================`  
`🚀 INITIALIZING RUN: EFFICIENTNET_B7`  
`======================================================================`

`--- Fold 1/5 ---`  
`Downloading: "https://download.pytorch.org/models/efficientnet_b7_lukemelas-c5b4e57e.pth" to /root/.cache/torch/hub/checkpoints/efficientnet_b7_lukemelas-c5b4e57e.pth`

`100%|██████████| 255M/255M [00:01<00:00, 241MB/s]`

 `Acc: 0.5759 | F1: 0.5828 | AUC: 0.9005`

`--- Fold 2/5 ---`  
  `Acc: 0.6027 | F1: 0.5748 | AUC: 0.8758`

`--- Fold 3/5 ---`  
  `Acc: 0.5890 | F1: 0.5982 | AUC: 0.9142`

`--- Fold 4/5 ---`  
  `Acc: 0.6159 | F1: 0.6413 | AUC: 0.9346`

`--- Fold 5/5 ---`  
  `Acc: 0.5766 | F1: 0.5532 | AUC: 0.8981`

`==========================================================================`  
`🏆 MASTER ABLATION STUDY RESULTS (STRATIFIED PATIENT CV)`  
`==========================================================================`  
\|    \| Model              \|   Accuracy \|   Precision \|   Recall \|   F1-Score \|    AUC \|  
\|---:\|:-------------------\|-----------:\|------------:\|---------:\|-----------:\|-------:\|  
\|  0 \| RESNET50           \|    78.9969 \|      0.779  \|   0.7913 \|     0.7768 \| 0.9578 \|  
\|  1 \| RESNET101          \|    77.6447 \|      0.7617 \|   0.7876 \|     0.7648 \| 0.9603 \|  
\|  2 \| INCEPTION\_V3       \|    76.3382 \|      0.7444 \|   0.7732 \|     0.7457 \| 0.9511 \|  
\|  3 \| GOOGLENET          \|    70.6689 \|      0.6807 \|   0.7303 \|     0.6924 \| 0.9458 \|  
\|  4 \| MOBILENET\_V3\_LARGE \|    69.5256 \|      0.6941 \|   0.7341 \|     0.6861 \| 0.9442 \|  
\|  5 \| VGG19              \|    69.2518 \|      0.6755 \|   0.6941 \|     0.6728 \| 0.9266 \|  
\|  6 \| ALEXNET            \|    67.4264 \|      0.6626 \|   0.6884 \|     0.6606 \| 0.9257 \|  
\|  7 \| EFFICIENTNET\_B0    \|    63.2965 \|      0.6418 \|   0.6548 \|     0.6127 \| 0.9162 \|  
\|  8 \| EFFICIENTNET\_B4    \|    62.8962 \|      0.6311 \|   0.6876 \|     0.6229 \| 0.9126 \|  
\|  9 \| DENSENET121        \|    61.8558 \|      0.6563 \|   0.6484 \|     0.5909 \| 0.9184 \|  
\| 10 \| MOBILENET\_V2       \|    61.3227 \|      0.632  \|   0.6639 \|     0.5994 \| 0.9159 \|  
\| 11 \| EFFICIENTNET\_B7    \|    59.2037 \|      0.608  \|   0.6569 \|     0.5901 \| 0.9046 \|  
\============================================================================

**CODE 3:**  
`import os`  
`import copy`  
`import random`  
`import numpy as np`  
`import pandas as pd`  
`import seaborn as sns`  
`import matplotlib.pyplot as plt`  
`from collections import defaultdict`  
`from PIL import Image`

`import torch`  
`import torch.nn as nn`  
`import torch.optim as optim`  
`from torch.utils.data import DataLoader, Dataset`  
`from torchvision import transforms, models`  
`import torchvision.transforms.functional as TF`

`from sklearn.model_selection import train_test_split`  
`from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,`   
                             `roc_auc_score, confusion_matrix, classification_report)`  
`from sklearn.utils.class_weight import compute_class_weight`  
`from sklearn.linear_model import LogisticRegression`  
`import xgboost as xgb`

`import warnings`  
`warnings.filterwarnings('ignore')`

*`# =============================================================================`*  
*`# 1. DETERMINISM & HYPERPARAMETERS`*  
*`# =============================================================================`*  
`def seed_everything(seed=42):`  
    `random.seed(seed)`  
    `os.environ['PYTHONHASHSEED'] = str(seed)`  
    `np.random.seed(seed)`  
    `torch.manual_seed(seed)`  
    `torch.cuda.manual_seed(seed)`  
    `torch.cuda.manual_seed_all(seed)`  
    `torch.backends.cudnn.deterministic = True`  
    `torch.backends.cudnn.benchmark = False`

`seed_everything(42)`

`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`  
`device_type = 'cuda' if torch.cuda.is_available() else 'cpu'`  
`use_amp = torch.cuda.is_available()`

*`# Experiment Configurations`*  
`EPOCHS = 60`  
`WARMUP_EPOCHS = 5`  
`BATCH_SIZE = 32`  
`PATIENCE = 10`  
`WEIGHT_DECAY = 1e-4`

`MODEL_CONFIGS = {`  
    `'resnet50':        {'res': 384, 'lr': 3e-4, 'backbone_lr': 3e-5},`  
    `'resnet101':       {'res': 384, 'lr': 3e-4, 'backbone_lr': 3e-5},`  
    `'inception_v3':    {'res': 299, 'lr': 3e-4, 'backbone_lr': 3e-5},`  
    `'googlenet':       {'res': 224, 'lr': 3e-4, 'backbone_lr': 3e-5},`  
    `'mobilenet_v3_large': {'res': 224, 'lr': 3e-4, 'backbone_lr': 3e-5},`  
    `'efficientnet_b4': {'res': 380, 'lr': 1e-4, 'backbone_lr': 1e-5}`  
`}`

`print(f"Using compute device: {device} | AMP Enabled: {use_amp}")`

*`# =============================================================================`*  
*`# 2. DATA HANDLING (STRICT PATIENT-LEVEL SPLIT)`*  
*`# =============================================================================`*  
`print("\n" + "="*60)`  
`print("📦 PHASE 1: STRICT PATIENT-LEVEL DATA SPLITTING")`  
`print("="*60)`

`DATA_DIR = None`  
`for root, dirs, files in os.walk('/kaggle/input'):`  
    `if 'fold1' in [d.lower() for d in dirs]:`  
        `DATA_DIR = os.path.join(root, next(d for d in dirs if d.lower() == 'fold1'))`  
        `break`

`if not DATA_DIR:`  
    `raise FileNotFoundError("Could not locate 'fold1' in /kaggle/input/")`

`subdirs = os.listdir(DATA_DIR)`  
`train_dir = os.path.join(DATA_DIR, next(d for d in subdirs if d.lower() == 'train'))`  
`val_dir = os.path.join(DATA_DIR, next(d for d in subdirs if d.lower() in ['val', 'validation', 'valid']))`  
`test_dir = os.path.join(DATA_DIR, next(d for d in subdirs if d.lower() == 'test'))`

`classes = sorted(os.listdir(train_dir))`  
`class_to_idx = {c: i for i, c in enumerate(classes)}`  
`idx_to_class = {i: c for c, i in class_to_idx.items()}`  
`NUM_CLASSES = len(classes)`

`patient_data = defaultdict(list)`  
`for split_dir in [train_dir, val_dir, test_dir]:`  
    `for cls in classes:`  
        `cls_dir = os.path.join(split_dir, cls)`  
        `if not os.path.isdir(cls_dir): continue`  
        `for fname in os.listdir(cls_dir):`  
            `if fname.lower().endswith(('.png', '.jpg', '.jpeg')):`  
                `filepath = os.path.join(cls_dir, fname)`  
                `try:`   
                    `pat_id = f"{cls}_{fname.split('_')[1]}"`  
                `except IndexError:`   
                    `pat_id = f"{cls}_{fname}"`  
                `patient_data[pat_id].append((filepath, class_to_idx[cls]))`

`unique_patients = list(patient_data.keys())`  
*`# Primary label for stratification = label of the patient's first image`*  
`patient_labels = [patient_data[p][0][1] for p in unique_patients]`

*`# Stratified Split: 70% Train, 10% Val, 20% Test`*  
`train_pat, temp_pat, train_lbl, temp_lbl = train_test_split(`  
    `unique_patients, patient_labels, test_size=0.30, stratify=patient_labels, random_state=42`  
`)`  
`val_pat, test_pat, _, _ = train_test_split(`  
    `temp_pat, temp_lbl, test_size=(2.0/3.0), stratify=temp_lbl, random_state=42`  
`)`

`def get_files_from_patients(pat_list):`  
    `files = []`  
    `for p in pat_list:`  
        `files.extend(patient_data[p])`  
    `return files`

`train_files = get_files_from_patients(train_pat)`  
`val_files = get_files_from_patients(val_pat)`  
`test_files = get_files_from_patients(test_pat)`

`print(f"Patients -> Train: {len(train_pat)} | Val: {len(val_pat)} | Test: {len(test_pat)}")`  
`print(f"Images   -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")`

`def print_distribution(files, name):`  
    `df = pd.DataFrame([idx_to_class[f[1]] for f in files], columns=['Class'])`  
    `print(f"\n{name} Class Distribution:")`  
    `print(df.value_counts().to_string())`

`print_distribution(train_files, "Train")`  
`print_distribution(val_files, "Validation")`  
`print_distribution(test_files, "Test")`

`train_targets = [f[1] for f in train_files]`  
`class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_targets), y=train_targets)`  
`class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)`

*`# =============================================================================`*  
*`# 3. DATASETS & TEST TIME AUGMENTATION (TTA)`*  
*`# =============================================================================`*  
`class SkinLesionDataset(Dataset):`  
    `def __init__(self, file_list, target_size, mode='train'):`  
        `self.file_list = file_list`  
        `self.target_size = target_size`  
        `self.mode = mode`  
          
        `self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`  
          
        `self.train_transform = transforms.Compose([`  
            `transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),`  
            `transforms.RandomHorizontalFlip(p=0.5),`  
            `transforms.RandomRotation(15),`  
            `transforms.ColorJitter(brightness=0.1, contrast=0.1),`  
            `transforms.ToTensor(),`  
            `self.normalize`  
        `])`  
          
        `self.val_transform = transforms.Compose([`  
            `transforms.Resize((int(target_size * 1.15), int(target_size * 1.15))),`  
            `transforms.CenterCrop(target_size),`  
            `transforms.ToTensor(),`  
            `self.normalize`  
        `])`

    `def __len__(self): return len(self.file_list)`  
      
    `def __getitem__(self, idx):`  
        `filepath, label = self.file_list[idx]`  
        `img = Image.open(filepath).convert('RGB')`  
          
        `if self.mode == 'train':`  
            `return self.train_transform(img), label`  
        `elif self.mode == 'val':`  
            `return self.val_transform(img), label`  
        `elif self.mode == 'test_tta':`  
            `img_resized = transforms.Resize((int(self.target_size * 1.15), int(self.target_size * 1.15)))(img)`  
              
            `crops = []`  
            `crops.append(transforms.CenterCrop(self.target_size)(img_resized)) # Original`  
            `crops.append(TF.hflip(crops[0])) # HFlip`  
            `five_crops = transforms.FiveCrop(self.target_size)(img_resized) # 5-Crop`  
            `crops.extend(list(five_crops))`  
              
            `tensor_crops = torch.stack([self.normalize(transforms.ToTensor()(c)) for c in crops])`  
            `return tensor_crops, label`

*`# =============================================================================`*  
*`# 4. ARCHITECTURE BUILDER`*  
*`# =============================================================================`*  
`def build_model(model_name):`  
    `if model_name == 'resnet50':`  
        `model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)`  
        `model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, NUM_CLASSES))`  
        `head, backbone = model.fc.parameters(), [p for n, p in model.named_parameters() if 'fc' not in n]`  
    `elif model_name == 'resnet101':`  
        `model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)`  
        `model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, NUM_CLASSES))`  
        `head, backbone = model.fc.parameters(), [p for n, p in model.named_parameters() if 'fc' not in n]`  
    `elif model_name == 'inception_v3':`  
        `model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)`  
        `model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES)`  
        `model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, NUM_CLASSES))`  
        `head = list(model.fc.parameters()) + list(model.AuxLogits.fc.parameters())`  
        `backbone = [p for n, p in model.named_parameters() if 'fc' not in n]`  
    `elif model_name == 'googlenet':`  
        `model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1, transform_input=False)`  
        `model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, NUM_CLASSES))`  
        `model.aux_logits = False`  
        `head, backbone = model.fc.parameters(), [p for n, p in model.named_parameters() if 'fc' not in n]`  
    `elif model_name == 'mobilenet_v3_large':`  
        `model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)`  
        `model.classifier[3] = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.classifier[3].in_features, NUM_CLASSES))`  
        `head, backbone = model.classifier.parameters(), model.features.parameters()`  
    `elif model_name == 'efficientnet_b4':`  
        `model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)`  
        `model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.classifier[1].in_features, NUM_CLASSES))`  
        `head, backbone = model.classifier.parameters(), model.features.parameters()`  
      
    `for p in backbone: p.requires_grad = False`  
    `return model.to(device), backbone, head`

*`# =============================================================================`*  
*`# 5. TRAINING & EVALUATION ENGINE`*  
*`# =============================================================================`*  
`criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)`

`model_results = {}`  
`val_probs_dict = {}`  
`test_probs_dict = {}`

`print("\n" + "="*60)`  
`print("🚀 PHASE 2: TRAINING ABLATION STUDY")`  
`print("="*60)`

`for model_name, config in MODEL_CONFIGS.items():`  
    `print(f"\n>>> INITIALIZING: {model_name.upper()} (Res: {config['res']}x{config['res']})")`  
      
    `train_loader = DataLoader(SkinLesionDataset(train_files, config['res'], 'train'), batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)`  
    `val_loader = DataLoader(SkinLesionDataset(val_files, config['res'], 'val'), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)`  
    `test_loader_tta = DataLoader(SkinLesionDataset(test_files, config['res'], 'test_tta'), batch_size=1, shuffle=False, num_workers=2)`  
      
    `model, backbone, head = build_model(model_name)`  
    `optimizer = optim.Adam([{'params': backbone, 'lr': config['backbone_lr']},`   
                            `{'params': head, 'lr': config['lr']}], weight_decay=WEIGHT_DECAY)`  
    `scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)`  
    `scaler = torch.amp.GradScaler(device_type, enabled=use_amp)`  
      
    `best_f1 = 0.0`  
    `patience_counter = 0`  
    `best_weights = None`  
      
    `for epoch in range(EPOCHS):`  
        `if epoch == WARMUP_EPOCHS:`  
            `for p in backbone: p.requires_grad = True`  
              
        `model.train()`  
        `train_loss = 0.0`  
        `for images, labels in train_loader:`  
            `images, labels = images.to(device), labels.to(device)`  
            `optimizer.zero_grad()`  
            `with torch.amp.autocast(device_type, enabled=use_amp):`  
                `if model_name == 'inception_v3':`  
                    `out, aux = model(images)`  
                    `loss = criterion(out, labels) + 0.4 * criterion(aux, labels)`  
                `else:`  
                    `loss = criterion(model(images), labels)`  
            `scaler.scale(loss).backward()`  
            `scaler.step(optimizer)`  
            `scaler.update()`  
            `train_loss += loss.item() * images.size(0)`  
              
        `scheduler.step()`  
        `train_loss /= len(train_loader.dataset)`  
          
        `# Validation`  
        `model.eval()`  
        `val_loss = 0.0`  
        `all_preds, all_labels = [], []`  
        `with torch.no_grad():`  
            `for images, labels in val_loader:`  
                `images, labels = images.to(device), labels.to(device)`  
                `with torch.amp.autocast(device_type, enabled=use_amp):`  
                    `out = model(images)`  
                    `loss = criterion(out, labels)`  
                `val_loss += loss.item() * images.size(0)`  
                `all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())`  
                `all_labels.extend(labels.cpu().numpy())`  
                  
        `val_loss /= len(val_loader.dataset)`  
        `val_f1 = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)[2]`  
          
        `if val_f1 > best_f1:`  
            `best_f1 = val_f1`  
            `patience_counter = 0`  
            `best_weights = copy.deepcopy(model.state_dict())`  
            `print(f"  Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} 🌟")`  
        `else:`  
            `patience_counter += 1`  
            `if patience_counter >= PATIENCE:`  
                `print(f"  Early stopping at epoch {epoch+1}")`  
                `break`

    `# TTA Inference (Test & Val for Ensembling)`  
    `model.load_state_dict(best_weights)`  
    `model.eval()`  
      
    `# Extract Val Probabilities`  
    `v_probs, v_labels = [], []`  
    `with torch.no_grad():`  
        `for images, labels in val_loader:`  
            `images = images.to(device)`  
            `with torch.amp.autocast(device_type, enabled=use_amp):`  
                `probs = torch.softmax(model(images), dim=1)`  
            `v_probs.extend(probs.cpu().numpy())`  
            `v_labels.extend(labels.numpy())`  
    `val_probs_dict[model_name] = np.array(v_probs)`  
    `val_y_true = np.array(v_labels)`  
      
    `# Extract Test Probabilities with TTA`  
    `t_probs, t_labels = [], []`  
    `with torch.no_grad():`  
        `for tta_images, labels in test_loader_tta:`  
            `tta_images = tta_images.squeeze(0).to(device)`  
            `with torch.amp.autocast(device_type, enabled=use_amp):`  
                `probs = torch.softmax(model(tta_images), dim=1)`  
            `avg_probs = probs.mean(dim=0).cpu().numpy()`  
            `t_probs.append(avg_probs)`  
            `t_labels.append(labels.item())`  
              
    `t_probs = np.array(t_probs)`  
    `t_labels = np.array(t_labels)`  
    `test_probs_dict[model_name] = t_probs`  
    `test_y_true = t_labels`  
      
    `# Metrics`  
    `preds = np.argmax(t_probs, axis=1)`  
    `acc = accuracy_score(t_labels, preds)`  
    `prec, rec, f1, _ = precision_recall_fscore_support(t_labels, preds, average='macro', zero_division=0)`  
    `try: auc = roc_auc_score(t_labels, t_probs, multi_class='ovr', average='macro')`  
    `except: auc = float('nan')`  
      
    `model_results[model_name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'AUC': auc}`  
      
    `print(f"  [TEST TTA] Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")`  
    `del model, optimizer, scaler`  
    `torch.cuda.empty_cache()`

*`# =============================================================================`*  
*`# 6. ENSEMBLE ENGINEERING`*  
*`# =============================================================================`*  
`print("\n" + "="*60)`  
`print("🧩 PHASE 3: ENSEMBLE EVALUATION")`  
`print("="*60)`

`ensemble_results = {}`  
`ensemble_probs_dict = {}` 

`def eval_ensemble(probs, name):`  
    `preds = np.argmax(probs, axis=1)`  
    `acc = accuracy_score(test_y_true, preds)`  
    `prec, rec, f1, _ = precision_recall_fscore_support(test_y_true, preds, average='macro', zero_division=0)`  
    `try: auc = roc_auc_score(test_y_true, probs, multi_class='ovr', average='macro')`  
    `except: auc = float('nan')`  
    `ensemble_results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'AUC': auc}`  
    `ensemble_probs_dict[name] = probs`  
    `print(f"{name} -> Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")`  
    `return preds, probs`

*`# ENSEMBLE 1: Strongest Trio (Soft Voting)`*  
`e1_models = ['resnet50', 'inception_v3', 'googlenet']`  
`e1_probs = np.mean([test_probs_dict[m] for m in e1_models], axis=0)`  
`eval_ensemble(e1_probs, 'Ens 1: Trio Soft Voting')`

*`# ENSEMBLE 2: Controlled Diversity (Soft Voting)`*  
`e2_models = ['resnet50', 'mobilenet_v3_large', 'inception_v3']`  
`e2_probs = np.mean([test_probs_dict[m] for m in e2_models], axis=0)`  
`eval_ensemble(e2_probs, 'Ens 2: Diverse Soft Voting')`

*`# ENSEMBLE 3: Stacking (Meta-Learners)`*  
`stack_models = ['resnet50', 'resnet101', 'inception_v3', 'googlenet']`  
`X_val_stack = np.concatenate([val_probs_dict[m] for m in stack_models], axis=1)`  
`X_test_stack = np.concatenate([test_probs_dict[m] for m in stack_models], axis=1)`

*`# LogReg Stacking`*  
`logreg = LogisticRegression(max_iter=1000, class_weight='balanced')`  
`logreg.fit(X_val_stack, val_y_true)`  
`e3_lr_probs = logreg.predict_proba(X_test_stack)`  
`eval_ensemble(e3_lr_probs, 'Ens 3: Stacking (LogReg)')`

*`# XGBoost Stacking`*  
`xgb_meta = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)`  
`xgb_meta.fit(X_val_stack, val_y_true)`  
`e3_xgb_probs = xgb_meta.predict_proba(X_test_stack)`  
`eval_ensemble(e3_xgb_probs, 'Ens 3: Stacking (XGBoost)')`

*`# ENSEMBLE 4: Disagreement-Based Selection (FIXED: Using Validation Set)`*  
`d_models = ['resnet50', 'resnet101', 'inception_v3', 'googlenet', 'mobilenet_v3_large']`  
`disagreement_matrix = np.zeros((len(d_models), len(d_models)))`

`for i, m1 in enumerate(d_models):`  
    `for j, m2 in enumerate(d_models):`  
        `# Calculate disagreement strictly on the VALIDATION set`  
        `p1 = np.argmax(val_probs_dict[m1], axis=1)`  
        `p2 = np.argmax(val_probs_dict[m2], axis=1)`  
        `disagreement_matrix[i, j] = np.mean(p1 != p2)`

*`# Save Disagreement Heatmap`*  
`plt.figure(figsize=(8, 6))`  
`sns.heatmap(disagreement_matrix, annot=True, xticklabels=d_models, yticklabels=d_models, cmap='Reds', fmt='.3f')`  
`plt.title("Validation Disagreement Matrix (% differing predictions)")`  
`plt.tight_layout()`  
`plt.savefig('val_disagreement_heatmap.png', dpi=300)`  
`plt.close()`

*`# Identify 3 most complementary`*  
`disagreement_scores = disagreement_matrix.sum(axis=1)`  
`top_3_idx = np.argsort(disagreement_scores)[-3:]`  
`e4_models = [d_models[i] for i in top_3_idx]`  
`print(f"  Selected for Ens 4 (High Val Disagreement): {e4_models}")`

`e4_probs = np.mean([test_probs_dict[m] for m in e4_models], axis=0)`  
`eval_ensemble(e4_probs, 'Ens 4: Disagreement Soft Voting')`

*`# =============================================================================`*  
*`# 7. FINAL RANKINGS & VISUALIZATIONS`*  
*`# =============================================================================`*  
`print("\n" + "="*60)`  
`print("📊 PHASE 4: REPORTS & RANKINGS")`  
`print("="*60)`

*`# Combine results`*  
`all_results = {**model_results, **ensemble_results}`  
`df_res = pd.DataFrame(all_results).T`  
`df_res['Accuracy'] *= 100`  
`df_res = df_res.sort_values(by=['Accuracy', 'F1-Score'], ascending=False).round(4)`

`print("\n--- MASTER PERFORMANCE LEADERBOARD ---")`  
`print(df_res.to_markdown())`

`best_single = pd.DataFrame(model_results).T.sort_values(by='Accuracy', ascending=False).iloc[0]`  
`best_ensemble = pd.DataFrame(ensemble_results).T.sort_values(by='Accuracy', ascending=False).iloc[0]`  
`baseline = model_results['resnet50']['Accuracy'] * 100`

`print(f"\n🏆 Best Single Model : {best_single.name} (Acc: {best_single['Accuracy']*100:.2f}%)")`  
`print(f"🏆 Best Ensemble     : {best_ensemble.name} (Acc: {best_ensemble['Accuracy']*100:.2f}%)")`  
`print(f"📈 Improvement over ResNet50 baseline: {(best_ensemble['Accuracy']*100) - baseline:+.2f}%")`

*`# Plot ROC for Best Ensemble dynamically based on the leaderboard winner`*  
`best_ensemble_name = best_ensemble.name`  
`best_probs = ensemble_probs_dict[best_ensemble_name]`  
`best_preds = np.argmax(best_probs, axis=1)`

`from sklearn.preprocessing import label_binarize`  
`from sklearn.metrics import roc_curve, auc`

`y_test_bin = label_binarize(test_y_true, classes=range(NUM_CLASSES))`

*`# Plot ROC for Best Ensemble`*  
`plt.figure(figsize=(10, 8))`  
`for i in range(NUM_CLASSES):`  
    `fpr, tpr, _ = roc_curve(y_test_bin[:, i], best_probs[:, i])`  
    `roc_auc = auc(fpr, tpr)`  
    `plt.plot(fpr, tpr, lw=2, label=f'{idx_to_class[i]} (AUC = {roc_auc:.3f})')`

`plt.plot([0, 1], [0, 1], 'k--', lw=2)`  
`plt.xlim([0.0, 1.0])`  
`plt.ylim([0.0, 1.05])`  
`plt.xlabel('False Positive Rate', fontweight='bold')`  
`plt.ylabel('True Positive Rate', fontweight='bold')`  
`plt.title(f'ROC Curves: Best Ensemble ({best_ensemble_name})', fontweight='bold')`  
`plt.legend(loc="lower right")`  
`plt.tight_layout()`  
`plt.savefig('best_ensemble_roc.png', dpi=300)`  
`plt.close()`

*`# Plot Confusion Matrix for Best Ensemble`*  
`cm = confusion_matrix(test_y_true, best_preds)`

`plt.figure(figsize=(8, 6))`  
`sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)`  
`plt.title(f'Confusion Matrix: Best Ensemble ({best_ensemble_name})', fontweight='bold')`  
`plt.ylabel('True Label', fontweight='bold')`  
`plt.xlabel('Predicted Label', fontweight='bold')`  
`plt.tight_layout()`  
`plt.savefig('best_ensemble_cm.png', dpi=300)`  
`plt.close()`

*`# Detailed Classification Report`*  
`print(f"\n--- DETAILED CLASSIFICATION REPORT: {best_ensemble_name} ---")`  
`print(classification_report(test_y_true, best_preds, target_names=classes))`

`print("\n✅ Execution Complete. Visualizations saved as .png files.")`  
**CODE 3 Output :**  
**`Using compute device: cuda | AMP Enabled: True`**

**`============================================================`**  
**`📦 PHASE 1: STRICT PATIENT-LEVEL DATA SPLITTING`**  
**`============================================================`**  
**`Patients -> Train: 366 | Val: 52 | Test: 106`**  
**`Images   -> Train: 509 | Val: 87 | Test: 159`**

**`Train Class Distribution:`**  
**`Class`**       
**`Monkeypox     187`**  
**`HFMD          109`**  
**`Healthy        79`**  
**`Chickenpox     56`**  
**`Measles        40`**  
**`Cowpox         38`**

**`Validation Class Distribution:`**  
**`Class`**       
**`Monkeypox     45`**  
**`HFMD          15`**  
**`Healthy       10`**  
**`Chickenpox     6`**  
**`Cowpox         6`**  
**`Measles        5`**

**`Test Class Distribution:`**  
**`Class`**       
**`Monkeypox     52`**  
**`HFMD          37`**  
**`Healthy       25`**  
**`Cowpox        22`**  
**`Chickenpox    13`**  
**`Measles       10`**

**`============================================================`**  
**`🚀 PHASE 2: TRAINING ABLATION STUDY`**  
**`============================================================`**

**`>>> INITIALIZING: RESNET50 (Res: 384x384)`**  
**`Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`**

**`100%|██████████| 97.8M/97.8M [00:00<00:00, 154MB/s]`**

 **`Epoch 01 | Train: 1.7511 | Val Loss: 1.7676 | Val F1: 0.1667 🌟`**  
  **`Epoch 02 | Train: 1.6692 | Val Loss: 1.6637 | Val F1: 0.3893 🌟`**  
  **`Epoch 03 | Train: 1.5670 | Val Loss: 1.6162 | Val F1: 0.4654 🌟`**  
  **`Epoch 05 | Train: 1.4532 | Val Loss: 1.5003 | Val F1: 0.5442 🌟`**  
  **`Epoch 06 | Train: 1.3033 | Val Loss: 1.2463 | Val F1: 0.5737 🌟`**  
  **`Epoch 08 | Train: 0.7839 | Val Loss: 0.9437 | Val F1: 0.7001 🌟`**  
  **`Epoch 09 | Train: 0.6375 | Val Loss: 0.9158 | Val F1: 0.7223 🌟`**  
  **`Epoch 11 | Train: 0.5444 | Val Loss: 0.9447 | Val F1: 0.7226 🌟`**  
  **`Epoch 14 | Train: 0.4455 | Val Loss: 0.9253 | Val F1: 0.7252 🌟`**  
  **`Early stopping at epoch 24`**  
  **`[TEST TTA] Acc: 0.8491 | F1: 0.8267 | AUC: 0.9751`**

**`>>> INITIALIZING: RESNET101 (Res: 384x384)`**  
**`Downloading: "https://download.pytorch.org/models/resnet101-63fe2227.pth" to /root/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth`**

**`100%|██████████| 171M/171M [00:01<00:00, 173MB/s]`**

 **`Epoch 01 | Train: 1.7330 | Val Loss: 1.7877 | Val F1: 0.2225 🌟`**  
  **`Epoch 02 | Train: 1.6184 | Val Loss: 1.6913 | Val F1: 0.3410 🌟`**  
  **`Epoch 03 | Train: 1.5537 | Val Loss: 1.6075 | Val F1: 0.3495 🌟`**  
  **`Epoch 04 | Train: 1.4628 | Val Loss: 1.5487 | Val F1: 0.4048 🌟`**  
  **`Epoch 05 | Train: 1.4025 | Val Loss: 1.4720 | Val F1: 0.4207 🌟`**  
  **`Epoch 06 | Train: 1.2283 | Val Loss: 1.2361 | Val F1: 0.4684 🌟`**  
  **`Epoch 07 | Train: 0.9113 | Val Loss: 1.0345 | Val F1: 0.5778 🌟`**  
  **`Epoch 08 | Train: 0.7229 | Val Loss: 0.9464 | Val F1: 0.7080 🌟`**  
  **`Epoch 10 | Train: 0.4982 | Val Loss: 0.9154 | Val F1: 0.7347 🌟`**  
  **`Epoch 14 | Train: 0.3792 | Val Loss: 0.9321 | Val F1: 0.7794 🌟`**  
  **`Early stopping at epoch 24`**  
  **`[TEST TTA] Acc: 0.8176 | F1: 0.8189 | AUC: 0.9691`**

**`>>> INITIALIZING: INCEPTION_V3 (Res: 299x299)`**  
**`Downloading: "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth" to /root/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth`**

**`100%|██████████| 104M/104M [00:00<00:00, 200MB/s]`** 

 **`Epoch 01 | Train: 2.5098 | Val Loss: 1.8010 | Val F1: 0.1688 🌟`**  
  **`Epoch 02 | Train: 2.3676 | Val Loss: 1.7720 | Val F1: 0.2754 🌟`**  
  **`Epoch 03 | Train: 2.2613 | Val Loss: 1.7641 | Val F1: 0.2885 🌟`**  
  **`Epoch 04 | Train: 2.1518 | Val Loss: 1.7249 | Val F1: 0.2891 🌟`**  
  **`Epoch 05 | Train: 2.0798 | Val Loss: 1.6929 | Val F1: 0.4161 🌟`**  
  **`Epoch 06 | Train: 1.9526 | Val Loss: 1.5585 | Val F1: 0.4607 🌟`**  
  **`Epoch 07 | Train: 1.6820 | Val Loss: 1.4473 | Val F1: 0.5201 🌟`**  
  **`Epoch 08 | Train: 1.4544 | Val Loss: 1.3172 | Val F1: 0.5775 🌟`**  
  **`Epoch 09 | Train: 1.2666 | Val Loss: 1.1944 | Val F1: 0.6534 🌟`**  
  **`Epoch 10 | Train: 1.1012 | Val Loss: 1.1155 | Val F1: 0.6979 🌟`**  
  **`Epoch 11 | Train: 0.9677 | Val Loss: 1.0799 | Val F1: 0.7147 🌟`**  
  **`Epoch 13 | Train: 0.8420 | Val Loss: 0.9558 | Val F1: 0.7207 🌟`**  
  **`Epoch 14 | Train: 0.7724 | Val Loss: 0.9333 | Val F1: 0.7615 🌟`**  
  **`Epoch 15 | Train: 0.7180 | Val Loss: 0.9151 | Val F1: 0.7790 🌟`**  
  **`Early stopping at epoch 25`**  
  **`[TEST TTA] Acc: 0.7925 | F1: 0.7799 | AUC: 0.9509`**

**`>>> INITIALIZING: GOOGLENET (Res: 224x224)`**  
**`Downloading: "https://download.pytorch.org/models/googlenet-1378be20.pth" to /root/.cache/torch/hub/checkpoints/googlenet-1378be20.pth`**

**`100%|██████████| 49.7M/49.7M [00:00<00:00, 183MB/s]`**

 **`Epoch 01 | Train: 1.7685 | Val Loss: 1.8133 | Val F1: 0.2818 🌟`**  
  **`Epoch 02 | Train: 1.7223 | Val Loss: 1.7557 | Val F1: 0.3041 🌟`**  
  **`Epoch 03 | Train: 1.6729 | Val Loss: 1.7034 | Val F1: 0.3697 🌟`**  
  **`Epoch 04 | Train: 1.6237 | Val Loss: 1.6778 | Val F1: 0.3950 🌟`**  
  **`Epoch 05 | Train: 1.5825 | Val Loss: 1.6372 | Val F1: 0.4644 🌟`**  
  **`Epoch 06 | Train: 1.4975 | Val Loss: 1.5561 | Val F1: 0.5206 🌟`**  
  **`Epoch 08 | Train: 1.3078 | Val Loss: 1.3777 | Val F1: 0.5678 🌟`**  
  **`Epoch 09 | Train: 1.2036 | Val Loss: 1.3181 | Val F1: 0.5690 🌟`**  
  **`Epoch 10 | Train: 1.1049 | Val Loss: 1.2356 | Val F1: 0.6025 🌟`**  
  **`Epoch 12 | Train: 0.9546 | Val Loss: 1.1224 | Val F1: 0.6293 🌟`**  
  **`Epoch 13 | Train: 0.8950 | Val Loss: 1.0514 | Val F1: 0.6809 🌟`**  
  **`Epoch 15 | Train: 0.7889 | Val Loss: 0.9670 | Val F1: 0.6858 🌟`**  
  **`Epoch 20 | Train: 0.6436 | Val Loss: 0.9022 | Val F1: 0.7035 🌟`**  
  **`Early stopping at epoch 30`**  
  **`[TEST TTA] Acc: 0.7484 | F1: 0.7414 | AUC: 0.9482`**

**`>>> INITIALIZING: MOBILENET_V3_LARGE (Res: 224x224)`**  
**`Downloading: "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_large-8738ca79.pth`**

**`100%|██████████| 21.1M/21.1M [00:00<00:00, 143MB/s]`** 

 **`Epoch 01 | Train: 1.5821 | Val Loss: 1.7286 | Val F1: 0.3589 🌟`**  
  **`Epoch 02 | Train: 1.2449 | Val Loss: 1.5509 | Val F1: 0.4157 🌟`**  
  **`Epoch 03 | Train: 1.1180 | Val Loss: 1.4338 | Val F1: 0.4586 🌟`**  
  **`Epoch 04 | Train: 0.9533 | Val Loss: 1.3235 | Val F1: 0.4594 🌟`**  
  **`Epoch 05 | Train: 0.8699 | Val Loss: 1.2284 | Val F1: 0.4933 🌟`**  
  **`Epoch 06 | Train: 0.8430 | Val Loss: 1.1657 | Val F1: 0.5200 🌟`**  
  **`Epoch 07 | Train: 0.7840 | Val Loss: 1.1234 | Val F1: 0.5734 🌟`**  
  **`Epoch 08 | Train: 0.7585 | Val Loss: 1.0950 | Val F1: 0.5837 🌟`**  
  **`Epoch 09 | Train: 0.7064 | Val Loss: 1.0217 | Val F1: 0.5873 🌟`**  
  **`Epoch 11 | Train: 0.6956 | Val Loss: 0.9703 | Val F1: 0.6160 🌟`**  
  **`Epoch 16 | Train: 0.6242 | Val Loss: 0.9302 | Val F1: 0.6461 🌟`**  
  **`Epoch 17 | Train: 0.6271 | Val Loss: 0.9010 | Val F1: 0.6585 🌟`**  
  **`Epoch 18 | Train: 0.6070 | Val Loss: 0.9165 | Val F1: 0.6588 🌟`**  
  **`Epoch 21 | Train: 0.5838 | Val Loss: 0.8804 | Val F1: 0.6836 🌟`**  
  **`Early stopping at epoch 31`**  
  **`[TEST TTA] Acc: 0.6792 | F1: 0.6866 | AUC: 0.9406`**

**`>>> INITIALIZING: EFFICIENTNET_B4 (Res: 380x380)`**  
**`Downloading: "https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth" to /root/.cache/torch/hub/checkpoints/efficientnet_b4_rwightman-23ab8bcd.pth`**

**`100%|██████████| 74.5M/74.5M [00:00<00:00, 182MB/s]`**

 **`Epoch 01 | Train: 1.7174 | Val Loss: 1.8365 | Val F1: 0.1656 🌟`**  
  **`Epoch 03 | Train: 1.6927 | Val Loss: 1.8233 | Val F1: 0.1846 🌟`**  
  **`Epoch 04 | Train: 1.6872 | Val Loss: 1.8166 | Val F1: 0.2305 🌟`**  
  **`Epoch 05 | Train: 1.6768 | Val Loss: 1.8095 | Val F1: 0.2612 🌟`**  
  **`Epoch 06 | Train: 1.6678 | Val Loss: 1.8015 | Val F1: 0.2955 🌟`**  
  **`Epoch 07 | Train: 1.6626 | Val Loss: 1.7939 | Val F1: 0.3122 🌟`**  
  **`Epoch 08 | Train: 1.6536 | Val Loss: 1.7881 | Val F1: 0.3265 🌟`**  
  **`Epoch 09 | Train: 1.6481 | Val Loss: 1.7811 | Val F1: 0.3406 🌟`**  
  **`Epoch 10 | Train: 1.6356 | Val Loss: 1.7735 | Val F1: 0.3494 🌟`**  
  **`Epoch 11 | Train: 1.6262 | Val Loss: 1.7667 | Val F1: 0.3845 🌟`**  
  **`Epoch 12 | Train: 1.6125 | Val Loss: 1.7599 | Val F1: 0.3943 🌟`**  
  **`Epoch 13 | Train: 1.6073 | Val Loss: 1.7554 | Val F1: 0.3950 🌟`**  
  **`Epoch 14 | Train: 1.6049 | Val Loss: 1.7478 | Val F1: 0.4195 🌟`**  
  **`Epoch 15 | Train: 1.5964 | Val Loss: 1.7423 | Val F1: 0.4386 🌟`**  
  **`Epoch 18 | Train: 1.5790 | Val Loss: 1.7247 | Val F1: 0.4533 🌟`**  
  **`Epoch 19 | Train: 1.5686 | Val Loss: 1.7208 | Val F1: 0.4534 🌟`**  
  **`Epoch 24 | Train: 1.5337 | Val Loss: 1.6948 | Val F1: 0.4638 🌟`**  
  **`Epoch 25 | Train: 1.5285 | Val Loss: 1.6893 | Val F1: 0.4682 🌟`**  
  **`Epoch 33 | Train: 1.5004 | Val Loss: 1.6641 | Val F1: 0.4718 🌟`**  
  **`Epoch 37 | Train: 1.4934 | Val Loss: 1.6535 | Val F1: 0.4768 🌟`**  
  **`Epoch 46 | Train: 1.4778 | Val Loss: 1.6443 | Val F1: 0.4889 🌟`**  
  **`Epoch 50 | Train: 1.4731 | Val Loss: 1.6409 | Val F1: 0.4938 🌟`**  
  **`Epoch 53 | Train: 1.4694 | Val Loss: 1.6417 | Val F1: 0.5004 🌟`**  
  **`[TEST TTA] Acc: 0.5472 | F1: 0.5452 | AUC: 0.9209`**

**`============================================================`**  
**`🧩 PHASE 3: ENSEMBLE EVALUATION`**  
**`============================================================`**  
**`Ens 1: Trio Soft Voting -> Acc: 0.8302 | F1: 0.8215 | AUC: 0.9725`**  
**`Ens 2: Diverse Soft Voting -> Acc: 0.8113 | F1: 0.8003 | AUC: 0.9691`**  
**`Ens 3: Stacking (LogReg) -> Acc: 0.8365 | F1: 0.8360 | AUC: 0.9733`**  
**`Ens 3: Stacking (XGBoost) -> Acc: 0.8176 | F1: 0.8105 | AUC: 0.9482`**  
  **`Selected for Ens 4 (High Val Disagreement): ['resnet50', 'googlenet', 'mobilenet_v3_large']`**  
**`Ens 4: Disagreement Soft Voting -> Acc: 0.8050 | F1: 0.7952 | AUC: 0.9686`**

**`============================================================`**  
**`📊 PHASE 4: REPORTS & RANKINGS`**  
**`============================================================`**

**`--- MASTER PERFORMANCE LEADERBOARD ---`**  
**`|                                 |   Accuracy |   Precision |   Recall |   F1-Score |    AUC |`**  
**`|:--------------------------------|-----------:|------------:|---------:|-----------:|-------:|`**  
**`| resnet50                        |    84.9057 |      0.8211 |   0.8343 |     0.8267 | 0.9751 |`**  
**`| Ens 3: Stacking (LogReg)        |    83.6478 |      0.8345 |   0.8435 |     0.836  | 0.9733 |`**  
**`| Ens 1: Trio Soft Voting         |    83.0189 |      0.8262 |   0.819  |     0.8215 | 0.9725 |`**  
**`| resnet101                       |    81.761  |      0.8485 |   0.799  |     0.8189 | 0.9691 |`**  
**`| Ens 3: Stacking (XGBoost)       |    81.761  |      0.8633 |   0.7773 |     0.8105 | 0.9482 |`**  
**`| Ens 2: Diverse Soft Voting      |    81.1321 |      0.813  |   0.8021 |     0.8003 | 0.9691 |`**  
**`| Ens 4: Disagreement Soft Voting |    80.5031 |      0.7957 |   0.7996 |     0.7952 | 0.9686 |`**  
**`| inception_v3                    |    79.2453 |      0.7973 |   0.7741 |     0.7799 | 0.9509 |`**  
**`| googlenet                       |    74.8428 |      0.7609 |   0.7502 |     0.7414 | 0.9482 |`**  
**`| mobilenet_v3_large              |    67.9245 |      0.7005 |   0.7029 |     0.6866 | 0.9406 |`**  
**`| efficientnet_b4                 |    54.717  |      0.6081 |   0.6505 |     0.5452 | 0.9209 |`**

**`🏆 Best Single Model : resnet50 (Acc: 84.91%)`**  
**`🏆 Best Ensemble     : Ens 3: Stacking (LogReg) (Acc: 83.65%)`**  
**`📈 Improvement over ResNet50 baseline: -1.26%`**

**`--- DETAILED CLASSIFICATION REPORT: Ens 3: Stacking (LogReg) ---`**  
              **`precision    recall  f1-score   support`**

  **`Chickenpox       0.59      0.77      0.67        13`**  
      **`Cowpox       0.79      0.86      0.83        22`**  
        **`HFMD       0.92      0.92      0.92        37`**  
     **`Healthy       0.88      0.84      0.86        25`**  
     **`Measles       1.00      0.90      0.95        10`**  
   **`Monkeypox       0.83      0.77      0.80        52`**

    **`accuracy                           0.84       159`**  
   **`macro avg       0.83      0.84      0.84       159`**  
**`weighted avg       0.84      0.84      0.84       159`**

**`✅ Execution Complete. Visualizations saved as .png files.`**  
 

**CODE 4:**  
`import os`  
`import json`  
`import random`  
`import numpy as np`  
`import pandas as pd`  
`import torch`  
`import torch.nn as nn`  
`import torch.nn.functional as F`  
`import timm`  
`from glob import glob`  
`from PIL import Image`  
`from torch.utils.data import DataLoader, Dataset`  
`from torchvision import transforms`  
`from torch.optim.swa_utils import AveragedModel, SWALR`  
`from sklearn.metrics import accuracy_score, f1_score, roc_auc_score`  
`from sklearn.ensemble import RandomForestClassifier`  
`from sklearn.preprocessing import label_binarize`

*`# ==========================================`*  
*`# 1️⃣ STRICT REPRODUCIBILITY & CONFIGURATION`*  
*`# ==========================================`*  
`def set_seed(seed=42):`  
    `random.seed(seed)`  
    `np.random.seed(seed)`  
    `torch.manual_seed(seed)`  
    `torch.cuda.manual_seed_all(seed)`  
    `torch.backends.cudnn.deterministic = True`  
    `torch.backends.cudnn.benchmark = False`  
    `os.environ['PYTHONHASHSEED'] = str(seed)`

`set_seed(42)`

`CONFIG = {`  
    `"num_classes": 6,`  
    `"batch_size": 32,`  
    `"epochs": 60,`  
    `"resize_epoch": 999,  # FIX: Disabled to prevent Swin dimensionality crash`  
    `"mixup_alpha": 0.2,`  
    `"cutmix_prob": 0.5,`  
    `"lr": 1e-4,`  
    `"weight_decay": 1e-4,`  
    `"label_smoothing": 0.05,`  
    `"use_fusion": True,`  
    `"enable_stacking": True,`  
    `"swa_start": 50, # Final 10 epochs`  
    `"swa_lr": 5e-5   # FIX: Sane SWA learning rate`  
`}`

`with open("training_config.json", "w") as f:`  
    `json.dump(CONFIG, f, indent=4)`

*`# ==========================================`*  
*`# 2️⃣ DATA (KAGGLE MSLD v2.0)`*  
*`# ==========================================`*  
`class MSLDataset(Dataset):`  
    `def __init__(self, fold_dir, split="train", transform=None):`  
        `self.transform = transform`  
        `self.image_paths = []`  
        `self.labels = []`  
          
        `# Comprehensive mapping handling all possible naming variations`  
        `self.class_info = [`  
            `{"name": "Chickenpox", "code": "chp", "label": 0, "aliases": ["chickenpox"]},`  
            `{"name": "Cowpox",     "code": "cwp", "label": 1, "aliases": ["cowpox"]},`  
            `{"name": "HFMD",       "code": "hfmd","label": 2, "aliases": ["hfmd"]},`  
            `{"name": "Healthy",    "code": "healthy", "label": 3, "aliases": ["healthy"]},`  
            `{"name": "Measles",    "code": "msl", "label": 4, "aliases": ["measles"]},`  
            `{"name": "Mpox",       "code": "mkp", "label": 5, "aliases": ["mpox", "monkeypox"]},`  
        `]`  
          
        `# Handle split naming variations (train -> train, val -> valid/val)`  
        `split_keywords = {`  
            `"train": ["/train/"],`  
            `"val": ["/val/", "/valid/"],`  
            `"test": ["/test/"]`  
        `}`  
        `keywords = split_keywords.get(split.lower(), [f"/{split.lower()}/"])`  
          
        `# Find every single image file in the fold directory`  
        `all_possible_images = glob(os.path.join(fold_dir, "**", "*.*"), recursive=True)`  
          
        `for img_path in all_possible_images:`  
            `if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):`  
                `path_lower = img_path.lower()`  
                  
                `# Check if the image path contains the correct split folder name`  
                `if any(kw in path_lower for kw in keywords):`  
                    `for info in self.class_info:`  
                        `# Check if class code or any alias is in the path`  
                        `if f"/{info['code']}/" in path_lower or any(f"/{alias}/" in path_lower for alias in info["aliases"]):`  
                            `self.image_paths.append(img_path)`  
                            `self.labels.append(info['label'])`  
                            `break`

        `if len(self.image_paths) == 0:`  
            `example_files = all_possible_images[:5]`  
            `raise RuntimeError(f"Found 0 images for {split}. Fold Dir: {fold_dir}. "`  
                               `f"Sample files found: {example_files}")`  
          
        `print(f"  ✅ {split.capitalize()} set loaded: {len(self.image_paths)} images.")`

    `def __len__(self): return len(self.image_paths)`  
    `def __getitem__(self, idx):`  
        `img = Image.open(self.image_paths[idx]).convert("RGB")`  
        `if self.transform: img = self.transform(img)`  
        `return img, self.labels[idx]`

`def get_dataloaders(img_size, batch_size, fold_num=1):`  
    `# Hunt for the FOLDS directory anywhere in the Kaggle input [cite: 126]`  
    `search_pattern = f"/kaggle/input/**/fold{fold_num}"`  
    `found_paths = [p for p in glob(search_pattern, recursive=True) if os.path.isdir(p)]`  
      
    `if not found_paths:`  
        `raise RuntimeError(f"Could not find fold{fold_num} in /kaggle/input!")`  
      
    `fold_path = found_paths[0]`  
    `print(f"🚀 Found Fold {fold_num} at: {fold_path}")`

    `train_transform = transforms.Compose([`  
        `transforms.Resize((img_size, img_size)),`  
        `transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),`  
        `transforms.RandomHorizontalFlip(p=0.5),`  
        `transforms.RandomRotation(15),`  
        `transforms.ToTensor(),`  
        `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`  
    `])`  
      
    `val_transform = transforms.Compose([`  
        `transforms.Resize((img_size, img_size)),`  
        `transforms.ToTensor(),`  
        `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`  
    `])`  
      
    `train_ds = MSLDataset(fold_dir=fold_path, split="train", transform=train_transform)`  
    `val_ds = MSLDataset(fold_dir=fold_path, split="val", transform=val_transform)`  
    `test_ds = MSLDataset(fold_dir=fold_path, split="test", transform=val_transform)`  
      
    `train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)`  
    `val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)`  
    `test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)`  
      
    `return train_loader, val_loader, test_loader`  
*`# ==========================================`*  
*`# 3️⃣ MODEL ARCHITECTURE`*  
*`# ==========================================`*  
`class HybridSkinFusion(nn.Module):`  
    `def __init__(self, num_classes, cfg):`  
        `super().__init__()`  
        `self.cfg = cfg`  
        `self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)`  
        `swin_dim = self.swin.num_features`  
          
        `self.effnet = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0)`  
        `eff_dim = self.effnet.num_features`  
          
        `self.swin_head = nn.Linear(swin_dim, num_classes)`  
        `self.effnet_head = nn.Linear(eff_dim, num_classes)`  
          
        `self.fusion_head = nn.Sequential(`  
            `nn.Linear(swin_dim + eff_dim, 512),`  
            `nn.BatchNorm1d(512),`  
            `nn.ReLU(),`  
            `nn.Dropout(0.4),`  
            `nn.Linear(512, num_classes)`  
        `)`

    `def forward(self, x):`  
        `f_swin = self.swin(x)`  
        `f_eff = self.effnet(x)`  
          
        `swin_logits = self.swin_head(f_swin)`  
        `eff_logits = self.effnet_head(f_eff)`  
          
        `if self.cfg["use_fusion"]:`  
            `f_fused = torch.cat((f_swin, f_eff), dim=1)`  
            `fusion_logits = self.fusion_head(f_fused)`  
        `else:`  
            `fusion_logits, f_fused = None, None`  
              
        `return fusion_logits, swin_logits, eff_logits, f_swin, f_eff, f_fused`

*`# ==========================================`*  
*`# 4️⃣ TRAINING UTILITIES (SAM, MIXUP/CUTMIX, ECE)`*  
*`# ==========================================`*  
`class SAM(torch.optim.Optimizer):`  
    `def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):`  
        `defaults = dict(rho=rho, adaptive=adaptive, **kwargs)`  
        `super(SAM, self).__init__(params, defaults)`  
        `self.base_optimizer = base_optimizer(self.param_groups, **kwargs)`  
        `self.param_groups = self.base_optimizer.param_groups`

    `@torch.no_grad()`  
    `def first_step(self, zero_grad=False):`  
        `grad_norm = self._grad_norm()`  
        `for group in self.param_groups:`  
            `scale = group["rho"] / (grad_norm + 1e-12)`  
            `for p in group["params"]:`  
                `if p.grad is None: continue`  
                `e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)`  
                `p.add_(e_w)`   
                `self.state[p]["e_w"] = e_w`  
        `if zero_grad: self.zero_grad()`

    `@torch.no_grad()`  
    `def second_step(self, zero_grad=False):`  
        `for group in self.param_groups:`  
            `for p in group["params"]:`  
                `if p.grad is None: continue`  
                `p.sub_(self.state[p]["e_w"])`   
        `self.base_optimizer.step()`  
        `if zero_grad: self.zero_grad()`

    `def _grad_norm(self):`  
        `shared_device = self.param_groups[0]["params"][0].device`  
        `norm = torch.norm(`  
            `torch.stack([`  
                `((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)`  
                `for group in self.param_groups for p in group["params"] if p.grad is not None`  
            `]), p=2)`  
        `return norm`

`def mixup_cutmix_data(x, y, alpha=0.2, cutmix_prob=0.5):`  
    `lam = np.random.beta(alpha, alpha) if alpha > 0 else 1`  
    `batch_size = x.size()[0]`  
    `index = torch.randperm(batch_size).to(x.device)`  
      
    `x = x.clone() # FIX: Safe in-place copy`  
    `if np.random.rand() < cutmix_prob: # CutMix`  
        `W, H = x.size(2), x.size(3)`  
        `cut_rat = np.sqrt(1. - lam)`  
        `cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)`  
        `cx, cy = np.random.randint(W), np.random.randint(H)`  
        `bbx1 = np.clip(cx - cut_w // 2, 0, W)`  
        `bby1 = np.clip(cy - cut_h // 2, 0, H)`  
        `bbx2 = np.clip(cx + cut_w // 2, 0, W)`  
        `bby2 = np.clip(cy + cut_h // 2, 0, H)`  
        `x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]`  
        `lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))`  
    `else: # Mixup`  
        `x = lam * x + (1 - lam) * x[index, :]`  
          
    `y_a, y_b = y, y[index]`  
    `return x, y_a, y_b, lam`

`def mixup_criterion(criterion, pred, y_a, y_b, lam):`  
    `return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)`

`def compute_ece(y_true, y_probs, num_bins=10):`  
    `confidences, predictions = np.max(y_probs, axis=1), np.argmax(y_probs, axis=1)`  
    `accuracies = predictions == y_true`  
    `ece = 0.0`  
    `bin_boundaries = np.linspace(0, 1, num_bins + 1)`  
    `for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):`  
        `in_bin = (confidences > bin_lower) & (confidences <= bin_upper)`  
        `prop_in_bin = in_bin.mean()`  
        `if prop_in_bin > 0:`  
            `accuracy_in_bin = accuracies[in_bin].mean()`  
            `avg_confidence_in_bin = confidences[in_bin].mean()`  
            `ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin`  
    `return ece`

*`# ==========================================`*  
*`# 5️⃣ CALIBRATION & DYNAMIC LOSS HELPER`*  
*`# ==========================================`*  
*`# ==========================================`*  
*`# 5️⃣ CALIBRATION & DYNAMIC LOSS HELPER`*  
*`# ==========================================`*  
`class ModelWithTemperature(nn.Module):`  
    `def __init__(self, model):`  
        `super().__init__()`  
        `self.model = model`  
        `self.temperature = nn.Parameter(torch.ones(1) * 1.5)`

    `def forward(self, input):`  
        `fusion_logits, _, _, _, _, _ = self.model(input)`  
        `return fusion_logits / self.temperature`

    `def calibrate(self, valid_loader, device):`  
        `nll_criterion = nn.CrossEntropyLoss().to(device)`  
        `self.model.eval()`  
          
        `# 1. Extract and cache all logits and labels ONCE`  
        `all_logits = []`  
        `all_labels = []`  
        `with torch.no_grad():`  
            `for inputs, labels in valid_loader:`  
                `inputs = inputs.to(device)`  
                `logits = self.model(inputs)[0]`  
                `all_logits.append(logits)`  
                `all_labels.append(labels.to(device))`  
                  
        `all_logits = torch.cat(all_logits).to(device)`  
        `all_labels = torch.cat(all_labels).to(device)`  
          
        `# 2. Optimize the temperature using ONLY the cached logits`  
        `optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)`  
          
        `def eval_closure():`  
            `optimizer.zero_grad()`  
            `# Scale cached logits by temperature`  
            `scaled_logits = all_logits / self.temperature`  
            `loss = nll_criterion(scaled_logits, all_labels)`  
            `loss.backward()`  
            `return loss`  
              
        `optimizer.step(eval_closure)`  
        `print(f"Optimal Temperature found: {self.temperature.item():.4f}")`  
*`# FIX: Spicy Dynamic Orthogonality Loss Helper`*  
`def calculate_dynamic_fusion_loss(fus, swin, eff, y_a, y_b, lam, criterion):`  
    `loss_fus = mixup_criterion(criterion, fus, y_a, y_b, lam) if fus is not None else 0`  
    `loss_swin = mixup_criterion(criterion, swin, y_a, y_b, lam)`  
    `loss_eff = mixup_criterion(criterion, eff, y_a, y_b, lam)`

    `cos_sim = F.cosine_similarity(swin, eff, dim=1).mean()`  
    `ortho_penalty = torch.abs(cos_sim)`  
      
    `branch_weight = 0.5 * (1.0 - cos_sim.detach())`  
    `total_loss = loss_fus + (branch_weight * loss_swin) + (branch_weight * loss_eff) + (0.1 * ortho_penalty)`  
    `return total_loss`

*`# ==========================================`*  
*`# 6️⃣ EVALUATION PIPELINE`*  
*`# ==========================================`*  
`def evaluate_model(model, loader, device, return_logits=False, temp=1.0):`  
    `model.eval()`  
    `all_targets = []`  
    `logits_dict = {"swin": [], "eff": [], "fusion": []}`  
    `f_stats = {"swin_mag": [], "eff_mag": [], "fused_var": [], "logit_cos_sim": []}`  
      
    `with torch.no_grad():`  
        `for inputs, targets in loader:`  
            `inputs, targets = inputs.to(device), targets.to(device)`  
            `fus_log, swin_log, eff_log, f_swin, f_eff, f_fused = model(inputs)`  
              
            `all_targets.extend(targets.cpu().numpy())`  
            `logits_dict["swin"].append(swin_log.cpu().numpy())`  
            `logits_dict["eff"].append(eff_log.cpu().numpy())`  
            `if fus_log is not None: logits_dict["fusion"].append((fus_log / temp).cpu().numpy())`  
              
            `f_stats["swin_mag"].append(torch.norm(f_swin, dim=1).mean().item())`  
            `f_stats["eff_mag"].append(torch.norm(f_eff, dim=1).mean().item())`  
            `if f_fused is not None: f_stats["fused_var"].append(torch.var(f_fused, dim=1).mean().item())`  
            `f_stats["logit_cos_sim"].append(F.cosine_similarity(swin_log, eff_log, dim=1).mean().item())`

    `results = {}`  
    `for key, logits_list in logits_dict.items():`  
        `if not logits_list or logits_list[0] is None: continue`  
        `logits = np.vstack(logits_list)`  
        `probs = F.softmax(torch.tensor(logits), dim=1).numpy()`  
        `preds = np.argmax(probs, axis=1)`  
          
        `y_true_bin = label_binarize(all_targets, classes=range(CONFIG["num_classes"]))`  
        `auc = roc_auc_score(y_true_bin, probs, average='macro', multi_class='ovr') if CONFIG["num_classes"] > 2 else 0`  
          
        `results[key] = {`  
            `"Acc": accuracy_score(all_targets, preds),`  
            `"F1": f1_score(all_targets, preds, average='macro'),`  
            `"AUC": auc,`  
            `"ECE": compute_ece(np.array(all_targets), probs),`  
            `"Logits": logits,`  
            `"Targets": all_targets`  
        `}`  
          
    `avg_f_stats = {k: np.mean(v) for k, v in f_stats.items()}`  
    `return results, avg_f_stats`

*`# ==========================================`*  
*`# 7️⃣ FOLD TRAINING LOOP`*  
*`# ==========================================`*  
`def train_fold(fold_num):`  
    `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`  
    `print(f"\n{'='*20} RUNNING FOLD {fold_num} ON {device} {'='*20}")`  
      
    `model = HybridSkinFusion(CONFIG["num_classes"], CONFIG).to(device)`  
      
    `class_weights = torch.ones(CONFIG["num_classes"]).to(device)`  
    `criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=CONFIG["label_smoothing"])`  
      
    `base_optimizer = torch.optim.AdamW`  
    `optimizer = SAM(model.parameters(), base_optimizer, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])`  
    `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=CONFIG["epochs"])`  
      
    `swa_model = AveragedModel(model)`  
    `swa_scheduler = SWALR(optimizer.base_optimizer, swa_lr=CONFIG["swa_lr"])`  
      
    `# FIX: Start directly at 224 to avoid Swin dimensionality crashes`  
    `train_loader, val_loader, test_loader = get_dataloaders(224, CONFIG["batch_size"], fold_num=fold_num)`  
      
    `best_fusion_val_acc = 0.0`  
    `model_save_path = f"best_hybrid_fusion_fold{fold_num}.pth"`

    `for epoch in range(1, CONFIG["epochs"] + 1):`  
        `model.train()`  
        `epoch_loss = 0`  
        `for images, labels in train_loader:`  
            `images, labels = images.to(device), labels.to(device)`  
            `images, y_a, y_b, lam = mixup_cutmix_data(images, labels, CONFIG["mixup_alpha"], CONFIG["cutmix_prob"])`  
              
            `# Step 1 (SAM)`  
            `fus, swin, eff, _, _, _ = model(images)`  
            `loss = calculate_dynamic_fusion_loss(fus, swin, eff, y_a, y_b, lam, criterion)`  
            `loss.backward()`  
            `optimizer.first_step(zero_grad=True)`  
              
            `# Step 2 (SAM)`  
            `fus, swin, eff, _, _, _ = model(images)`  
            `loss2 = calculate_dynamic_fusion_loss(fus, swin, eff, y_a, y_b, lam, criterion)`  
            `loss2.backward()`  
            `optimizer.second_step(zero_grad=True)`  
              
            `epoch_loss += loss.item()`

        `if epoch > CONFIG["swa_start"]:`  
            `swa_model.update_parameters(model)`  
            `swa_scheduler.step()`  
        `else:`  
            `scheduler.step()`

        `# Validation Step`  
        `val_results, f_stats = evaluate_model(model, val_loader, device)`  
        `fus_acc = val_results['fusion']['Acc']`  
          
        `print(f"Epoch {epoch}/{CONFIG['epochs']} - Loss: {epoch_loss/len(train_loader):.4f} | "`  
              `f"Val Swin: {val_results['swin']['Acc']:.4f} | Val Eff: {val_results['eff']['Acc']:.4f} | Val Fusion: {fus_acc:.4f}")`

        `if fus_acc > best_fusion_val_acc:`  
            `best_fusion_val_acc = fus_acc`  
            `torch.save(model.state_dict(), model_save_path)`

    `print(f"\n--- FOLD {fold_num} TRAINING COMPLETE ---")`  
      
    `# === MEMORY CLEANUP ===`  
    `del optimizer`  
    `del scheduler`  
    `torch.cuda.empty_cache()`  
    `# ======================`  
      
*`# ==========================================`*  
    `# 8️⃣ POST-TRAINING: SWA, CALIBRATION, STACKING (PER FOLD)`  
    `# ==========================================`  
      
    `# FIX: Safe Model Loading`  
    `if os.path.exists(model_save_path):`  
        `model.load_state_dict(torch.load(model_save_path, map_location=device))`  
        `print(f"Loaded best validation model for Fold {fold_num} successfully.")`  
      
    `# Evaluate Best Model on Val and Test Sets`  
    `best_val_results, _ = evaluate_model(model, val_loader, device)`  
    `test_results_base, _ = evaluate_model(model, test_loader, device)`  
      
    `torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)`  
    `test_results_swa, _ = evaluate_model(swa_model, test_loader, device)`  
      
    `temp_model = ModelWithTemperature(model).to(device)`  
    `temp_model.calibrate(val_loader, device)`  
    `test_results_calib, _ = evaluate_model(model, test_loader, device, temp=temp_model.temperature.item())`  
      
    `stack_metrics = {"Acc": np.nan, "F1": np.nan, "AUC": np.nan, "ECE": np.nan}`  
    `if CONFIG["enable_stacking"]:`  
        `# Use the best_val_results we just calculated`  
        `X_train_meta = np.hstack([best_val_results['swin']['Logits'], best_val_results['eff']['Logits'], best_val_results['fusion']['Logits']])`  
        `y_train_meta = best_val_results['swin']['Targets']`  
          
        `rf = RandomForestClassifier(n_estimators=100, random_state=42)`  
        `rf.fit(X_train_meta, y_train_meta)`  
          
        `X_test_meta = np.hstack([test_results_base['swin']['Logits'], test_results_base['eff']['Logits'], test_results_base['fusion']['Logits']])`  
        `y_test_meta = test_results_base['swin']['Targets']`  
          
        `meta_preds = rf.predict(X_test_meta)`  
        `meta_probs = rf.predict_proba(X_test_meta)`  
          
        `y_true_bin = label_binarize(y_test_meta, classes=range(CONFIG["num_classes"]))`  
        `stack_metrics = {`  
            `"Acc": accuracy_score(y_test_meta, meta_preds),`  
            `"F1": f1_score(y_test_meta, meta_preds, average='macro'),`  
            `"AUC": roc_auc_score(y_true_bin, meta_probs, average='macro', multi_class='ovr'),`  
            `"ECE": compute_ece(np.array(y_test_meta), meta_probs)`  
        `}`

    `# Structure data using best_val_results instead of the Epoch 60 val_results`  
    `fold_data = {`  
        `"Swin Only": [best_val_results['swin']['Acc'], test_results_base['swin']['Acc'], test_results_base['swin']['F1'], test_results_base['swin']['AUC'], test_results_base['swin']['ECE']],`  
        `"EffNet Only": [best_val_results['eff']['Acc'], test_results_base['eff']['Acc'], test_results_base['eff']['F1'], test_results_base['eff']['AUC'], test_results_base['eff']['ECE']],`  
        `"Base Fusion": [best_val_results['fusion']['Acc'], test_results_base['fusion']['Acc'], test_results_base['fusion']['F1'], test_results_base['fusion']['AUC'], test_results_base['fusion']['ECE']],`  
        `"Fusion + SWA": [np.nan, test_results_swa['fusion']['Acc'], test_results_swa['fusion']['F1'], test_results_swa['fusion']['AUC'], test_results_swa['fusion']['ECE']],`  
        `"Fusion + Temp": [np.nan, test_results_calib['fusion']['Acc'], test_results_calib['fusion']['F1'], test_results_calib['fusion']['AUC'], test_results_calib['fusion']['ECE']],`  
        `"Stacking (RF)": [np.nan, stack_metrics['Acc'], stack_metrics['F1'], stack_metrics['AUC'], stack_metrics['ECE']]`  
    `}`  
      
    `return fold_data`

*`# ==========================================`*  
*`# 9️⃣ K-FOLD CROSS VALIDATION WRAPPER`*  
*`# ==========================================`*  
`def main():`  
    `all_folds_data = []`  
      
    `# Loop over all 5 folds`  
    `for fold in range(1, 6):`  
        `fold_results = train_fold(fold_num=fold)`  
        `all_folds_data.append(fold_results)`  
          
    `print("\n" + "="*70)`  
    `print("🏆 AGGREGATED 5-FOLD MASTER ABLATION STUDY RESULTS 🏆")`  
    `print("="*70)`  
      
    `# Aggregate and average the results across all 5 folds`  
    `aggregated_data = []`  
    `variants = ["Swin Only", "EffNet Only", "Base Fusion", "Fusion + SWA", "Fusion + Temp", "Stacking (RF)"]`  
      
    `for variant in variants:`  
        `# Extract the metrics for this variant across all folds`  
        `variant_metrics = [fold[variant] for fold in all_folds_data]`  
        `# Calculate the mean across the folds for each metric (ignoring NaNs)`  
        `avg_metrics = np.nanmean(variant_metrics, axis=0)`  
          
        `# Format the row for the DataFrame`  
        `row = [variant] + [f"{m:.4f}" if not np.isnan(m) else "-" for m in avg_metrics]`  
        `aggregated_data.append(row)`  
          
    `df = pd.DataFrame(aggregated_data, columns=["Variant", "Avg Val Acc", "Avg Test Acc", "Avg Macro F1", "Avg AUC", "Avg ECE"])`  
    `print(df.to_string(index=False))`  
    `print("="*70)`

`if __name__ == "__main__":`  
    `main()`  
**CODE 4 Output:**  
`==================== RUNNING FOLD 1 ON cuda ====================`

`Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.`

`🚀 Found Fold 1 at: /kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20/Original Images/Original Images/FOLDS/fold1`  
  `✅ Train set loaded: 537 images.`  
  `✅ Val set loaded: 144 images.`  
  `✅ Test set loaded: 74 images.`  
`Epoch 1/60 - Loss: 2.8060 | Val Swin: 0.5556 | Val Eff: 0.4306 | Val Fusion: 0.6458`  
`Epoch 2/60 - Loss: 2.0476 | Val Swin: 0.7014 | Val Eff: 0.5556 | Val Fusion: 0.6875`  
`Epoch 3/60 - Loss: 1.7381 | Val Swin: 0.7361 | Val Eff: 0.5903 | Val Fusion: 0.7708`  
`Epoch 4/60 - Loss: 1.8836 | Val Swin: 0.7569 | Val Eff: 0.6458 | Val Fusion: 0.7917`  
`Epoch 5/60 - Loss: 1.5376 | Val Swin: 0.7708 | Val Eff: 0.6944 | Val Fusion: 0.7847`  
`Epoch 6/60 - Loss: 1.5031 | Val Swin: 0.7569 | Val Eff: 0.7292 | Val Fusion: 0.8125`  
`Epoch 7/60 - Loss: 1.3171 | Val Swin: 0.7986 | Val Eff: 0.7847 | Val Fusion: 0.8264`  
`Epoch 8/60 - Loss: 1.5175 | Val Swin: 0.7986 | Val Eff: 0.7917 | Val Fusion: 0.8056`  
`Epoch 9/60 - Loss: 1.3323 | Val Swin: 0.8264 | Val Eff: 0.7847 | Val Fusion: 0.8472`  
`Epoch 10/60 - Loss: 1.5028 | Val Swin: 0.7569 | Val Eff: 0.7917 | Val Fusion: 0.8056`  
`Epoch 11/60 - Loss: 1.1626 | Val Swin: 0.7986 | Val Eff: 0.8125 | Val Fusion: 0.8194`  
`Epoch 12/60 - Loss: 1.4005 | Val Swin: 0.8125 | Val Eff: 0.7917 | Val Fusion: 0.8194`  
`Epoch 13/60 - Loss: 1.3086 | Val Swin: 0.8056 | Val Eff: 0.8056 | Val Fusion: 0.8264`  
`Epoch 14/60 - Loss: 1.1907 | Val Swin: 0.8125 | Val Eff: 0.8403 | Val Fusion: 0.8264`  
`Epoch 15/60 - Loss: 1.5664 | Val Swin: 0.7986 | Val Eff: 0.8194 | Val Fusion: 0.8056`  
`Epoch 16/60 - Loss: 0.9512 | Val Swin: 0.7917 | Val Eff: 0.8125 | Val Fusion: 0.8125`  
`Epoch 17/60 - Loss: 1.3313 | Val Swin: 0.8056 | Val Eff: 0.8403 | Val Fusion: 0.8194`  
`Epoch 18/60 - Loss: 1.2119 | Val Swin: 0.7847 | Val Eff: 0.8056 | Val Fusion: 0.8056`  
`Epoch 19/60 - Loss: 1.2442 | Val Swin: 0.7778 | Val Eff: 0.8056 | Val Fusion: 0.8125`  
`Epoch 20/60 - Loss: 0.9481 | Val Swin: 0.7917 | Val Eff: 0.8472 | Val Fusion: 0.7986`  
`Epoch 21/60 - Loss: 1.3117 | Val Swin: 0.7986 | Val Eff: 0.8611 | Val Fusion: 0.8333`  
`Epoch 22/60 - Loss: 1.1874 | Val Swin: 0.7778 | Val Eff: 0.8403 | Val Fusion: 0.8125`  
`Epoch 23/60 - Loss: 1.1860 | Val Swin: 0.8125 | Val Eff: 0.8125 | Val Fusion: 0.8125`  
`Epoch 24/60 - Loss: 1.1659 | Val Swin: 0.7917 | Val Eff: 0.8611 | Val Fusion: 0.8125`  
`Epoch 25/60 - Loss: 1.0966 | Val Swin: 0.7847 | Val Eff: 0.8472 | Val Fusion: 0.8194`  
`Epoch 26/60 - Loss: 1.3797 | Val Swin: 0.7847 | Val Eff: 0.8264 | Val Fusion: 0.7847`  
`Epoch 27/60 - Loss: 1.1710 | Val Swin: 0.7917 | Val Eff: 0.8264 | Val Fusion: 0.8056`  
`Epoch 28/60 - Loss: 1.1327 | Val Swin: 0.8056 | Val Eff: 0.8611 | Val Fusion: 0.8056`  
`Epoch 29/60 - Loss: 0.8689 | Val Swin: 0.7847 | Val Eff: 0.8333 | Val Fusion: 0.7917`  
`Epoch 30/60 - Loss: 1.0095 | Val Swin: 0.7778 | Val Eff: 0.8264 | Val Fusion: 0.7778`  
`Epoch 31/60 - Loss: 1.0211 | Val Swin: 0.7778 | Val Eff: 0.8472 | Val Fusion: 0.7917`  
`Epoch 32/60 - Loss: 1.0908 | Val Swin: 0.7847 | Val Eff: 0.8403 | Val Fusion: 0.7778`  
`Epoch 33/60 - Loss: 1.1020 | Val Swin: 0.7639 | Val Eff: 0.8333 | Val Fusion: 0.7986`  
`Epoch 34/60 - Loss: 1.0527 | Val Swin: 0.7986 | Val Eff: 0.8194 | Val Fusion: 0.8056`  
`Epoch 35/60 - Loss: 0.9488 | Val Swin: 0.7708 | Val Eff: 0.8403 | Val Fusion: 0.7917`  
`Epoch 36/60 - Loss: 1.1371 | Val Swin: 0.7847 | Val Eff: 0.8264 | Val Fusion: 0.7986`  
`Epoch 37/60 - Loss: 0.9573 | Val Swin: 0.7778 | Val Eff: 0.8333 | Val Fusion: 0.7847`  
`Epoch 38/60 - Loss: 1.1185 | Val Swin: 0.7917 | Val Eff: 0.8611 | Val Fusion: 0.7847`  
`Epoch 39/60 - Loss: 1.1460 | Val Swin: 0.7986 | Val Eff: 0.8333 | Val Fusion: 0.7986`  
`Epoch 40/60 - Loss: 1.0012 | Val Swin: 0.7778 | Val Eff: 0.8611 | Val Fusion: 0.7917`  
`Epoch 41/60 - Loss: 1.0238 | Val Swin: 0.7986 | Val Eff: 0.8611 | Val Fusion: 0.7986`  
`Epoch 42/60 - Loss: 0.9995 | Val Swin: 0.7917 | Val Eff: 0.8542 | Val Fusion: 0.7847`  
`Epoch 43/60 - Loss: 1.0432 | Val Swin: 0.7917 | Val Eff: 0.8472 | Val Fusion: 0.7847`  
`Epoch 44/60 - Loss: 1.0663 | Val Swin: 0.8056 | Val Eff: 0.8403 | Val Fusion: 0.8056`  
`Epoch 45/60 - Loss: 1.0806 | Val Swin: 0.7917 | Val Eff: 0.8611 | Val Fusion: 0.7986`  
`Epoch 46/60 - Loss: 1.1470 | Val Swin: 0.7986 | Val Eff: 0.8472 | Val Fusion: 0.7917`  
`Epoch 47/60 - Loss: 0.9364 | Val Swin: 0.7917 | Val Eff: 0.8611 | Val Fusion: 0.7986`  
`Epoch 48/60 - Loss: 1.0042 | Val Swin: 0.7917 | Val Eff: 0.8611 | Val Fusion: 0.7917`  
`Epoch 49/60 - Loss: 1.2446 | Val Swin: 0.7917 | Val Eff: 0.8125 | Val Fusion: 0.7986`  
`Epoch 50/60 - Loss: 0.9814 | Val Swin: 0.8056 | Val Eff: 0.8472 | Val Fusion: 0.8056`  
`Epoch 51/60 - Loss: 1.1714 | Val Swin: 0.8056 | Val Eff: 0.8403 | Val Fusion: 0.7986`  
`Epoch 52/60 - Loss: 1.1377 | Val Swin: 0.7986 | Val Eff: 0.8542 | Val Fusion: 0.7986`  
`Epoch 53/60 - Loss: 0.9753 | Val Swin: 0.8056 | Val Eff: 0.8403 | Val Fusion: 0.7986`  
`Epoch 54/60 - Loss: 1.0595 | Val Swin: 0.7986 | Val Eff: 0.8333 | Val Fusion: 0.8056`  
`Epoch 55/60 - Loss: 1.1685 | Val Swin: 0.7917 | Val Eff: 0.8472 | Val Fusion: 0.7917`  
`Epoch 56/60 - Loss: 1.0628 | Val Swin: 0.7986 | Val Eff: 0.8472 | Val Fusion: 0.8056`  
`Epoch 57/60 - Loss: 1.0376 | Val Swin: 0.7917 | Val Eff: 0.8333 | Val Fusion: 0.7847`  
`Epoch 58/60 - Loss: 1.1897 | Val Swin: 0.7847 | Val Eff: 0.8333 | Val Fusion: 0.8125`  
`Epoch 59/60 - Loss: 1.2590 | Val Swin: 0.7778 | Val Eff: 0.8403 | Val Fusion: 0.7917`  
`Epoch 60/60 - Loss: 1.0568 | Val Swin: 0.8056 | Val Eff: 0.8403 | Val Fusion: 0.8264`

`--- FOLD 1 TRAINING COMPLETE ---`  
`Loaded best validation model for Fold 1 successfully.`  
`Optimal Temperature found: -20.8638`

`==================== RUNNING FOLD 2 ON cuda ====================`  
`🚀 Found Fold 2 at: /kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20/Original Images/Original Images/FOLDS/fold2`  
  `✅ Train set loaded: 509 images.`  
  `✅ Val set loaded: 164 images.`  
  `✅ Test set loaded: 82 images.`  
`Epoch 1/60 - Loss: 3.0570 | Val Swin: 0.6585 | Val Eff: 0.3232 | Val Fusion: 0.7317`  
`Epoch 2/60 - Loss: 2.4772 | Val Swin: 0.7256 | Val Eff: 0.5061 | Val Fusion: 0.7988`  
`Epoch 3/60 - Loss: 2.3109 | Val Swin: 0.7683 | Val Eff: 0.6463 | Val Fusion: 0.8476`  
`Epoch 4/60 - Loss: 1.7965 | Val Swin: 0.8232 | Val Eff: 0.7195 | Val Fusion: 0.8293`  
`Epoch 5/60 - Loss: 1.7201 | Val Swin: 0.8049 | Val Eff: 0.7378 | Val Fusion: 0.8049`  
`Epoch 6/60 - Loss: 1.4631 | Val Swin: 0.8110 | Val Eff: 0.7622 | Val Fusion: 0.8293`  
`Epoch 7/60 - Loss: 1.4728 | Val Swin: 0.8293 | Val Eff: 0.7561 | Val Fusion: 0.8415`  
`Epoch 8/60 - Loss: 1.5468 | Val Swin: 0.8598 | Val Eff: 0.7317 | Val Fusion: 0.8293`  
`Epoch 9/60 - Loss: 1.5580 | Val Swin: 0.8537 | Val Eff: 0.7500 | Val Fusion: 0.8415`  
`Epoch 10/60 - Loss: 1.4098 | Val Swin: 0.8780 | Val Eff: 0.7378 | Val Fusion: 0.8659`  
`Epoch 11/60 - Loss: 1.4856 | Val Swin: 0.8841 | Val Eff: 0.7378 | Val Fusion: 0.8354`  
`Epoch 12/60 - Loss: 1.2153 | Val Swin: 0.8354 | Val Eff: 0.7561 | Val Fusion: 0.8110`  
`Epoch 13/60 - Loss: 1.2328 | Val Swin: 0.8537 | Val Eff: 0.7866 | Val Fusion: 0.8537`  
`Epoch 14/60 - Loss: 1.1726 | Val Swin: 0.8598 | Val Eff: 0.8171 | Val Fusion: 0.8841`  
`Epoch 15/60 - Loss: 1.3358 | Val Swin: 0.8780 | Val Eff: 0.8110 | Val Fusion: 0.8841`  
`Epoch 16/60 - Loss: 1.6051 | Val Swin: 0.8537 | Val Eff: 0.8110 | Val Fusion: 0.8780`  
`Epoch 17/60 - Loss: 1.3580 | Val Swin: 0.8720 | Val Eff: 0.7866 | Val Fusion: 0.8902`  
`Epoch 18/60 - Loss: 1.1273 | Val Swin: 0.8659 | Val Eff: 0.8049 | Val Fusion: 0.8537`  
`Epoch 19/60 - Loss: 1.5057 | Val Swin: 0.8537 | Val Eff: 0.8354 | Val Fusion: 0.9024`  
`Epoch 20/60 - Loss: 1.2895 | Val Swin: 0.8476 | Val Eff: 0.8171 | Val Fusion: 0.8720`  
`Epoch 21/60 - Loss: 1.1438 | Val Swin: 0.8293 | Val Eff: 0.7927 | Val Fusion: 0.8659`  
`Epoch 22/60 - Loss: 1.0499 | Val Swin: 0.8598 | Val Eff: 0.7988 | Val Fusion: 0.8841`  
`Epoch 23/60 - Loss: 1.2077 | Val Swin: 0.8598 | Val Eff: 0.7927 | Val Fusion: 0.8963`  
`Epoch 24/60 - Loss: 1.1393 | Val Swin: 0.8841 | Val Eff: 0.7622 | Val Fusion: 0.8720`  
`Epoch 25/60 - Loss: 1.4738 | Val Swin: 0.8537 | Val Eff: 0.8293 | Val Fusion: 0.8659`  
`Epoch 26/60 - Loss: 1.3647 | Val Swin: 0.8293 | Val Eff: 0.7866 | Val Fusion: 0.8659`  
`Epoch 27/60 - Loss: 1.2303 | Val Swin: 0.8659 | Val Eff: 0.7866 | Val Fusion: 0.8659`  
`Epoch 28/60 - Loss: 1.0434 | Val Swin: 0.8659 | Val Eff: 0.7988 | Val Fusion: 0.8720`  
`Epoch 29/60 - Loss: 1.2587 | Val Swin: 0.8537 | Val Eff: 0.8110 | Val Fusion: 0.8598`  
`Epoch 30/60 - Loss: 1.0503 | Val Swin: 0.8780 | Val Eff: 0.7927 | Val Fusion: 0.8902`  
`Epoch 31/60 - Loss: 1.2956 | Val Swin: 0.8659 | Val Eff: 0.7927 | Val Fusion: 0.8780`  
`Epoch 32/60 - Loss: 0.9541 | Val Swin: 0.8659 | Val Eff: 0.7927 | Val Fusion: 0.8659`  
`Epoch 33/60 - Loss: 0.9784 | Val Swin: 0.8354 | Val Eff: 0.8232 | Val Fusion: 0.8659`  
`Epoch 34/60 - Loss: 0.9573 | Val Swin: 0.8415 | Val Eff: 0.8171 | Val Fusion: 0.8720`  
`Epoch 35/60 - Loss: 1.2974 | Val Swin: 0.8415 | Val Eff: 0.7988 | Val Fusion: 0.8720`  
`Epoch 36/60 - Loss: 1.1730 | Val Swin: 0.8720 | Val Eff: 0.8110 | Val Fusion: 0.8841`  
`Epoch 37/60 - Loss: 1.0256 | Val Swin: 0.8293 | Val Eff: 0.8171 | Val Fusion: 0.8780`  
`Epoch 38/60 - Loss: 1.1027 | Val Swin: 0.8841 | Val Eff: 0.8049 | Val Fusion: 0.8780`  
`Epoch 39/60 - Loss: 1.2568 | Val Swin: 0.8476 | Val Eff: 0.7927 | Val Fusion: 0.8354`  
`Epoch 40/60 - Loss: 0.9902 | Val Swin: 0.8537 | Val Eff: 0.7988 | Val Fusion: 0.8537`  
`Epoch 41/60 - Loss: 1.1565 | Val Swin: 0.8171 | Val Eff: 0.7988 | Val Fusion: 0.8537`  
`Epoch 42/60 - Loss: 0.9780 | Val Swin: 0.8537 | Val Eff: 0.8049 | Val Fusion: 0.8598`  
`Epoch 43/60 - Loss: 1.0259 | Val Swin: 0.8415 | Val Eff: 0.7988 | Val Fusion: 0.8537`  
`Epoch 44/60 - Loss: 1.2038 | Val Swin: 0.8232 | Val Eff: 0.7988 | Val Fusion: 0.8537`  
`Epoch 45/60 - Loss: 1.0449 | Val Swin: 0.8415 | Val Eff: 0.7988 | Val Fusion: 0.8476`  
`Epoch 46/60 - Loss: 1.0859 | Val Swin: 0.8415 | Val Eff: 0.7927 | Val Fusion: 0.8476`  
`Epoch 47/60 - Loss: 1.2163 | Val Swin: 0.8476 | Val Eff: 0.7988 | Val Fusion: 0.8476`  
`Epoch 48/60 - Loss: 1.0391 | Val Swin: 0.8598 | Val Eff: 0.7866 | Val Fusion: 0.8476`  
`Epoch 49/60 - Loss: 1.1001 | Val Swin: 0.8476 | Val Eff: 0.7927 | Val Fusion: 0.8415`  
`Epoch 50/60 - Loss: 1.1516 | Val Swin: 0.8476 | Val Eff: 0.8110 | Val Fusion: 0.8415`  
`Epoch 51/60 - Loss: 1.0002 | Val Swin: 0.8476 | Val Eff: 0.8110 | Val Fusion: 0.8598`  
`Epoch 52/60 - Loss: 1.1167 | Val Swin: 0.8476 | Val Eff: 0.7988 | Val Fusion: 0.8476`  
`Epoch 53/60 - Loss: 1.3254 | Val Swin: 0.8659 | Val Eff: 0.8110 | Val Fusion: 0.8537`  
`Epoch 54/60 - Loss: 1.1264 | Val Swin: 0.8293 | Val Eff: 0.7927 | Val Fusion: 0.8415`  
`Epoch 55/60 - Loss: 1.1859 | Val Swin: 0.8476 | Val Eff: 0.8293 | Val Fusion: 0.8537`  
`Epoch 56/60 - Loss: 0.9594 | Val Swin: 0.8293 | Val Eff: 0.8110 | Val Fusion: 0.8537`  
`Epoch 57/60 - Loss: 1.1572 | Val Swin: 0.8232 | Val Eff: 0.7927 | Val Fusion: 0.8598`  
`Epoch 58/60 - Loss: 0.9047 | Val Swin: 0.8659 | Val Eff: 0.7988 | Val Fusion: 0.8780`  
`Epoch 59/60 - Loss: 0.9077 | Val Swin: 0.8598 | Val Eff: 0.8171 | Val Fusion: 0.8720`  
`Epoch 60/60 - Loss: 0.8897 | Val Swin: 0.8659 | Val Eff: 0.8049 | Val Fusion: 0.8720`

`--- FOLD 2 TRAINING COMPLETE ---`  
`Loaded best validation model for Fold 2 successfully.`  
`Optimal Temperature found: 0.9070`

`==================== RUNNING FOLD 3 ON cuda ====================`  
`🚀 Found Fold 3 at: /kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20/Original Images/Original Images/FOLDS/fold3`  
  `✅ Train set loaded: 538 images.`  
  `✅ Val set loaded: 153 images.`  
  `✅ Test set loaded: 64 images.`  
`Epoch 1/60 - Loss: 2.6861 | Val Swin: 0.5686 | Val Eff: 0.2876 | Val Fusion: 0.5817`  
`Epoch 2/60 - Loss: 1.8904 | Val Swin: 0.6340 | Val Eff: 0.4641 | Val Fusion: 0.6863`  
`Epoch 3/60 - Loss: 1.5804 | Val Swin: 0.7124 | Val Eff: 0.5359 | Val Fusion: 0.7386`  
`Epoch 4/60 - Loss: 1.5547 | Val Swin: 0.6928 | Val Eff: 0.6209 | Val Fusion: 0.7124`  
`Epoch 5/60 - Loss: 1.5192 | Val Swin: 0.7712 | Val Eff: 0.6928 | Val Fusion: 0.7647`  
`Epoch 6/60 - Loss: 1.3182 | Val Swin: 0.6993 | Val Eff: 0.7124 | Val Fusion: 0.7843`  
`Epoch 7/60 - Loss: 1.2234 | Val Swin: 0.8039 | Val Eff: 0.7451 | Val Fusion: 0.7974`  
`Epoch 8/60 - Loss: 1.2603 | Val Swin: 0.7516 | Val Eff: 0.7124 | Val Fusion: 0.7778`  
`Epoch 9/60 - Loss: 1.3770 | Val Swin: 0.6797 | Val Eff: 0.7320 | Val Fusion: 0.7386`  
`Epoch 10/60 - Loss: 1.2409 | Val Swin: 0.7582 | Val Eff: 0.7255 | Val Fusion: 0.7778`  
`Epoch 11/60 - Loss: 1.2523 | Val Swin: 0.7712 | Val Eff: 0.7255 | Val Fusion: 0.7908`  
`Epoch 12/60 - Loss: 1.3855 | Val Swin: 0.7712 | Val Eff: 0.7386 | Val Fusion: 0.7974`  
`Epoch 13/60 - Loss: 1.3290 | Val Swin: 0.7647 | Val Eff: 0.7386 | Val Fusion: 0.7778`  
`Epoch 14/60 - Loss: 0.9487 | Val Swin: 0.7386 | Val Eff: 0.7386 | Val Fusion: 0.7778`  
`Epoch 15/60 - Loss: 1.1788 | Val Swin: 0.7712 | Val Eff: 0.7320 | Val Fusion: 0.7843`  
`Epoch 16/60 - Loss: 1.4114 | Val Swin: 0.7843 | Val Eff: 0.7386 | Val Fusion: 0.7843`  
`Epoch 17/60 - Loss: 1.2091 | Val Swin: 0.7908 | Val Eff: 0.7255 | Val Fusion: 0.7843`  
`Epoch 18/60 - Loss: 1.1604 | Val Swin: 0.7908 | Val Eff: 0.7124 | Val Fusion: 0.8105`  
`Epoch 19/60 - Loss: 1.2104 | Val Swin: 0.7582 | Val Eff: 0.7516 | Val Fusion: 0.7778`  
`Epoch 20/60 - Loss: 1.1242 | Val Swin: 0.7516 | Val Eff: 0.7582 | Val Fusion: 0.7712`  
`Epoch 21/60 - Loss: 1.2111 | Val Swin: 0.7582 | Val Eff: 0.7386 | Val Fusion: 0.7778`  
`Epoch 22/60 - Loss: 1.2489 | Val Swin: 0.7908 | Val Eff: 0.7190 | Val Fusion: 0.7974`  
`Epoch 23/60 - Loss: 1.0935 | Val Swin: 0.7778 | Val Eff: 0.7582 | Val Fusion: 0.7843`  
`Epoch 24/60 - Loss: 0.9840 | Val Swin: 0.7778 | Val Eff: 0.7516 | Val Fusion: 0.8039`  
`Epoch 25/60 - Loss: 1.2209 | Val Swin: 0.7843 | Val Eff: 0.7320 | Val Fusion: 0.8105`  
`Epoch 26/60 - Loss: 1.2246 | Val Swin: 0.8039 | Val Eff: 0.7320 | Val Fusion: 0.8105`  
`Epoch 27/60 - Loss: 1.0595 | Val Swin: 0.7516 | Val Eff: 0.7386 | Val Fusion: 0.7582`  
`Epoch 28/60 - Loss: 1.0021 | Val Swin: 0.7712 | Val Eff: 0.7320 | Val Fusion: 0.7974`  
`Epoch 29/60 - Loss: 1.1132 | Val Swin: 0.7908 | Val Eff: 0.7255 | Val Fusion: 0.7908`  
`Epoch 30/60 - Loss: 1.0352 | Val Swin: 0.7647 | Val Eff: 0.7320 | Val Fusion: 0.7843`  
`Epoch 31/60 - Loss: 1.3250 | Val Swin: 0.7908 | Val Eff: 0.7059 | Val Fusion: 0.7974`  
`Epoch 32/60 - Loss: 1.1564 | Val Swin: 0.7974 | Val Eff: 0.7190 | Val Fusion: 0.8039`  
`Epoch 33/60 - Loss: 0.8165 | Val Swin: 0.7843 | Val Eff: 0.7386 | Val Fusion: 0.8170`  
`Epoch 34/60 - Loss: 1.1134 | Val Swin: 0.8105 | Val Eff: 0.7320 | Val Fusion: 0.8039`  
`Epoch 35/60 - Loss: 1.1732 | Val Swin: 0.7974 | Val Eff: 0.7386 | Val Fusion: 0.8105`  
`Epoch 36/60 - Loss: 1.0956 | Val Swin: 0.7778 | Val Eff: 0.7255 | Val Fusion: 0.7974`  
`Epoch 37/60 - Loss: 1.2311 | Val Swin: 0.8039 | Val Eff: 0.7516 | Val Fusion: 0.8039`  
`Epoch 38/60 - Loss: 1.0686 | Val Swin: 0.8170 | Val Eff: 0.7582 | Val Fusion: 0.7974`  
`Epoch 39/60 - Loss: 1.2294 | Val Swin: 0.8170 | Val Eff: 0.7190 | Val Fusion: 0.8039`  
`Epoch 40/60 - Loss: 1.4968 | Val Swin: 0.8105 | Val Eff: 0.7059 | Val Fusion: 0.7974`  
`Epoch 41/60 - Loss: 1.1595 | Val Swin: 0.7974 | Val Eff: 0.7320 | Val Fusion: 0.7974`  
`Epoch 42/60 - Loss: 1.1114 | Val Swin: 0.7974 | Val Eff: 0.7190 | Val Fusion: 0.8170`  
`Epoch 43/60 - Loss: 0.7249 | Val Swin: 0.8039 | Val Eff: 0.7451 | Val Fusion: 0.8039`  
`Epoch 44/60 - Loss: 1.1209 | Val Swin: 0.8039 | Val Eff: 0.7516 | Val Fusion: 0.8105`  
`Epoch 45/60 - Loss: 0.9438 | Val Swin: 0.8170 | Val Eff: 0.7582 | Val Fusion: 0.8170`  
`Epoch 46/60 - Loss: 1.2121 | Val Swin: 0.8039 | Val Eff: 0.7190 | Val Fusion: 0.7974`  
`Epoch 47/60 - Loss: 1.2471 | Val Swin: 0.8039 | Val Eff: 0.7582 | Val Fusion: 0.8170`  
`Epoch 48/60 - Loss: 1.2208 | Val Swin: 0.8170 | Val Eff: 0.7451 | Val Fusion: 0.8235`  
`Epoch 49/60 - Loss: 1.1291 | Val Swin: 0.8105 | Val Eff: 0.7451 | Val Fusion: 0.8105`  
`Epoch 50/60 - Loss: 1.2193 | Val Swin: 0.8170 | Val Eff: 0.7255 | Val Fusion: 0.8235`  
`Epoch 51/60 - Loss: 0.9111 | Val Swin: 0.8235 | Val Eff: 0.7451 | Val Fusion: 0.8235`  
`Epoch 52/60 - Loss: 1.2954 | Val Swin: 0.8235 | Val Eff: 0.7190 | Val Fusion: 0.8105`  
`Epoch 53/60 - Loss: 1.1584 | Val Swin: 0.8105 | Val Eff: 0.7190 | Val Fusion: 0.8170`  
`Epoch 54/60 - Loss: 1.2658 | Val Swin: 0.8105 | Val Eff: 0.7386 | Val Fusion: 0.8105`  
`Epoch 55/60 - Loss: 1.3102 | Val Swin: 0.8039 | Val Eff: 0.7190 | Val Fusion: 0.8039`  
`Epoch 56/60 - Loss: 1.1122 | Val Swin: 0.8105 | Val Eff: 0.7516 | Val Fusion: 0.7974`  
`Epoch 57/60 - Loss: 1.1121 | Val Swin: 0.7908 | Val Eff: 0.7255 | Val Fusion: 0.7908`  
`Epoch 58/60 - Loss: 1.3754 | Val Swin: 0.8039 | Val Eff: 0.7255 | Val Fusion: 0.8105`  
`Epoch 59/60 - Loss: 0.9466 | Val Swin: 0.7974 | Val Eff: 0.7451 | Val Fusion: 0.8105`  
`Epoch 60/60 - Loss: 1.1347 | Val Swin: 0.7908 | Val Eff: 0.7451 | Val Fusion: 0.7778`

`--- FOLD 3 TRAINING COMPLETE ---`  
`Loaded best validation model for Fold 3 successfully.`  
`Optimal Temperature found: 1.2662`

`==================== RUNNING FOLD 4 ON cuda ====================`  
`🚀 Found Fold 4 at: /kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20/Original Images/Original Images/FOLDS/fold4`  
  `✅ Train set loaded: 524 images.`  
  `✅ Val set loaded: 150 images.`  
  `✅ Test set loaded: 81 images.`  
`Epoch 1/60 - Loss: 2.9384 | Val Swin: 0.6267 | Val Eff: 0.2800 | Val Fusion: 0.6533`  
`Epoch 2/60 - Loss: 2.2379 | Val Swin: 0.7267 | Val Eff: 0.4067 | Val Fusion: 0.6933`  
`Epoch 3/60 - Loss: 1.7001 | Val Swin: 0.7667 | Val Eff: 0.6267 | Val Fusion: 0.7800`  
`Epoch 4/60 - Loss: 1.8551 | Val Swin: 0.7867 | Val Eff: 0.6400 | Val Fusion: 0.8133`  
`Epoch 5/60 - Loss: 1.7144 | Val Swin: 0.7733 | Val Eff: 0.6867 | Val Fusion: 0.7933`  
`Epoch 6/60 - Loss: 1.4943 | Val Swin: 0.8200 | Val Eff: 0.7400 | Val Fusion: 0.8200`  
`Epoch 7/60 - Loss: 1.6211 | Val Swin: 0.8067 | Val Eff: 0.7133 | Val Fusion: 0.8200`  
`Epoch 8/60 - Loss: 1.4950 | Val Swin: 0.8200 | Val Eff: 0.7867 | Val Fusion: 0.8200`  
`Epoch 9/60 - Loss: 1.7225 | Val Swin: 0.8000 | Val Eff: 0.7533 | Val Fusion: 0.8400`  
`Epoch 10/60 - Loss: 0.9918 | Val Swin: 0.7733 | Val Eff: 0.7533 | Val Fusion: 0.8200`  
`Epoch 11/60 - Loss: 1.3707 | Val Swin: 0.8200 | Val Eff: 0.7533 | Val Fusion: 0.8333`  
`Epoch 12/60 - Loss: 1.3288 | Val Swin: 0.8333 | Val Eff: 0.7800 | Val Fusion: 0.8267`  
`Epoch 13/60 - Loss: 1.1788 | Val Swin: 0.8333 | Val Eff: 0.7800 | Val Fusion: 0.8400`  
`Epoch 14/60 - Loss: 1.3592 | Val Swin: 0.8467 | Val Eff: 0.7933 | Val Fusion: 0.8333`  
`Epoch 15/60 - Loss: 1.0456 | Val Swin: 0.8400 | Val Eff: 0.8133 | Val Fusion: 0.8533`  
`Epoch 16/60 - Loss: 1.1310 | Val Swin: 0.8000 | Val Eff: 0.8133 | Val Fusion: 0.8533`  
`Epoch 17/60 - Loss: 1.1796 | Val Swin: 0.8267 | Val Eff: 0.8067 | Val Fusion: 0.8667`  
`Epoch 18/60 - Loss: 1.0561 | Val Swin: 0.8533 | Val Eff: 0.8133 | Val Fusion: 0.8533`  
`Epoch 19/60 - Loss: 1.1675 | Val Swin: 0.8533 | Val Eff: 0.8267 | Val Fusion: 0.8467`  
`Epoch 20/60 - Loss: 1.3642 | Val Swin: 0.8200 | Val Eff: 0.8067 | Val Fusion: 0.8200`  
`Epoch 21/60 - Loss: 1.2015 | Val Swin: 0.8467 | Val Eff: 0.8200 | Val Fusion: 0.8467`  
`Epoch 22/60 - Loss: 1.2705 | Val Swin: 0.8467 | Val Eff: 0.8000 | Val Fusion: 0.8400`  
`Epoch 23/60 - Loss: 1.2560 | Val Swin: 0.8600 | Val Eff: 0.8333 | Val Fusion: 0.8467`  
`Epoch 24/60 - Loss: 1.2676 | Val Swin: 0.8400 | Val Eff: 0.8267 | Val Fusion: 0.8200`  
`Epoch 25/60 - Loss: 1.1923 | Val Swin: 0.8267 | Val Eff: 0.8267 | Val Fusion: 0.8333`  
`Epoch 26/60 - Loss: 1.0381 | Val Swin: 0.8333 | Val Eff: 0.8400 | Val Fusion: 0.8333`  
`Epoch 27/60 - Loss: 1.1180 | Val Swin: 0.8333 | Val Eff: 0.8133 | Val Fusion: 0.8333`  
`Epoch 28/60 - Loss: 1.0022 | Val Swin: 0.8333 | Val Eff: 0.8267 | Val Fusion: 0.8333`  
`Epoch 29/60 - Loss: 1.1701 | Val Swin: 0.8333 | Val Eff: 0.7533 | Val Fusion: 0.8333`  
`Epoch 30/60 - Loss: 1.2705 | Val Swin: 0.8467 | Val Eff: 0.8000 | Val Fusion: 0.8200`  
`Epoch 31/60 - Loss: 1.1137 | Val Swin: 0.8333 | Val Eff: 0.8200 | Val Fusion: 0.8533`  
`Epoch 32/60 - Loss: 1.1365 | Val Swin: 0.8133 | Val Eff: 0.8267 | Val Fusion: 0.8267`  
`Epoch 33/60 - Loss: 0.9488 | Val Swin: 0.8400 | Val Eff: 0.8067 | Val Fusion: 0.8267`  
`Epoch 34/60 - Loss: 1.3133 | Val Swin: 0.8333 | Val Eff: 0.8333 | Val Fusion: 0.8333`  
`Epoch 35/60 - Loss: 1.2499 | Val Swin: 0.8600 | Val Eff: 0.8333 | Val Fusion: 0.8600`  
`Epoch 36/60 - Loss: 0.9909 | Val Swin: 0.8600 | Val Eff: 0.8533 | Val Fusion: 0.8600`  
`Epoch 37/60 - Loss: 1.2810 | Val Swin: 0.8733 | Val Eff: 0.8467 | Val Fusion: 0.8533`  
`Epoch 38/60 - Loss: 1.1338 | Val Swin: 0.8533 | Val Eff: 0.8267 | Val Fusion: 0.8667`  
`Epoch 39/60 - Loss: 1.1812 | Val Swin: 0.8533 | Val Eff: 0.8333 | Val Fusion: 0.8467`  
`Epoch 40/60 - Loss: 0.8748 | Val Swin: 0.8533 | Val Eff: 0.8467 | Val Fusion: 0.8533`  
`Epoch 41/60 - Loss: 1.2373 | Val Swin: 0.8467 | Val Eff: 0.8133 | Val Fusion: 0.8400`  
`Epoch 42/60 - Loss: 0.9735 | Val Swin: 0.8533 | Val Eff: 0.8400 | Val Fusion: 0.8400`  
`Epoch 43/60 - Loss: 1.3066 | Val Swin: 0.8667 | Val Eff: 0.8067 | Val Fusion: 0.8400`  
`Epoch 44/60 - Loss: 1.1865 | Val Swin: 0.8600 | Val Eff: 0.8400 | Val Fusion: 0.8400`  
`Epoch 45/60 - Loss: 1.1739 | Val Swin: 0.8600 | Val Eff: 0.8333 | Val Fusion: 0.8467`  
`Epoch 46/60 - Loss: 1.0484 | Val Swin: 0.8600 | Val Eff: 0.8467 | Val Fusion: 0.8467`  
`Epoch 47/60 - Loss: 1.0966 | Val Swin: 0.8667 | Val Eff: 0.8333 | Val Fusion: 0.8533`  
`Epoch 48/60 - Loss: 1.2607 | Val Swin: 0.8600 | Val Eff: 0.8333 | Val Fusion: 0.8467`  
`Epoch 49/60 - Loss: 1.0452 | Val Swin: 0.8667 | Val Eff: 0.8267 | Val Fusion: 0.8467`  
`Epoch 50/60 - Loss: 0.9841 | Val Swin: 0.8600 | Val Eff: 0.8200 | Val Fusion: 0.8467`  
`Epoch 51/60 - Loss: 1.0003 | Val Swin: 0.8600 | Val Eff: 0.8333 | Val Fusion: 0.8467`  
`Epoch 52/60 - Loss: 1.0550 | Val Swin: 0.8467 | Val Eff: 0.8133 | Val Fusion: 0.8467`  
`Epoch 53/60 - Loss: 1.0771 | Val Swin: 0.8333 | Val Eff: 0.8267 | Val Fusion: 0.8333`  
`Epoch 54/60 - Loss: 1.0032 | Val Swin: 0.8667 | Val Eff: 0.8333 | Val Fusion: 0.8333`  
`Epoch 55/60 - Loss: 0.9212 | Val Swin: 0.8333 | Val Eff: 0.8333 | Val Fusion: 0.8467`  
`Epoch 56/60 - Loss: 1.3093 | Val Swin: 0.8600 | Val Eff: 0.8267 | Val Fusion: 0.8400`  
`Epoch 57/60 - Loss: 0.9287 | Val Swin: 0.8533 | Val Eff: 0.8533 | Val Fusion: 0.8533`  
`Epoch 58/60 - Loss: 1.1638 | Val Swin: 0.8400 | Val Eff: 0.8600 | Val Fusion: 0.8400`  
`Epoch 59/60 - Loss: 1.1275 | Val Swin: 0.8667 | Val Eff: 0.8467 | Val Fusion: 0.8533`  
`Epoch 60/60 - Loss: 1.2333 | Val Swin: 0.8600 | Val Eff: 0.8467 | Val Fusion: 0.8533`

`--- FOLD 4 TRAINING COMPLETE ---`  
`Loaded best validation model for Fold 4 successfully.`  
`Optimal Temperature found: -11.3054`

`==================== RUNNING FOLD 5 ON cuda ====================`  
`🚀 Found Fold 5 at: /kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20/Original Images/Original Images/FOLDS/fold5`  
  `✅ Train set loaded: 538 images.`  
  `✅ Val set loaded: 154 images.`  
  `✅ Test set loaded: 63 images.`  
`Epoch 1/60 - Loss: 3.0546 | Val Swin: 0.6039 | Val Eff: 0.4091 | Val Fusion: 0.5714`  
`Epoch 2/60 - Loss: 2.2624 | Val Swin: 0.6883 | Val Eff: 0.5649 | Val Fusion: 0.7208`  
`Epoch 3/60 - Loss: 2.0645 | Val Swin: 0.7078 | Val Eff: 0.6818 | Val Fusion: 0.7403`  
`Epoch 4/60 - Loss: 1.7608 | Val Swin: 0.7143 | Val Eff: 0.6558 | Val Fusion: 0.7727`  
`Epoch 5/60 - Loss: 1.7388 | Val Swin: 0.7662 | Val Eff: 0.6753 | Val Fusion: 0.7922`  
`Epoch 6/60 - Loss: 1.5690 | Val Swin: 0.7857 | Val Eff: 0.7273 | Val Fusion: 0.7922`  
`Epoch 7/60 - Loss: 1.7619 | Val Swin: 0.7792 | Val Eff: 0.7078 | Val Fusion: 0.8117`  
`Epoch 8/60 - Loss: 1.4914 | Val Swin: 0.7727 | Val Eff: 0.7727 | Val Fusion: 0.7987`  
`Epoch 9/60 - Loss: 1.5855 | Val Swin: 0.7857 | Val Eff: 0.7922 | Val Fusion: 0.7987`  
`Epoch 10/60 - Loss: 0.9492 | Val Swin: 0.7987 | Val Eff: 0.7922 | Val Fusion: 0.8117`  
`Epoch 11/60 - Loss: 1.6987 | Val Swin: 0.7662 | Val Eff: 0.7597 | Val Fusion: 0.8052`  
`Epoch 12/60 - Loss: 1.3642 | Val Swin: 0.7987 | Val Eff: 0.7857 | Val Fusion: 0.8052`  
`Epoch 13/60 - Loss: 1.3054 | Val Swin: 0.7792 | Val Eff: 0.8117 | Val Fusion: 0.8117`  
`Epoch 14/60 - Loss: 1.2386 | Val Swin: 0.7662 | Val Eff: 0.7727 | Val Fusion: 0.7987`  
`Epoch 15/60 - Loss: 1.1959 | Val Swin: 0.7922 | Val Eff: 0.8052 | Val Fusion: 0.7987`  
`Epoch 16/60 - Loss: 1.1797 | Val Swin: 0.7857 | Val Eff: 0.8052 | Val Fusion: 0.8182`  
`Epoch 17/60 - Loss: 1.2200 | Val Swin: 0.7532 | Val Eff: 0.8117 | Val Fusion: 0.7727`  
`Epoch 18/60 - Loss: 1.4858 | Val Swin: 0.7922 | Val Eff: 0.7727 | Val Fusion: 0.7792`  
`Epoch 19/60 - Loss: 1.0102 | Val Swin: 0.7792 | Val Eff: 0.8052 | Val Fusion: 0.7857`  
`Epoch 20/60 - Loss: 1.0266 | Val Swin: 0.7662 | Val Eff: 0.7792 | Val Fusion: 0.7922`  
`Epoch 21/60 - Loss: 1.2546 | Val Swin: 0.7662 | Val Eff: 0.8312 | Val Fusion: 0.7727`  
`Epoch 22/60 - Loss: 1.1753 | Val Swin: 0.7662 | Val Eff: 0.8182 | Val Fusion: 0.7727`  
`Epoch 23/60 - Loss: 1.0227 | Val Swin: 0.7987 | Val Eff: 0.7987 | Val Fusion: 0.8052`  
`Epoch 24/60 - Loss: 1.1017 | Val Swin: 0.7792 | Val Eff: 0.8117 | Val Fusion: 0.7922`  
`Epoch 25/60 - Loss: 1.1676 | Val Swin: 0.7922 | Val Eff: 0.8506 | Val Fusion: 0.8052`  
`Epoch 26/60 - Loss: 0.9964 | Val Swin: 0.7922 | Val Eff: 0.8377 | Val Fusion: 0.8117`  
`Epoch 27/60 - Loss: 1.2109 | Val Swin: 0.7987 | Val Eff: 0.7922 | Val Fusion: 0.8247`  
`Epoch 28/60 - Loss: 1.2593 | Val Swin: 0.7922 | Val Eff: 0.7922 | Val Fusion: 0.7792`  
`Epoch 29/60 - Loss: 1.4343 | Val Swin: 0.7922 | Val Eff: 0.8377 | Val Fusion: 0.8247`  
`Epoch 30/60 - Loss: 1.1771 | Val Swin: 0.7857 | Val Eff: 0.7987 | Val Fusion: 0.7922`  
`Epoch 31/60 - Loss: 1.2274 | Val Swin: 0.7857 | Val Eff: 0.8312 | Val Fusion: 0.7922`  
`Epoch 32/60 - Loss: 0.9559 | Val Swin: 0.7857 | Val Eff: 0.8052 | Val Fusion: 0.8182`  
`Epoch 33/60 - Loss: 1.0056 | Val Swin: 0.7987 | Val Eff: 0.8117 | Val Fusion: 0.7987`  
`Epoch 34/60 - Loss: 1.1938 | Val Swin: 0.7987 | Val Eff: 0.8182 | Val Fusion: 0.8117`  
`Epoch 35/60 - Loss: 1.0774 | Val Swin: 0.7922 | Val Eff: 0.8052 | Val Fusion: 0.8247`  
`Epoch 36/60 - Loss: 1.0479 | Val Swin: 0.7987 | Val Eff: 0.8247 | Val Fusion: 0.8052`  
`Epoch 37/60 - Loss: 1.1938 | Val Swin: 0.7792 | Val Eff: 0.8117 | Val Fusion: 0.7987`  
`Epoch 38/60 - Loss: 1.1315 | Val Swin: 0.7922 | Val Eff: 0.8182 | Val Fusion: 0.7922`  
`Epoch 39/60 - Loss: 0.9921 | Val Swin: 0.7727 | Val Eff: 0.8182 | Val Fusion: 0.7857`  
`Epoch 40/60 - Loss: 1.2158 | Val Swin: 0.7987 | Val Eff: 0.8117 | Val Fusion: 0.7987`  
`Epoch 41/60 - Loss: 0.8543 | Val Swin: 0.7987 | Val Eff: 0.8052 | Val Fusion: 0.8052`  
`Epoch 42/60 - Loss: 1.0404 | Val Swin: 0.7792 | Val Eff: 0.8247 | Val Fusion: 0.7987`  
`Epoch 43/60 - Loss: 1.2135 | Val Swin: 0.7857 | Val Eff: 0.7987 | Val Fusion: 0.7857`  
`Epoch 44/60 - Loss: 1.0615 | Val Swin: 0.7857 | Val Eff: 0.8117 | Val Fusion: 0.8052`  
`Epoch 45/60 - Loss: 1.3534 | Val Swin: 0.7792 | Val Eff: 0.8052 | Val Fusion: 0.7857`  
`Epoch 46/60 - Loss: 1.1581 | Val Swin: 0.7727 | Val Eff: 0.8052 | Val Fusion: 0.7922`  
`Epoch 47/60 - Loss: 1.0875 | Val Swin: 0.7727 | Val Eff: 0.8117 | Val Fusion: 0.7922`  
`Epoch 48/60 - Loss: 1.3630 | Val Swin: 0.7792 | Val Eff: 0.8182 | Val Fusion: 0.8117`  
`Epoch 49/60 - Loss: 1.0858 | Val Swin: 0.7857 | Val Eff: 0.7922 | Val Fusion: 0.7922`  
`Epoch 50/60 - Loss: 1.1144 | Val Swin: 0.7857 | Val Eff: 0.8117 | Val Fusion: 0.8117`  
`Epoch 51/60 - Loss: 1.1407 | Val Swin: 0.7792 | Val Eff: 0.8182 | Val Fusion: 0.8117`  
`Epoch 52/60 - Loss: 1.0300 | Val Swin: 0.7857 | Val Eff: 0.8052 | Val Fusion: 0.8052`  
`Epoch 53/60 - Loss: 1.0329 | Val Swin: 0.7922 | Val Eff: 0.8312 | Val Fusion: 0.8117`  
`Epoch 54/60 - Loss: 1.1702 | Val Swin: 0.7922 | Val Eff: 0.8247 | Val Fusion: 0.8052`  
`Epoch 55/60 - Loss: 1.1106 | Val Swin: 0.7922 | Val Eff: 0.8182 | Val Fusion: 0.8247`  
`Epoch 56/60 - Loss: 0.9849 | Val Swin: 0.7857 | Val Eff: 0.8247 | Val Fusion: 0.8182`  
`Epoch 57/60 - Loss: 1.2906 | Val Swin: 0.7857 | Val Eff: 0.8506 | Val Fusion: 0.8182`  
`Epoch 58/60 - Loss: 1.1924 | Val Swin: 0.7922 | Val Eff: 0.8247 | Val Fusion: 0.7922`  
`Epoch 59/60 - Loss: 1.0665 | Val Swin: 0.7857 | Val Eff: 0.8247 | Val Fusion: 0.7987`  
`Epoch 60/60 - Loss: 1.2842 | Val Swin: 0.7727 | Val Eff: 0.8182 | Val Fusion: 0.7857`

`--- FOLD 5 TRAINING COMPLETE ---`  
`Loaded best validation model for Fold 5 successfully.`  
`Optimal Temperature found: 1.0326`

`======================================================================`  
`🏆 AGGREGATED 5-FOLD MASTER ABLATION STUDY RESULTS 🏆`  
`======================================================================`  
      `Variant Avg Val Acc Avg Test Acc Avg Macro F1 Avg AUC Avg ECE`  
    `Swin Only      0.8245       0.8278       0.8058  0.9590  0.0771`  
  `EffNet Only      0.7928       0.7810       0.7597  0.9538  0.1310`  
  `Base Fusion      0.8529       0.8296       0.8040  0.9648  0.0985`  
 `Fusion + SWA           -       0.8396       0.8157  0.9647  0.0717`  
`Fusion + Temp           -       0.4971       0.4715  0.5904  0.1448`  
`Stacking (RF)           -       0.8378       0.8126  0.9538  0.0956`  
`======================================================================`

`/tmp/ipykernel_24/4104621220.py:517: RuntimeWarning: Mean of empty slice`  
  `avg_metrics = np.nanmean(variant_metrics, axis=0)`

**CODE 5:**  
`import os`  
`import random`  
`import glob`  
`import warnings`  
`import numpy as np`  
`import pandas as pd`  
`import matplotlib.pyplot as plt`  
`import seaborn as sns`

`import torch`  
`import torch.nn as nn`  
`import torch.optim as optim`  
`from torch.utils.data import Dataset, DataLoader`  
`from torch.amp import autocast, GradScaler`  
`from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn`

`import torchvision.transforms as transforms`  
`from PIL import Image`

`import timm`  
`from sklearn.model_selection import StratifiedGroupKFold`  
`from sklearn.metrics import (`  
    `accuracy_score, f1_score, roc_auc_score, confusion_matrix`  
`)`  
`from sklearn.utils.class_weight import compute_class_weight`

`warnings.filterwarnings('ignore')`

*`# ==========================================`*  
*`# 1. CONFIGURATION & SEEDING`*  
*`# ==========================================`*  
`class Config:`  
    `SEED = 42`  
    `NUM_CLASSES = 6`  
    `BATCH_SIZE = 16`   
    `IMG_SIZE = 384`  
      
    `MODEL_A = 'swin_tiny_patch4_window7_224'`   
    `MODEL_B = 'tf_efficientnetv2_s'`  
      
    `EPOCHS = 40`  
    `LR_HEAD = 3e-4`  
    `LR_BACKBONE = 1e-4`  
    `WEIGHT_DECAY = 1e-4`  
    `PATIENCE = 7`  
    `EMA_DECAY = 0.995`  
      
    `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`  
    `SAVE_DIR = "/kaggle/working/pipeline_results"`  
      
    `# 🚨 FIX: Updated to match the exact folder names in the Kaggle dataset!`  
    `CLASS_MAP = {`  
        `'HEALTHY': 0,`   
        `'COWPOX': 1,`   
        `'MONKEYPOX': 2,`   
        `'MEASLES': 3,`   
        `'CHICKENPOX': 4,`   
        `'HFMD': 5`  
    `}`

`def seed_everything(seed):`  
    `random.seed(seed)`  
    `np.random.seed(seed)`  
    `torch.manual_seed(seed)`  
    `torch.cuda.manual_seed_all(seed)`  
    `torch.backends.cudnn.deterministic = True`  
    `torch.backends.cudnn.benchmark = False`

`seed_everything(Config.SEED)`

*`# ==========================================`*  
*`# 2. DATA HANDLING & INTEGRITY CHECKS`*  
*`# ==========================================`*  
`def verify_no_patient_overlap(train_df, val_df):`  
    `"""Fast-fail assertion to guarantee zero patient leakage between folds."""`  
    `train_patients = set(train_df['patient_id'].unique())`  
    `val_patients = set(val_df['patient_id'].unique())`  
    `overlap = train_patients.intersection(val_patients)`  
      
    `if overlap:`  
        `raise ValueError(f"CRITICAL LEAKAGE ERROR: Found {len(overlap)} patients in both train and val! IDs: {overlap}")`  
    `print("Leakage check passed: 0 overlapping patients.")`

`class MedicalDataset(Dataset):`  
    `def __init__(self, df, transform=None):`  
        `self.df = df`  
        `self.transform = transform`  
          
    `def __len__(self):`  
        `return len(self.df)`  
          
    `def __getitem__(self, idx):`  
        `row = self.df.iloc[idx]`  
        `img = Image.open(row['filepath']).convert('RGB')`  
        `if self.transform:`  
            `img = self.transform(img)`  
        `label = torch.tensor(row['label'], dtype=torch.long)`  
        `return img, label`

`def get_transforms():`  
    `normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`  
      
    `train_tf = transforms.Compose([`  
        `transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0)),`  
        `transforms.RandomHorizontalFlip(),`  
        `transforms.RandomRotation(15),`  
        `transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),`  
        `transforms.ToTensor(),`  
        `normalize`  
    `])`  
      
    `val_tf = transforms.Compose([`  
        `transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),`  
        `transforms.ToTensor(),`  
        `normalize`  
    `])`  
    `return train_tf, val_tf`

*`# ==========================================`*  
*`# 3. ARCHITECTURE & OPTIMIZATION`*  
*`# ==========================================`*  
`def build_model(model_name, num_classes):`  
    `kwargs = {'pretrained': True, 'num_classes': num_classes}`  
      
    `# 🚨 FIX 2: Tell timm to interpolate the positional embeddings to 384x384`  
    `if 'swin' in model_name:`  
        `kwargs['img_size'] = Config.IMG_SIZE`  
          
    `model = timm.create_model(model_name, **kwargs)`  
    `return model.to(Config.DEVICE)`

`def get_optimizer_and_scheduler(model, steps_per_epoch):`  
    `head_params, backbone_params = [], []`  
    `head_name = model.default_cfg.get('classifier', 'fc')`  
      
    `for name, param in model.named_parameters():`  
        `if not param.requires_grad: continue`  
        `if head_name in name or 'head' in name:`  
            `head_params.append(param)`  
        `else:`  
            `backbone_params.append(param)`  
              
    `optimizer = optim.AdamW([`  
        `{'params': backbone_params, 'lr': Config.LR_BACKBONE},`  
        `{'params': head_params, 'lr': Config.LR_HEAD}`  
    `], weight_decay=Config.WEIGHT_DECAY)`  
      
    `scheduler = optim.lr_scheduler.CosineAnnealingLR(`  
        `optimizer,`   
        `T_max=Config.EPOCHS * steps_per_epoch,`   
        `eta_min=1e-6`  
    `)`  
    `return optimizer, scheduler`

*`# ==========================================`*  
*`# 4. TRAINING ENGINE & INFERENCE LOADER`*  
*`# ==========================================`*  
`def load_ema_inference_model(model_name, fold_path):`  
    `base_model = build_model(model_name, Config.NUM_CLASSES)`  
    `ema_avg_fn = get_ema_multi_avg_fn(decay=Config.EMA_DECAY)`  
    `inference_model = AveragedModel(base_model, multi_avg_fn=ema_avg_fn)`  
    `inference_model.load_state_dict(torch.load(fold_path, map_location=Config.DEVICE))`  
    `inference_model.eval()`  
    `return inference_model`

`def train_epoch(model, ema_model, loader, criterion, optimizer, scheduler, scaler):`  
    `model.train()`  
    `running_loss = 0.0`  
      
    `for inputs, labels in loader:`  
        `inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)`  
        `optimizer.zero_grad()`  
          
        `with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):`  
            `outputs = model(inputs)`  
            `loss = criterion(outputs, labels)`  
              
        `scaler.scale(loss).backward()`  
        `scaler.step(optimizer)`  
        `scaler.update()`  
        `scheduler.step()`  
          
        `ema_model.update_parameters(model)`  
        `running_loss += loss.item() * inputs.size(0)`  
          
    `return running_loss / len(loader.dataset)`

`def validate_epoch(model, loader, criterion):`  
    `model.eval()`  
    `running_loss = 0.0`  
    `all_preds, all_labels, all_logits = [], [], []`  
      
    `with torch.no_grad():`  
        `for inputs, labels in loader:`  
            `inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)`  
            `outputs = model(inputs)`  
            `loss = criterion(outputs, labels)`  
              
            `running_loss += loss.item() * inputs.size(0)`  
            `all_logits.extend(outputs.cpu().numpy())`  
            `all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())`  
            `all_labels.extend(labels.cpu().numpy())`  
              
    `val_loss = running_loss / len(loader.dataset)`  
    `macro_f1 = f1_score(all_labels, all_preds, average='macro')`  
    `return val_loss, macro_f1, np.array(all_logits), np.array(all_labels)`

*`# ==========================================`*  
*`# 5. PATIENT-LEVEL EVALUATION`*  
*`# ==========================================`*  
`def calculate_ece(probs, labels, n_bins=10):`  
    `bin_boundaries = np.linspace(0, 1, n_bins + 1)`  
    `confidences = np.max(probs, axis=1)`  
    `predictions = np.argmax(probs, axis=1)`  
    `accuracies = predictions == labels`  
      
    `ece = 0.0`  
    `for i in range(n_bins):`  
        `in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])`  
        `prop_in_bin = in_bin.mean()`  
        `if prop_in_bin > 0:`  
            `ece += np.abs(confidences[in_bin].mean() - accuracies[in_bin].mean()) * prop_in_bin`  
    `return ece`

`def evaluate_patient_level(oof_df, title="Patient-Level Evaluation"):`  
    `inconsistent_patients = oof_df.groupby('patient_id')['true_label'].nunique()`  
    `if inconsistent_patients.max() > 1:`  
        `bad_pids = inconsistent_patients[inconsistent_patients > 1].index.tolist()`  
        `raise ValueError(f"CRITICAL DATA ERROR: Patients with conflicting ground-truth labels found: {bad_pids}")`

    `patient_df = oof_df.groupby('patient_id').agg({`  
        `'true_label': 'first',`  
        `'probs': lambda x: np.mean(np.stack(x), axis=0).tolist()`  
    `}).reset_index()`

    `y_true = patient_df['true_label'].values`  
    `probs = np.stack(patient_df['probs'].values)`  
    `preds = np.argmax(probs, axis=1)`

    `acc = accuracy_score(y_true, preds)`  
    `f1 = f1_score(y_true, preds, average='macro')`  
      
    `try:`  
        `auc = roc_auc_score(y_true, probs, multi_class='ovr')`  
    `except ValueError:`  
        `auc = float('nan')`   
          
    `ece = calculate_ece(probs, y_true)`

    `print(f"\n--- {title} ---")`  
    `print(f"Total Unique Patients: {len(patient_df)}")`  
    `print(f"Patient Accuracy : {acc:.4f}\nPatient Macro F1 : {f1:.4f}\nPatient ROC-AUC  : {auc:.4f}\nPatient ECE      : {ece:.4f}")`  
    `return patient_df`

*`# ==========================================`*  
*`# 6. PIPELINE EXECUTION`*  
*`# ==========================================`*  
`def run_kfold_pipeline(df):`  
    `sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=Config.SEED)`  
    `train_tf, val_tf = get_transforms()`  
      
    `oof_data = {'fold': [], 'model': [], 'filepath': [], 'patient_id': [], 'true_label': [], 'logits': [], 'probs': []}`  
      
    `for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df['label'], df['patient_id'])):`  
        `print(f"\n{'='*40}\nFOLD {fold+1}/5\n{'='*40}")`  
          
        `train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]`  
        `verify_no_patient_overlap(train_df, val_df)`  
          
        `cls_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])`  
        `cls_weights_tensor = torch.tensor(cls_weights, dtype=torch.float32).to(Config.DEVICE)`  
        `criterion = nn.CrossEntropyLoss(weight=cls_weights_tensor)`  
          
        `train_loader = DataLoader(MedicalDataset(train_df, train_tf), batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)`  
        `val_loader = DataLoader(MedicalDataset(val_df, val_tf), batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)`  
          
        `for model_name in [Config.MODEL_A, Config.MODEL_B]:`  
            `print(f"\n--> Training {model_name}")`  
            `model = build_model(model_name, Config.NUM_CLASSES)`  
              
            `ema_avg_fn = get_ema_multi_avg_fn(decay=Config.EMA_DECAY)`  
            `ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)`  
              
            `optimizer, scheduler = get_optimizer_and_scheduler(model, len(train_loader))`  
            `scaler = GradScaler()`  
              
            `best_f1 = 0.0`  
            `patience_counter = 0`  
            `best_logits = None`  
            `best_labels = None`    
              
            `for epoch in range(Config.EPOCHS):`  
                `tl = train_epoch(model, ema_model, train_loader, criterion, optimizer, scheduler, scaler)`  
                `vl, vf1, v_logits, v_labels = validate_epoch(ema_model, val_loader, criterion)`  
                  
                `if vf1 > best_f1:`  
                    `best_f1 = vf1`  
                    `best_logits = v_logits`  
                    `best_labels = v_labels`    
                    `patience_counter = 0`  
                      
                    `torch.save(ema_model.state_dict(), f"{Config.SAVE_DIR}/{model_name}_fold{fold}.pth")`  
                `else:`  
                    `patience_counter += 1`  
                      
                `if patience_counter >= Config.PATIENCE:`  
                    `print(f"Early stopping triggered at epoch {epoch+1}")`  
                    `break`  
                      
            `print(f"Best Image-Level Val Macro F1: {best_f1:.4f}")`  
              
            `probs = torch.softmax(torch.tensor(best_logits), dim=1).numpy()`  
            `for i, idx in enumerate(val_idx):`  
                `row = df.iloc[idx]`  
                `oof_data['fold'].append(fold)`  
                `oof_data['model'].append(model_name)`  
                `oof_data['filepath'].append(row['filepath'])`  
                `oof_data['patient_id'].append(row['patient_id'])`  
                `oof_data['true_label'].append(best_labels[i])`  
                `oof_data['logits'].append(best_logits[i])`  
                `oof_data['probs'].append(probs[i])`  
                  
    `return pd.DataFrame(oof_data)`

*`# ==========================================`*  
*`# 7. KAGGLE DATA PARSING & EXECUTION`*  
*`# ==========================================`*  
`def parse_msld_dataset(base_path):`  
    `all_files = glob.glob(os.path.join(base_path, "**/*.*"), recursive=True)`  
    `valid_exts = ('.jpg', '.jpeg', '.png')`  
    `image_paths = [f for f in all_files if f.lower().endswith(valid_exts)]`  
      
    `if not image_paths:`  
        `raise ValueError(f"No images found in {base_path}!")`

    `data = []`  
    `seen_images = set() # Track by Class + Filename to prevent duplicates`  
      
    `for img_path in image_paths:`  
        `filename = os.path.basename(img_path)`  
        `folder_name = os.path.basename(os.path.dirname(img_path)).upper()`  
          
        `# Skip folders that aren't in our class list`  
        `if folder_name not in Config.CLASS_MAP:`  
            `continue`  
              
        `# 🚨 SMART DEDUPLICATION: Create a unique signature (Class + Filename)`  
        `# This prevents loading fold1/HFMD/01.jpg and fold2/HFMD/01.jpg twice,`  
        `# but allows HFMD/01.jpg and COWPOX/01.jpg to both load safely.`  
        `img_signature = f"{folder_name}_{filename}"`  
        `if img_signature in seen_images:`  
            `continue`  
              
        `seen_images.add(img_signature)`  
        `label = Config.CLASS_MAP[folder_name]`  
          
        `# Extract Patient ID (e.g., HFMD_127_01.jpg -> Patient: HFMD_127)`  
        `parts = filename.replace('-', '_').split('_')`  
        `if len(parts) >= 2:`  
            `patient_id = f"{parts[0]}_{parts[1]}"`  
        `else:`  
            `patient_id = filename`   
              
        `data.append({`  
            `'filepath': img_path,`  
            `'label': label,`  
            `'patient_id': patient_id`  
        `})`  
          
    `df = pd.DataFrame(data)`  
      
    `if len(df) == 0:`  
        `raise ValueError("DataFrame is empty! The images were found but skipped by the class filter.")`  
          
    `print(f"Loaded {len(df)} unique images belonging to {df['patient_id'].nunique()} unique patients.")`  
    `return df`

`def main():`  
    `print("Initializing Patient-Level Medical Deep Learning Pipeline for Kaggle...")`  
      
    `# 🚨 THE FIX: Force Kaggle to create the output folder right now`  
    `os.makedirs(Config.SAVE_DIR, exist_ok=True)`   
      
    `# ⚠️ EXACT KAGGLE DATASET PATH`  
    `dataset_dir = "/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20/Original Images/Original Images"`   
      
    `df = parse_msld_dataset(dataset_dir)`  
      
    `oof_df = run_kfold_pipeline(df)`  
      
    `# ... rest of your main() remains exactly the same ...`  
      
    `swin_oof = oof_df[oof_df['model'] == Config.MODEL_A]`  
    `eff_oof = oof_df[oof_df['model'] == Config.MODEL_B]`  
      
    `# 1. Base Models Patient-Level Evaluation`  
    `evaluate_patient_level(swin_oof, "Swin-Tiny (Patient Aggregated)")`  
    `evaluate_patient_level(eff_oof, "EfficientNetV2-S (Patient Aggregated)")`  
      
    `# 2. Safe Soft Voting Ensemble`  
    `print("\n--- Constructing Ensemble ---")`  
    `ensemble_oof = pd.merge(`  
        `swin_oof,`   
        `eff_oof[['filepath', 'probs']],`   
        `on='filepath',`   
        `suffixes=('', '_eff')`  
    `)`  
      
    `# Calculate perfectly aligned average`  
    `avg_probs = (`  
        `np.stack(ensemble_oof['probs'].values) +`   
        `np.stack(ensemble_oof['probs_eff'].values)`  
    `) / 2.0`  
      
    `# 🚨 THE FIX: Convert the 2D numpy matrix back into a list of 1D arrays`  
    `# so Pandas stores one probability array per row properly!`  
    `ensemble_oof['probs'] = list(avg_probs)`  
      
    `# Cleanup`  
    `ensemble_oof.drop(columns=['probs_eff'], inplace=True)`  
      
    `patient_ensemble_df = evaluate_patient_level(ensemble_oof, "Soft Voting Ensemble (Patient Aggregated)")`  
      
    `# 3. Save Final Artifacts`  
    `patient_preds = np.argmax(np.stack(patient_ensemble_df['probs'].values), axis=1)`  
    `patient_true = patient_ensemble_df['true_label'].values`  
      
    `plt.figure(figsize=(8, 6))`  
    `sns.heatmap(confusion_matrix(patient_true, patient_preds), annot=True, fmt='d', cmap='Blues',`   
                `xticklabels=Config.CLASS_MAP.keys(), yticklabels=Config.CLASS_MAP.keys())`  
    `plt.ylabel('True Patient Label')`  
    `plt.xlabel('Predicted Patient Label')`  
    `plt.title("Patient-Level Ensemble Confusion Matrix")`  
    `plt.savefig(f"{Config.SAVE_DIR}/patient_confusion_matrix.png")`  
      
    `oof_df.to_csv(f"{Config.SAVE_DIR}/raw_oof_predictions.csv", index=False)`  
    `patient_ensemble_df.to_csv(f"{Config.SAVE_DIR}/patient_aggregated_predictions.csv", index=False)`  
      
    `print(f"\nPipeline complete. Assets saved to {Config.SAVE_DIR}")`  
    `# 3. Save Final Artifacts`  
    `patient_preds = np.argmax(np.stack(patient_ensemble_df['probs'].values), axis=1)`  
    `patient_true = patient_ensemble_df['true_label'].values`  
      
    `plt.figure(figsize=(8, 6))`  
    `sns.heatmap(confusion_matrix(patient_true, patient_preds), annot=True, fmt='d', cmap='Blues',`   
                `xticklabels=Config.CLASS_MAP.keys(), yticklabels=Config.CLASS_MAP.keys())`  
    `plt.ylabel('True Patient Label')`  
    `plt.xlabel('Predicted Patient Label')`  
    `plt.title("Patient-Level Ensemble Confusion Matrix")`  
    `plt.savefig(f"{Config.SAVE_DIR}/patient_confusion_matrix.png")`  
      
    `oof_df.to_csv(f"{Config.SAVE_DIR}/raw_oof_predictions.csv", index=False)`  
    `patient_ensemble_df.to_csv(f"{Config.SAVE_DIR}/patient_aggregated_predictions.csv", index=False)`  
      
    `print(f"\nPipeline complete. Assets saved to {Config.SAVE_DIR}")`

`if __name__ == "__main__":`  
    `main()`

**CODE 5 Output :**  
`Initializing Patient-Level Medical Deep Learning Pipeline for Kaggle...`  
`Loaded 755 unique images belonging to 524 unique patients.`

`========================================`  
`FOLD 1/5`  
`========================================`  
`Leakage check passed: 0 overlapping patients.`

`--> Training swin_tiny_patch4_window7_224`

`Early stopping triggered at epoch 15`  
`Best Image-Level Val Macro F1: 0.7952`

`--> Training tf_efficientnetv2_s`

`Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.`

`Early stopping triggered at epoch 34`  
`Best Image-Level Val Macro F1: 0.7405`

`========================================`  
`FOLD 2/5`  
`========================================`  
`Leakage check passed: 0 overlapping patients.`

`--> Training swin_tiny_patch4_window7_224`  
`Early stopping triggered at epoch 17`  
`Best Image-Level Val Macro F1: 0.7963`

`--> Training tf_efficientnetv2_s`  
`Early stopping triggered at epoch 25`  
`Best Image-Level Val Macro F1: 0.7355`

`========================================`  
`FOLD 3/5`  
`========================================`  
`Leakage check passed: 0 overlapping patients.`

`--> Training swin_tiny_patch4_window7_224`  
`Early stopping triggered at epoch 24`  
`Best Image-Level Val Macro F1: 0.7956`

`--> Training tf_efficientnetv2_s`  
`Early stopping triggered at epoch 35`  
`Best Image-Level Val Macro F1: 0.7577`

`========================================`  
`FOLD 4/5`  
`========================================`  
`Leakage check passed: 0 overlapping patients.`

`--> Training swin_tiny_patch4_window7_224`  
`Early stopping triggered at epoch 28`  
`Best Image-Level Val Macro F1: 0.8679`

`--> Training tf_efficientnetv2_s`  
`Early stopping triggered at epoch 24`  
`Best Image-Level Val Macro F1: 0.7058`

`========================================`  
`FOLD 5/5`  
`========================================`  
`Leakage check passed: 0 overlapping patients.`

`--> Training swin_tiny_patch4_window7_224`  
`Early stopping triggered at epoch 22`  
`Best Image-Level Val Macro F1: 0.8620`

`--> Training tf_efficientnetv2_s`  
`Early stopping triggered at epoch 21`  
`Best Image-Level Val Macro F1: 0.7441`

`--- Swin-Tiny (Patient Aggregated) ---`  
`Total Unique Patients: 524`  
`Patient Accuracy : 0.8626`  
`Patient Macro F1 : 0.8494`  
`Patient ROC-AUC  : 0.9766`  
`Patient ECE      : 0.0364`

`--- EfficientNetV2-S (Patient Aggregated) ---`  
`Total Unique Patients: 524`  
`Patient Accuracy : 0.7882`  
`Patient Macro F1 : 0.7719`  
`Patient ROC-AUC  : 0.9499`  
`Patient ECE      : 0.1170`

`--- Constructing Ensemble ---`

`--- Soft Voting Ensemble (Patient Aggregated) ---`  
`Total Unique Patients: 524`  
`Patient Accuracy : 0.8473`  
`Patient Macro F1 : 0.8333`  
`Patient ROC-AUC  : 0.9769`  
`Patient ECE      : 0.0335`

`Pipeline complete. Assets saved to /kaggle/working/pipeline_results`

`Pipeline complete. Assets saved to /kaggle/working/pipeline_results`

`![][image2]`  
`![][image3]`

**CODE 6 :**   
`import os`  
`import random`  
`import glob`  
`import warnings`  
`import numpy as np`  
`import pandas as pd`  
`import matplotlib.pyplot as plt`  
`import seaborn as sns`

`import torch`  
`import torch.nn as nn`  
`import torch.optim as optim`  
`from torch.utils.data import Dataset, DataLoader`  
`from torch.amp import autocast, GradScaler`  
`from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn`

`import torchvision.transforms as transforms`  
`from PIL import Image`

`import timm`  
`from sklearn.model_selection import StratifiedGroupKFold`  
`from sklearn.metrics import (`  
    `accuracy_score, f1_score, roc_auc_score, confusion_matrix`  
`)`  
`from sklearn.utils.class_weight import compute_class_weight`

`warnings.filterwarnings('ignore')`

*`# ==========================================`*  
*`# 1. CONFIGURATION & SEEDING`*  
*`# ==========================================`*  
`class Config:`  
    `SEED = 42`  
    `NUM_CLASSES = 6`  
    `BATCH_SIZE = 16`   
    `IMG_SIZE = 384`  
      
    `MODEL_A = 'swin_tiny_patch4_window7_224'`   
    `MODEL_B = 'tf_efficientnetv2_s'`  
      
    `EPOCHS = 40`  
    `LR_HEAD = 3e-4`  
    `LR_BACKBONE = 1e-4`  
    `WEIGHT_DECAY = 1e-4`  
    `PATIENCE = 7`  
    `EMA_DECAY = 0.995`  
      
    `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`  
    `SAVE_DIR = "/kaggle/working/pipeline_results"`  
      
    `# 🚨 FIX: Updated to match the exact folder names in the Kaggle dataset!`  
    `CLASS_MAP = {`  
        `'HEALTHY': 0,`   
        `'COWPOX': 1,`   
        `'MONKEYPOX': 2,`   
        `'MEASLES': 3,`   
        `'CHICKENPOX': 4,`   
        `'HFMD': 5`  
    `}`

`def seed_everything(seed):`  
    `random.seed(seed)`  
    `np.random.seed(seed)`  
    `torch.manual_seed(seed)`  
    `torch.cuda.manual_seed_all(seed)`  
    `torch.backends.cudnn.deterministic = True`  
    `torch.backends.cudnn.benchmark = False`

`seed_everything(Config.SEED)`

*`# ==========================================`*  
*`# 2. DATA HANDLING & INTEGRITY CHECKS`*  
*`# ==========================================`*  
`def verify_no_patient_overlap(train_df, val_df):`  
    `"""Fast-fail assertion to guarantee zero patient leakage between folds."""`  
    `train_patients = set(train_df['patient_id'].unique())`  
    `val_patients = set(val_df['patient_id'].unique())`  
    `overlap = train_patients.intersection(val_patients)`  
      
    `if overlap:`  
        `raise ValueError(f"CRITICAL LEAKAGE ERROR: Found {len(overlap)} patients in both train and val! IDs: {overlap}")`  
    `print("Leakage check passed: 0 overlapping patients.")`

`class MedicalDataset(Dataset):`  
    `def __init__(self, df, transform=None):`  
        `self.df = df`  
        `self.transform = transform`  
          
    `def __len__(self):`  
        `return len(self.df)`  
          
    `def __getitem__(self, idx):`  
        `row = self.df.iloc[idx]`  
        `img = Image.open(row['filepath']).convert('RGB')`  
        `if self.transform:`  
            `img = self.transform(img)`  
        `label = torch.tensor(row['label'], dtype=torch.long)`  
        `return img, label`

`def get_transforms():`  
    `normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`  
      
    `train_tf = transforms.Compose([`  
        `transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0)),`  
        `transforms.RandomHorizontalFlip(),`  
        `transforms.RandomRotation(15),`  
        `transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),`  
        `transforms.ToTensor(),`  
        `normalize`  
    `])`  
      
    `val_tf = transforms.Compose([`  
        `transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),`  
        `transforms.ToTensor(),`  
        `normalize`  
    `])`  
    `return train_tf, val_tf`

*`# ==========================================`*  
*`# 3. ARCHITECTURE & OPTIMIZATION`*  
*`# ==========================================`*  
`def build_model(model_name, num_classes):`  
    `kwargs = {'pretrained': True, 'num_classes': num_classes}`  
      
    `# 🚨 FIX 2: Tell timm to interpolate the positional embeddings to 384x384`  
    `if 'swin' in model_name:`  
        `kwargs['img_size'] = Config.IMG_SIZE`  
          
    `model = timm.create_model(model_name, **kwargs)`  
    `return model.to(Config.DEVICE)`

`def get_optimizer_and_scheduler(model, steps_per_epoch):`  
    `head_params, backbone_params = [], []`  
    `head_name = model.default_cfg.get('classifier', 'fc')`  
      
    `for name, param in model.named_parameters():`  
        `if not param.requires_grad: continue`  
        `if head_name in name or 'head' in name:`  
            `head_params.append(param)`  
        `else:`  
            `backbone_params.append(param)`  
              
    `optimizer = optim.AdamW([`  
        `{'params': backbone_params, 'lr': Config.LR_BACKBONE},`  
        `{'params': head_params, 'lr': Config.LR_HEAD}`  
    `], weight_decay=Config.WEIGHT_DECAY)`  
      
    `scheduler = optim.lr_scheduler.CosineAnnealingLR(`  
        `optimizer,`   
        `T_max=Config.EPOCHS * steps_per_epoch,`   
        `eta_min=1e-6`  
    `)`  
    `return optimizer, scheduler`

*`# ==========================================`*  
*`# 4. TRAINING ENGINE & INFERENCE LOADER`*  
*`# ==========================================`*  
`def load_ema_inference_model(model_name, fold_path):`  
    `base_model = build_model(model_name, Config.NUM_CLASSES)`  
    `ema_avg_fn = get_ema_multi_avg_fn(decay=Config.EMA_DECAY)`  
    `inference_model = AveragedModel(base_model, multi_avg_fn=ema_avg_fn)`  
    `inference_model.load_state_dict(torch.load(fold_path, map_location=Config.DEVICE))`  
    `inference_model.eval()`  
    `return inference_model`

`def train_epoch(model, ema_model, loader, criterion, optimizer, scheduler, scaler):`  
    `model.train()`  
    `running_loss = 0.0`  
      
    `for inputs, labels in loader:`  
        `inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)`  
        `optimizer.zero_grad()`  
          
        `with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):`  
            `outputs = model(inputs)`  
            `loss = criterion(outputs, labels)`  
              
        `scaler.scale(loss).backward()`  
        `scaler.step(optimizer)`  
        `scaler.update()`  
        `scheduler.step()`  
          
        `ema_model.update_parameters(model)`  
        `running_loss += loss.item() * inputs.size(0)`  
          
    `return running_loss / len(loader.dataset)`

`import torch.nn.functional as F`

`def validate_epoch(model, loader, criterion):`  
    `model.eval()`  
    `running_loss = 0.0`  
    `all_preds, all_labels, all_logits = [], [], []`  
      
    `with torch.no_grad():`  
        `for inputs, labels in loader:`  
            `inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)`  
              
            `# --- TTA Pass 1: Original Image ---`  
            `logits_orig = model(inputs)`  
            `probs_orig = F.softmax(logits_orig, dim=1)`  
              
            `# --- TTA Pass 2: Horizontal Flip ---`  
            `# dims=[3] corresponds to the Width dimension in [Batch, Channel, Height, Width]`  
            `inputs_hflip = torch.flip(inputs, dims=[3])`  
            `logits_hflip = model(inputs_hflip)`  
            `probs_hflip = F.softmax(logits_hflip, dim=1)`  
              
            `# --- TTA Pass 3: Vertical Flip ---`  
            `# dims=[2] corresponds to the Height dimension`  
            `inputs_vflip = torch.flip(inputs, dims=[2])`  
            `logits_vflip = model(inputs_vflip)`  
            `probs_vflip = F.softmax(logits_vflip, dim=1)`  
              
            `# --- The TTA Ensemble ---`  
            `# Average the probabilities across all 3 views`  
            `avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0`  
              
            `# Convert averaged probabilities back to "log-probabilities"`   
            `# so the rest of your pipeline (which expects logits) doesn't break.`  
            `avg_pseudo_logits = torch.log(avg_probs + 1e-8)`  
              
            `# For the epoch loss monitor, we just calculate against the original logits`  
            `loss = criterion(logits_orig, labels)`  
            `running_loss += loss.item() * inputs.size(0)`  
              
            `# Store the TTA-boosted predictions`  
            `all_logits.extend(avg_pseudo_logits.cpu().numpy())`  
            `all_preds.extend(torch.argmax(avg_pseudo_logits, 1).cpu().numpy())`  
            `all_labels.extend(labels.cpu().numpy())`  
              
    `val_loss = running_loss / len(loader.dataset)`  
    `macro_f1 = f1_score(all_labels, all_preds, average='macro')`  
      
    `return val_loss, macro_f1, np.array(all_logits), np.array(all_labels)`  
*`# ==========================================`*  
*`# 5. PATIENT-LEVEL EVALUATION`*  
*`# ==========================================`*  
`def calculate_ece(probs, labels, n_bins=10):`  
    `bin_boundaries = np.linspace(0, 1, n_bins + 1)`  
    `confidences = np.max(probs, axis=1)`  
    `predictions = np.argmax(probs, axis=1)`  
    `accuracies = predictions == labels`  
      
    `ece = 0.0`  
    `for i in range(n_bins):`  
        `in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])`  
        `prop_in_bin = in_bin.mean()`  
        `if prop_in_bin > 0:`  
            `ece += np.abs(confidences[in_bin].mean() - accuracies[in_bin].mean()) * prop_in_bin`  
    `return ece`

`def evaluate_patient_level(oof_df, title="Patient-Level Evaluation"):`  
    `inconsistent_patients = oof_df.groupby('patient_id')['true_label'].nunique()`  
    `if inconsistent_patients.max() > 1:`  
        `bad_pids = inconsistent_patients[inconsistent_patients > 1].index.tolist()`  
        `raise ValueError(f"CRITICAL DATA ERROR: Patients with conflicting ground-truth labels found: {bad_pids}")`

    `patient_df = oof_df.groupby('patient_id').agg({`  
        `'true_label': 'first',`  
        `'probs': lambda x: np.mean(np.stack(x), axis=0).tolist()`  
    `}).reset_index()`

    `y_true = patient_df['true_label'].values`  
    `probs = np.stack(patient_df['probs'].values)`  
    `preds = np.argmax(probs, axis=1)`

    `acc = accuracy_score(y_true, preds)`  
    `f1 = f1_score(y_true, preds, average='macro')`  
      
    `try:`  
        `auc = roc_auc_score(y_true, probs, multi_class='ovr')`  
    `except ValueError:`  
        `auc = float('nan')`   
          
    `ece = calculate_ece(probs, y_true)`

    `print(f"\n--- {title} ---")`  
    `print(f"Total Unique Patients: {len(patient_df)}")`  
    `print(f"Patient Accuracy : {acc:.4f}\nPatient Macro F1 : {f1:.4f}\nPatient ROC-AUC  : {auc:.4f}\nPatient ECE      : {ece:.4f}")`  
    `return patient_df`

*`# ==========================================`*  
*`# 6. PIPELINE EXECUTION`*  
*`# ==========================================`*  
`def run_kfold_pipeline(df):`  
    `sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=Config.SEED)`  
    `train_tf, val_tf = get_transforms()`  
      
    `oof_data = {'fold': [], 'model': [], 'filepath': [], 'patient_id': [], 'true_label': [], 'logits': [], 'probs': []}`  
      
    `for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df['label'], df['patient_id'])):`  
        `print(f"\n{'='*40}\nFOLD {fold+1}/5\n{'='*40}")`  
          
        `train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]`  
        `verify_no_patient_overlap(train_df, val_df)`  
          
        `cls_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])`  
        `cls_weights_tensor = torch.tensor(cls_weights, dtype=torch.float32).to(Config.DEVICE)`  
        `criterion = nn.CrossEntropyLoss(weight=cls_weights_tensor)`  
          
        `train_loader = DataLoader(MedicalDataset(train_df, train_tf), batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)`  
        `val_loader = DataLoader(MedicalDataset(val_df, val_tf), batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)`  
          
        `for model_name in [Config.MODEL_A, Config.MODEL_B]:`  
            `print(f"\n--> Training {model_name}")`  
            `model = build_model(model_name, Config.NUM_CLASSES)`  
              
            `ema_avg_fn = get_ema_multi_avg_fn(decay=Config.EMA_DECAY)`  
            `ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)`  
              
            `optimizer, scheduler = get_optimizer_and_scheduler(model, len(train_loader))`  
            `scaler = GradScaler()`  
              
            `best_f1 = 0.0`  
            `patience_counter = 0`  
            `best_logits = None`  
            `best_labels = None`    
              
            `for epoch in range(Config.EPOCHS):`  
                `tl = train_epoch(model, ema_model, train_loader, criterion, optimizer, scheduler, scaler)`  
                `vl, vf1, v_logits, v_labels = validate_epoch(ema_model, val_loader, criterion)`  
                  
                `if vf1 > best_f1:`  
                    `best_f1 = vf1`  
                    `best_logits = v_logits`  
                    `best_labels = v_labels`    
                    `patience_counter = 0`  
                      
                    `torch.save(ema_model.state_dict(), f"{Config.SAVE_DIR}/{model_name}_fold{fold}.pth")`  
                `else:`  
                    `patience_counter += 1`  
                      
                `if patience_counter >= Config.PATIENCE:`  
                    `print(f"Early stopping triggered at epoch {epoch+1}")`  
                    `break`  
                      
            `print(f"Best Image-Level Val Macro F1: {best_f1:.4f}")`  
              
            `probs = torch.softmax(torch.tensor(best_logits), dim=1).numpy()`  
            `for i, idx in enumerate(val_idx):`  
                `row = df.iloc[idx]`  
                `oof_data['fold'].append(fold)`  
                `oof_data['model'].append(model_name)`  
                `oof_data['filepath'].append(row['filepath'])`  
                `oof_data['patient_id'].append(row['patient_id'])`  
                `oof_data['true_label'].append(best_labels[i])`  
                `oof_data['logits'].append(best_logits[i])`  
                `oof_data['probs'].append(probs[i])`  
                  
    `return pd.DataFrame(oof_data)`

*`# ==========================================`*  
*`# 7. KAGGLE DATA PARSING & EXECUTION`*  
*`# ==========================================`*  
`def parse_msld_dataset(base_path):`  
    `all_files = glob.glob(os.path.join(base_path, "**/*.*"), recursive=True)`  
    `valid_exts = ('.jpg', '.jpeg', '.png')`  
    `image_paths = [f for f in all_files if f.lower().endswith(valid_exts)]`  
      
    `if not image_paths:`  
        `raise ValueError(f"No images found in {base_path}!")`

    `data = []`  
    `seen_images = set() # Track by Class + Filename to prevent duplicates`  
      
    `for img_path in image_paths:`  
        `filename = os.path.basename(img_path)`  
        `folder_name = os.path.basename(os.path.dirname(img_path)).upper()`  
          
        `# Skip folders that aren't in our class list`  
        `if folder_name not in Config.CLASS_MAP:`  
            `continue`  
              
        `# 🚨 SMART DEDUPLICATION: Create a unique signature (Class + Filename)`  
        `# This prevents loading fold1/HFMD/01.jpg and fold2/HFMD/01.jpg twice,`  
        `# but allows HFMD/01.jpg and COWPOX/01.jpg to both load safely.`  
        `img_signature = f"{folder_name}_{filename}"`  
        `if img_signature in seen_images:`  
            `continue`  
              
        `seen_images.add(img_signature)`  
        `label = Config.CLASS_MAP[folder_name]`  
          
        `# Extract Patient ID (e.g., HFMD_127_01.jpg -> Patient: HFMD_127)`  
        `parts = filename.replace('-', '_').split('_')`  
        `if len(parts) >= 2:`  
            `patient_id = f"{parts[0]}_{parts[1]}"`  
        `else:`  
            `patient_id = filename`   
              
        `data.append({`  
            `'filepath': img_path,`  
            `'label': label,`  
            `'patient_id': patient_id`  
        `})`  
          
    `df = pd.DataFrame(data)`  
      
    `if len(df) == 0:`  
        `raise ValueError("DataFrame is empty! The images were found but skipped by the class filter.")`  
          
    `print(f"Loaded {len(df)} unique images belonging to {df['patient_id'].nunique()} unique patients.")`  
    `return df`

`def main():`  
    `print("Initializing Patient-Level Medical Deep Learning Pipeline for Kaggle...")`  
      
    `# Force Kaggle to create the output folder`  
    `os.makedirs(Config.SAVE_DIR, exist_ok=True)`   
      
    `# EXACT KAGGLE DATASET PATH`  
    `dataset_dir = "/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20/Original Images/Original Images"`   
      
    `df = parse_msld_dataset(dataset_dir)`  
    `oof_df = run_kfold_pipeline(df)`  
      
    `swin_oof = oof_df[oof_df['model'] == Config.MODEL_A]`  
    `eff_oof = oof_df[oof_df['model'] == Config.MODEL_B]`  
      
    `# 1. Base Models Patient-Level Evaluation`  
    `evaluate_patient_level(swin_oof, "Swin-Tiny (Patient Aggregated)")`  
    `evaluate_patient_level(eff_oof, "EfficientNetV2-S (Patient Aggregated)")`  
      
    `# 2. Weighted Soft Voting Ensemble (75% Swin / 25% EffNet)`  
    `print("\n--- Constructing Weighted Ensemble (75% Swin / 25% EffNet) ---")`  
    `ensemble_oof = pd.merge(`  
        `swin_oof,`   
        `eff_oof[['filepath', 'probs']],`   
        `on='filepath',`   
        `suffixes=('', '_eff')`  
    `)`  
      
    `swin_weight = 0.75`  
    `eff_weight = 0.25`  
      
    `# Calculate the weighted average`  
    `avg_probs = (`  
        `(swin_weight * np.stack(ensemble_oof['probs'].values)) +`   
        `(eff_weight * np.stack(ensemble_oof['probs_eff'].values))`  
    `)`  
      
    `# Store the weighted results back into the dataframe`  
    `ensemble_oof['probs'] = list(avg_probs)`  
      
    `# Final Patient-Level Evaluation of the Weighted Ensemble`  
    `patient_ensemble_df = evaluate_patient_level(ensemble_oof, "Weighted Ensemble (Patient Aggregated)")`  
      
    `# 3. Save Final Artifacts & Confusion Matrix`  
    `patient_probs = np.stack(patient_ensemble_df['probs'].values)`  
    `patient_preds = np.argmax(patient_probs, axis=1)`  
    `patient_true = patient_ensemble_df['true_label'].values`  
      
    `plt.figure(figsize=(10, 8))`  
    `sns.heatmap(confusion_matrix(patient_true, patient_preds), annot=True, fmt='d', cmap='Blues',`   
                `xticklabels=Config.CLASS_MAP.keys(), yticklabels=Config.CLASS_MAP.keys())`  
    `plt.ylabel('True Patient Label')`  
    `plt.xlabel('Predicted Patient Label')`  
    `plt.title("Weighted Patient-Level Ensemble Confusion Matrix")`  
    `plt.savefig(f"{Config.SAVE_DIR}/patient_confusion_matrix.png")`  
      
    `# Save CSVs for further analysis (e.g., your Threshold Tuning)`  
    `oof_df.to_csv(f"{Config.SAVE_DIR}/raw_oof_predictions.csv", index=False)`  
    `patient_ensemble_df.to_csv(f"{Config.SAVE_DIR}/patient_aggregated_predictions.csv", index=False)`  
      
    `print(f"\nPipeline complete. Assets saved to {Config.SAVE_DIR}")`  
`if __name__ == "__main__":`  
    `main()`  
**CODE 6 Output:**  
`Initializing Patient-Level Medical Deep Learning Pipeline for Kaggle...`  
`Loaded 755 unique images belonging to 524 unique patients.`

`========================================`  
`FOLD 1/5`  
`========================================`  
`Leakage check passed: 0 overlapping patients.`

`--> Training swin_tiny_patch4_window7_224`

`Early stopping triggered at epoch 21`  
`Best Image-Level Val Macro F1: 0.7915`

`--> Training tf_efficientnetv2_s`

`Best Image-Level Val Macro F1: 0.7076`

`========================================`  
`FOLD 2/5`  
`========================================`  
`Leakage check passed: 0 overlapping patients.`

`--> Training swin_tiny_patch4_window7_224`  
`Early stopping triggered at epoch 20`  
`Best Image-Level Val Macro F1: 0.8143`

`--> Training tf_efficientnetv2_s`  
`Best Image-Level Val Macro F1: 0.7502`

`========================================`  
`FOLD 3/5`  
`========================================`  
`Leakage check passed: 0 overlapping patients.`

`--> Training swin_tiny_patch4_window7_224`  
`Early stopping triggered at epoch 20`  
`Best Image-Level Val Macro F1: 0.7745`

`--> Training tf_efficientnetv2_s`  
`Early stopping triggered at epoch 34`  
`Best Image-Level Val Macro F1: 0.7004`

`========================================`  
`FOLD 4/5`  
`========================================`  
`Leakage check passed: 0 overlapping patients.`

`--> Training swin_tiny_patch4_window7_224`  
`Early stopping triggered at epoch 21`  
`Best Image-Level Val Macro F1: 0.8763`

`--> Training tf_efficientnetv2_s`  
`Best Image-Level Val Macro F1: 0.7590`

`========================================`  
`FOLD 5/5`  
`========================================`  
`Leakage check passed: 0 overlapping patients.`

`--> Training swin_tiny_patch4_window7_224`

`Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.`

`Early stopping triggered at epoch 23`  
`Best Image-Level Val Macro F1: 0.8599`

`--> Training tf_efficientnetv2_s`  
`Early stopping triggered at epoch 35`  
`Best Image-Level Val Macro F1: 0.7937`

`--- Swin-Tiny (Patient Aggregated) ---`  
`Total Unique Patients: 524`  
`Patient Accuracy : 0.8683`  
`Patient Macro F1 : 0.8560`  
`Patient ROC-AUC  : 0.9760`  
`Patient ECE      : 0.0436`

`--- EfficientNetV2-S (Patient Aggregated) ---`  
`Total Unique Patients: 524`  
`Patient Accuracy : 0.7901`  
`Patient Macro F1 : 0.7709`  
`Patient ROC-AUC  : 0.9465`  
`Patient ECE      : 0.0809`

`--- Constructing Weighted Ensemble (75% Swin / 25% EffNet) ---`

`--- Weighted Ensemble (Patient Aggregated) ---`  
`Total Unique Patients: 524`  
`Patient Accuracy : 0.8607`  
`Patient Macro F1 : 0.8450`  
`Patient ROC-AUC  : 0.9763`  
`Patient ECE      : 0.0344`

`Pipeline complete. Assets saved to /kaggle/working/pipeline_results`

`![][image4]`  
**Partial Code 6 (Ran on Google Colab) :**  
import pandas as pd  
import numpy as np  
import ast  
from sklearn.metrics import f1\_score, accuracy\_score, classification\_report

print("Starting Patient-Level Probability Threshold Tuning...")

\# 1\. Load the Patient-Level Output from your Pipeline  
csv\_path \= "patient\_aggregated\_predictions.csv"  
patient\_df \= pd.read\_csv(csv\_path)

\# Pandas saves lists as strings in CSVs (e.g., "\[0.1, 0.8, ...\]"), so we must convert them back to arrays  
patient\_df\['probs'\] \= patient\_df\['probs'\].apply(ast.literal\_eval)

y\_true \= patient\_df\['true\_label'\].values  
probs \= np.stack(patient\_df\['probs'\].values)

\# Reconstruct the class names based on your exact mapping  
CLASS\_MAP \= {'HEALTHY': 0, 'COWPOX': 1, 'MONKEYPOX': 2, 'MEASLES': 3, 'CHICKENPOX': 4, 'HFMD': 5}  
target\_names \= \[k for k, v in sorted(CLASS\_MAP.items(), key=lambda item: item\[1\])\]

\# 2\. Baseline Metrics (Before Tuning)  
baseline\_preds \= np.argmax(probs, axis=1)  
best\_f1 \= f1\_score(y\_true, baseline\_preds, average='macro')  
best\_acc \= accuracy\_score(y\_true, baseline\_preds)  
best\_weights \= np.ones(6)  
best\_preds \= baseline\_preds.copy()

\# 3\. Random Search Optimization  
np.random.seed(42)  
iterations \= 20000  \# Pushed to 20k for a deeper search

for \_ in range(iterations):  
    \# Generate random class multipliers between 0.5 and 1.5  
    weights \= np.random.uniform(0.5, 1.5, size=6)

    \# Apply weights to the probabilities  
    adjusted\_probs \= probs \* weights

    \# Get new predictions  
    preds \= np.argmax(adjusted\_probs, axis=1)

    \# Calculate the new Macro F1 Score  
    macro\_f1 \= f1\_score(y\_true, preds, average='macro')

    \# Save if it's the new best  
    if macro\_f1 \> best\_f1:  
        best\_f1 \= macro\_f1  
        best\_weights \= weights  
        best\_preds \= preds

\# 4\. Print Results  
print("\\nOptimization Complete\!")  
print("-" \* 30)  
print(f"Original Patient Accuracy:  {best\_acc:.4f}")  
print(f"Optimized Patient Accuracy: {accuracy\_score(y\_true, best\_preds):.4f}")  
print(f"Original Patient Macro F1:  {f1\_score(y\_true, baseline\_preds, average='macro'):.4f}")  
print(f"Optimized Patient Macro F1: {best\_f1:.4f}")

print("\\n" \+ "="\*50)  
print("🏆 POST-TUNING PATIENT-LEVEL METRICS 🏆")  
print("="\*50)  
print(classification\_report(y\_true, best\_preds, target\_names=target\_names, digits=4))

print("\\nOptimal Class Multipliers (Threshold Adjustments):")  
for name, weight in zip(target\_names, best\_weights):  
    print(f"{name:\>10}: {weight:.4f}")

**Partial Code 6 Output (Ran on Google Colab) :**  
Starting Patient-Level Probability Threshold Tuning...

Optimization Complete\!  
\------------------------------  
Original Patient Accuracy:  0.8607  
Optimized Patient Accuracy: 0.8798  
Original Patient Macro F1:  0.8450  
Optimized Patient Macro F1: 0.8695

\==================================================  
🏆 POST-TUNING PATIENT-LEVEL METRICS 🏆  
\==================================================  
              precision    recall  f1-score   support

     HEALTHY     0.9528    0.9806    0.9665       103  
      COWPOX     0.8605    0.9024    0.8810        41  
   MONKEYPOX     0.8678    0.7895    0.8268       133  
     MEASLES     0.9048    0.8444    0.8736        45  
  CHICKENPOX     0.7419    0.7541    0.7480        61  
        HFMD     0.8933    0.9504    0.9210       141

    accuracy                         0.8798       524  
   macro avg     0.8702    0.8702    0.8695       524  
weighted avg     0.8793    0.8798    0.8787       524

Optimal Class Multipliers (Threshold Adjustments):  
   HEALTHY: 1.0340  
    COWPOX: 1.1342  
 MONKEYPOX: 0.9077  
   MEASLES: 1.4247  
CHICKENPOX: 0.8996  
      HFMD: 0.8074

**CODE 7 :**  
Cell 1 :   
import os  
import pandas as pd  
import glob  
from sklearn.model\_selection import GroupKFold

print("Searching for images...")  
all\_images \= glob.glob('/kaggle/input/\*\*/\*.jpg', recursive=True)

\# 1\. Filter and Deduplicate  
\# The dataset duplicates the 755 original images across 5 fold folders.   
\# We use a dictionary to keep only one unique path per image filename.  
unique\_images \= {}  
for path in all\_images:  
    if 'Original Images' in path:  
        filename \= os.path.basename(path) \# FIXED: os.path.basename  
        if filename not in unique\_images:  
            unique\_images\[filename\] \= path

image\_paths \= list(unique\_images.values())  
print(f"Found {len(image\_paths)} unique original images.")

if len(image\_paths) \== 0:  
    raise FileNotFoundError("Zero images found\! Check Kaggle dataset paths.")

\# 2\. Extract metadata from filenames  
data \= \[\]  
for path in image\_paths:  
    filename \= os.path.basename(path)  
    parts \= filename.replace('.jpg', '').split('\_')  
      
    \# Handle naming inconsistencies gracefully  
    if len(parts) \>= 3:  
        disease\_code \= parts\[0\]  
        patient\_id \= f"{disease\_code}\_{parts\[1\]}" \# e.g., 'MKP\_17'  
        data.append({'image\_path': path, 'label': disease\_code, 'patient\_id': patient\_id})

df \= pd.DataFrame(data)

if 'label' not in df.columns:  
    raise ValueError(f"Failed to extract labels. First filename: {os.path.basename(image\_paths\[0\])}")

\# 3\. Encode Labels (e.g., MKP \-\> 5, CHP \-\> 0\)  
label\_mapping \= {label: idx for idx, label in enumerate(df\['label'\].unique())}  
df\['target'\] \= df\['label'\].map(label\_mapping)

\# 4\. Strict Patient-Level Split  
gkf \= GroupKFold(n\_splits=5)  
train\_idx, val\_idx \= next(gkf.split(df, df\['target'\], groups=df\['patient\_id'\]))

train\_df \= df.iloc\[train\_idx\].reset\_index(drop=True)  
val\_df \= df.iloc\[val\_idx\].reset\_index(drop=True)

print("--- Split Successful \---")  
print(f"Train patients: {train\_df\['patient\_id'\].nunique()} \| Val patients: {val\_df\['patient\_id'\].nunique()}")  
print(f"Train images: {len(train\_df)} \| Val images: {len(val\_df)}")

**Cell 1 Output :**  
**Searching for images...**  
**Found 755 unique original images.**  
**\--- Split Successful \---**  
**Train patients: 420 \| Val patients: 104**  
**Train images: 604 \| Val images: 151**

**Cell 2 :**   
import torch  
import cv2  
import numpy as np  
from torch.utils.data import Dataset  
import albumentations as A  
from albumentations.pytorch import ToTensorV2

def apply\_gray\_world(image):  
    """Simple Color Constancy: normalizes illumination variations."""  
    image \= image.astype(np.float32)  
    b, g, r \= cv2.split(image)  
    m\_b, m\_g, m\_r \= np.mean(b), np.mean(g), np.mean(r)  
    m \= (m\_b \+ m\_g \+ m\_r) / 3.0  
      
    \# Safeguard against division by zero  
    b \= np.clip(b \* (m / (m\_b \+ 1e-5)), 0, 255\)  
    g \= np.clip(g \* (m / (m\_g \+ 1e-5)), 0, 255\)  
    r \= np.clip(r \* (m / (m\_r \+ 1e-5)), 0, 255\)  
    return cv2.merge((b, g, r)).astype(np.uint8)

class MpoxDataset(Dataset):  
    def \_\_init\_\_(self, df, transforms=None):  
        self.df \= df  
        self.transforms \= transforms

    def \_\_len\_\_(self):  
        return len(self.df)

    def \_\_getitem\_\_(self, idx):  
        img\_path \= self.df\['image\_path'\]\[idx\]  
        image \= cv2.imread(img\_path)  
        image \= cv2.cvtColor(image, cv2.COLOR\_BGR2RGB)  
          
        \# Apply domain-specific preprocessing to standardize skin tones  
        image \= apply\_gray\_world(image)  
          
        if self.transforms:  
            image \= self.transforms(image=image)\['image'\]  
              
        label \= torch.tensor(self.df\['target'\]\[idx\], dtype=torch.long)  
        return image, label

\# \---------------------------------------------------------  
\# Albumentations setup (High Res: 224x224)  
\# \---------------------------------------------------------

\# Training: Heavy augmentation to prevent overfitting  
train\_transforms \= A.Compose(\[  
    A.Resize(224, 224),  
    A.HorizontalFlip(p=0.5),  
    \# Replaced ShiftScaleRotate with the modern Affine equivalent  
    A.Affine(translate\_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),  
    A.Normalize(mean=\[0.485, 0.456, 0.406\], std=\[0.229, 0.224, 0.225\]),  
    ToTensorV2()  
\])

\# Validation: NO augmentation, just resize and normalize  
val\_transforms \= A.Compose(\[  
    A.Resize(224, 224),  
    A.Normalize(mean=\[0.485, 0.456, 0.406\], std=\[0.229, 0.224, 0.225\]),  
    ToTensorV2()  
\])  
**Cell 3 :**   
import torch  
import torch.nn as nn  
import timm

class MpoxHybridNet(nn.Module):  
    def \_\_init\_\_(self, num\_classes=6, pretrained=True):  
        super().\_\_init\_\_()  
          
        \# maxvit\_tiny\_tf\_224: A SOTA hybrid of CNNs and Vision Transformers  
        print("Downloading/Loading MaxViT backbone...")  
        self.backbone \= timm.create\_model('maxvit\_tiny\_tf\_224', pretrained=pretrained, num\_classes=0)  
        embed\_dim \= self.backbone.num\_features  
          
        \# Projection Head (Useful if we add Supervised Contrastive Learning later)  
        self.proj\_head \= nn.Sequential(  
            nn.Linear(embed\_dim, 128),  
            nn.ReLU(inplace=True),  
            nn.Linear(128, 64\)  
        )  
          
        \# Main Classification Head with Dropout for regularization  
        self.classifier \= nn.Sequential(  
            nn.Dropout(0.4),  
            nn.Linear(embed\_dim, num\_classes)  
        )

    def forward(self, x, return\_proj=False):  
        features \= self.backbone(x)  
        logits \= self.classifier(features)  
          
        if return\_proj:  
            \# L2 Normalize for Contrastive Space  
            projections \= torch.nn.functional.normalize(self.proj\_head(features), dim=1)   
            return logits, projections  
              
        return logits

\# \--- Quick Sanity Check \---  
print("Building temporary model for sanity check...")  
temp\_model \= MpoxHybridNet(num\_classes=6, pretrained=False)  
dummy\_input \= torch.randn(2, 3, 224, 224\) \# Simulate a batch of 2 images  
dummy\_output \= temp\_model(dummy\_input)

print("✅ Cell 3 ran perfectly\!")  
print(f"Output Tensor Shape: {dummy\_output.shape} (Expected: \[batch\_size, 6\])")  
**Cell 3 Output :**   
Building temporary model for sanity check...  
Downloading/Loading MaxViT backbone...  
✅ Cell 3 ran perfectly\!  
Output Tensor Shape: torch.Size(\[2, 6\]) (Expected: \[batch\_size, 6\])  
**Cell 4 :**   
import torch  
import torch.nn.functional as F  
import numpy as np

\# 1\. Focal Loss for Extreme Class Imbalance  
class SOTAFocalLoss(torch.nn.Module):  
    def \_\_init\_\_(self, alpha=0.25, gamma=2.0):  
        super().\_\_init\_\_()  
        self.alpha \= alpha  
        self.gamma \= gamma

    def forward(self, inputs, targets):  
        \# Calculate standard cross entropy  
        ce\_loss \= F.cross\_entropy(inputs, targets, reduction='none')  
        pt \= torch.exp(-ce\_loss)  
          
        \# Apply the Focal Loss formula: alpha \* (1-pt)^gamma \* CE  
        focal\_loss \= (self.alpha \* (1 \- pt) \*\* self.gamma \* ce\_loss).mean()  
        return focal\_loss

\# 2\. MixUp Augmentation Logic  
def mixup\_data(x, y, alpha=0.2):  
    """Applies MixUp to smooth decision boundaries."""  
    if alpha \> 0:  
        lam \= np.random.beta(alpha, alpha)  
    else:  
        lam \= 1.0

    batch\_size \= x.size()\[0\]  
    \# Ensure the random index permutations happen on the same device as the images (CPU or GPU)  
    index \= torch.randperm(batch\_size).to(x.device)

    \# Blend the images and keep track of both labels  
    mixed\_x \= lam \* x \+ (1 \- lam) \* x\[index, :\]  
    y\_a, y\_b \= y, y\[index\]  
      
    return mixed\_x, y\_a, y\_b, lam

def mixup\_criterion(criterion, pred, y\_a, y\_b, lam):  
    """Calculates the combined loss for MixUp data."""  
    return lam \* criterion(pred, y\_a) \+ (1 \- lam) \* criterion(pred, y\_b)

\# \--- Quick Sanity Check \---  
print("✅ Cell 4 ran perfectly\!")  
print("Focal Loss and MixUp functions are locked and loaded in memory.")  
**Cell 4 Output :**   
✅ Cell 4 ran perfectly\!  
Focal Loss and MixUp functions are locked and loaded in memory.  
**Cell 5 :**   
import torch  
import torch.optim as optim  
from torch.utils.data import DataLoader  
from tqdm import tqdm  
import copy

\# \---------------------------------------------------------  
\# 1\. Hyperparameters & Setup  
\# \---------------------------------------------------------  
BATCH\_SIZE \= 16  \# Safe size for Kaggle T4 GPUs with MaxViT  
EPOCHS \= 40      \# Start with 15 to see how quickly it converges  
LEARNING\_RATE \= 3e-4

\# Build Datasets and DataLoaders using the data from Cell 1 & 2  
train\_dataset \= MpoxDataset(train\_df, transforms=train\_transforms)  
val\_dataset \= MpoxDataset(val\_df, transforms=val\_transforms)

train\_loader \= DataLoader(train\_dataset, batch\_size=BATCH\_SIZE, shuffle=True, num\_workers=2, pin\_memory=True)  
val\_loader \= DataLoader(val\_dataset, batch\_size=BATCH\_SIZE, shuffle=False, num\_workers=2, pin\_memory=True)

\# Initialize Model, Loss, and Optimizer  
device \= torch.device("cuda" if torch.cuda.is\_available() else "cpu")  
print(f"Training on device: {device}")

model \= MpoxHybridNet(num\_classes=6, pretrained=True).to(device)  
criterion \= SOTAFocalLoss(alpha=0.25, gamma=2.0).to(device)  
optimizer \= optim.AdamW(model.parameters(), lr=LEARNING\_RATE, weight\_decay=1e-4)

\# Learning Rate Scheduler (FIXED: Removed 'verbose=True')  
scheduler \= optim.lr\_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

\# \---------------------------------------------------------  
\# 2\. The Training Loop  
\# \---------------------------------------------------------  
best\_val\_acc \= 0.0  
best\_model\_wts \= copy.deepcopy(model.state\_dict())

for epoch in range(EPOCHS):  
    print(f"\\nEpoch {epoch+1}/{EPOCHS}")  
    print("-" \* 20\)  
      
    \# \--- TRAIN PHASE \---  
    model.train()  
    running\_loss \= 0.0  
      
    train\_bar \= tqdm(train\_loader, desc="Training")  
    for inputs, labels in train\_bar:  
        inputs \= inputs.to(device)  
        labels \= labels.to(device)  
          
        optimizer.zero\_grad()  
          
        \# Apply MixUp on the fly  
        mixed\_inputs, labels\_a, labels\_b, lam \= mixup\_data(inputs, labels, alpha=0.2)  
          
        \# Forward pass  
        outputs \= model(mixed\_inputs)  
          
        \# Calculate MixUp Focal Loss  
        loss \= mixup\_criterion(criterion, outputs, labels\_a, labels\_b, lam)  
          
        \# Backward pass & Optimize  
        loss.backward()  
        optimizer.step()  
          
        running\_loss \+= loss.item() \* inputs.size(0)  
        train\_bar.set\_postfix(loss=loss.item())  
          
    epoch\_train\_loss \= running\_loss / len(train\_dataset)  
      
    \# \--- VALIDATION PHASE \---  
    model.eval()  
    val\_loss \= 0.0  
    val\_corrects \= 0  
      
    val\_bar \= tqdm(val\_loader, desc="Validation")  
    with torch.no\_grad():  
        for inputs, labels in val\_bar:  
            inputs \= inputs.to(device)  
            labels \= labels.to(device)  
              
            \# Forward pass (No MixUp during validation\!)  
            outputs \= model(inputs)  
            loss \= criterion(outputs, labels)  
              
            \_, preds \= torch.max(outputs, 1\)  
              
            val\_loss \+= loss.item() \* inputs.size(0)  
            val\_corrects \+= torch.sum(preds \== labels.data)  
              
    epoch\_val\_loss \= val\_loss / len(val\_dataset)  
    epoch\_val\_acc \= val\_corrects.double() / len(val\_dataset)  
      
    \# Get current learning rate for logging  
    current\_lr \= optimizer.param\_groups\[0\]\['lr'\]  
      
    print(f"Train Loss: {epoch\_train\_loss:.4f}")  
    print(f"Val Loss: {epoch\_val\_loss:.4f} \| Val Acc: {epoch\_val\_acc:.4f} \| LR: {current\_lr:.6f}")  
      
    \# Step the learning rate scheduler  
    scheduler.step(epoch\_val\_loss)  
      
    \# Save the best model weights  
    if epoch\_val\_acc \> best\_val\_acc:  
        best\_val\_acc \= epoch\_val\_acc  
        best\_model\_wts \= copy.deepcopy(model.state\_dict())  
        print("🌟 New best model saved\!")

\# \---------------------------------------------------------  
\# 3\. Final Wrap-up  
\# \---------------------------------------------------------  
print("\\nTraining Complete\!")  
print(f"Best Validation Accuracy: {best\_val\_acc:.4f}")

\# Load best weights back into the model  
model.load\_state\_dict(best\_model\_wts)  
\# Save the model to Kaggle's working directory  
torch.save(model.state\_dict(), 'Mpox\_Challenger\_SOTA.pth')  
print("Model saved to disk as 'Mpox\_Challenger\_SOTA.pth'")  
**Cell 5 Output :**   
Training on device: cuda  
Downloading/Loading MaxViT backbone...

Epoch 1/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:16\<00:00,  2.24it/s, loss=0.133\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.06it/s\]

Train Loss: 0.2134  
Val Loss: 0.1175 \| Val Acc: 0.6424 \| LR: 0.000300  
🌟 New best model saved\!

Epoch 2/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:16\<00:00,  2.28it/s, loss=0.146\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.74it/s\]

Train Loss: 0.1247  
Val Loss: 0.0954 \| Val Acc: 0.6821 \| LR: 0.000300  
🌟 New best model saved\!

Epoch 3/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.45it/s, loss=0.093\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.79it/s\]

Train Loss: 0.0761  
Val Loss: 0.0808 \| Val Acc: 0.7483 \| LR: 0.000300  
🌟 New best model saved\!

Epoch 4/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.47it/s, loss=0.0965\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.77it/s\]

Train Loss: 0.0611  
Val Loss: 0.0854 \| Val Acc: 0.7682 \| LR: 0.000300  
🌟 New best model saved\!

Epoch 5/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.0271\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.49it/s\]

Train Loss: 0.0748  
Val Loss: 0.0729 \| Val Acc: 0.7881 \| LR: 0.000300  
🌟 New best model saved\!

Epoch 6/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:16\<00:00,  2.34it/s, loss=0.00652\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.63it/s\]

Train Loss: 0.0512  
Val Loss: 0.0880 \| Val Acc: 0.7219 \| LR: 0.000300

Epoch 7/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.0166\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.67it/s\]

Train Loss: 0.0647  
Val Loss: 0.0851 \| Val Acc: 0.7219 \| LR: 0.000300

Epoch 8/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.42it/s, loss=0.103\]    
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.73it/s\]

Train Loss: 0.0609  
Val Loss: 0.0745 \| Val Acc: 0.7616 \| LR: 0.000300

Epoch 9/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.41it/s, loss=0.0677\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.64it/s\]

Train Loss: 0.0393  
Val Loss: 0.0921 \| Val Acc: 0.7748 \| LR: 0.000300

Epoch 10/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.0052\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.62it/s\]

Train Loss: 0.0419  
Val Loss: 0.0705 \| Val Acc: 0.8146 \| LR: 0.000150  
🌟 New best model saved\!

Epoch 11/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:16\<00:00,  2.37it/s, loss=0.00434\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.68it/s\]

Train Loss: 0.0397  
Val Loss: 0.0632 \| Val Acc: 0.8079 \| LR: 0.000150

Epoch 12/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.39it/s, loss=0.0108\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.50it/s\]

Train Loss: 0.0433  
Val Loss: 0.0620 \| Val Acc: 0.8212 \| LR: 0.000150  
🌟 New best model saved\!

Epoch 13/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.0849\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.58it/s\]

Train Loss: 0.0420  
Val Loss: 0.0737 \| Val Acc: 0.7815 \| LR: 0.000150

Epoch 14/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.40it/s, loss=0.00866\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.63it/s\]

Train Loss: 0.0519  
Val Loss: 0.0860 \| Val Acc: 0.7219 \| LR: 0.000150

Epoch 15/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.40it/s, loss=0.0428\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.71it/s\]

Train Loss: 0.0427  
Val Loss: 0.0781 \| Val Acc: 0.7483 \| LR: 0.000150

Epoch 16/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.40it/s, loss=0.0526\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.55it/s\]

Train Loss: 0.0337  
Val Loss: 0.0721 \| Val Acc: 0.7682 \| LR: 0.000150

Epoch 17/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.40it/s, loss=0.1\]      
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.63it/s\]

Train Loss: 0.0451  
Val Loss: 0.0736 \| Val Acc: 0.7815 \| LR: 0.000075

Epoch 18/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.39it/s, loss=0.121\]    
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.71it/s\]

Train Loss: 0.0400  
Val Loss: 0.0717 \| Val Acc: 0.7947 \| LR: 0.000075

Epoch 19/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.00461\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.69it/s\]

Train Loss: 0.0487  
Val Loss: 0.0640 \| Val Acc: 0.8212 \| LR: 0.000075

Epoch 20/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.0161\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.56it/s\]

Train Loss: 0.0363  
Val Loss: 0.0704 \| Val Acc: 0.7748 \| LR: 0.000075

Epoch 21/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.00254\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.50it/s\]

Train Loss: 0.0411  
Val Loss: 0.0663 \| Val Acc: 0.7881 \| LR: 0.000037

Epoch 22/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.00315\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.55it/s\]

Train Loss: 0.0368  
Val Loss: 0.0725 \| Val Acc: 0.7815 \| LR: 0.000037

Epoch 23/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.39it/s, loss=0.121\]    
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.61it/s\]

Train Loss: 0.0338  
Val Loss: 0.0676 \| Val Acc: 0.8079 \| LR: 0.000037

Epoch 24/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.00228\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.61it/s\]

Train Loss: 0.0328  
Val Loss: 0.0666 \| Val Acc: 0.7881 \| LR: 0.000037

Epoch 25/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.0522\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.57it/s\]  
Epoch 26/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.00325\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.69it/s\]

Train Loss: 0.0440  
Val Loss: 0.0633 \| Val Acc: 0.7947 \| LR: 0.000019

Epoch 27/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.00208\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.59it/s\]

Train Loss: 0.0402  
Val Loss: 0.0651 \| Val Acc: 0.8013 \| LR: 0.000019

Epoch 28/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.00209\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.57it/s\]

Train Loss: 0.0523  
Val Loss: 0.0616 \| Val Acc: 0.8079 \| LR: 0.000019

Epoch 29/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.39it/s, loss=0.0118\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.59it/s\]

Train Loss: 0.0278  
Val Loss: 0.0645 \| Val Acc: 0.7881 \| LR: 0.000019

Epoch 30/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:16\<00:00,  2.35it/s, loss=0.000906\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.65it/s\]

Train Loss: 0.0247  
Val Loss: 0.0667 \| Val Acc: 0.7881 \| LR: 0.000019

Epoch 31/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.00153\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.64it/s\]

Train Loss: 0.0390  
Val Loss: 0.0629 \| Val Acc: 0.8212 \| LR: 0.000019

Epoch 32/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.39it/s, loss=0.00325\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.61it/s\]

Train Loss: 0.0350  
Val Loss: 0.0595 \| Val Acc: 0.8079 \| LR: 0.000019

Epoch 33/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:16\<00:00,  2.37it/s, loss=0.0729\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.60it/s\]

Train Loss: 0.0336  
Val Loss: 0.0608 \| Val Acc: 0.7947 \| LR: 0.000019

Epoch 34/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.00947\]  
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.66it/s\]

Train Loss: 0.0412  
Val Loss: 0.0668 \| Val Acc: 0.7947 \| LR: 0.000019

Epoch 35/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.0448\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.64it/s\]

Train Loss: 0.0346  
Val Loss: 0.0625 \| Val Acc: 0.8079 \| LR: 0.000019

Epoch 36/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.0978\]    
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.55it/s\]

Train Loss: 0.0332  
Val Loss: 0.0611 \| Val Acc: 0.8278 \| LR: 0.000019  
🌟 New best model saved\!

Epoch 37/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.38it/s, loss=0.0113\]    
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.58it/s\]

Train Loss: 0.0404  
Val Loss: 0.0660 \| Val Acc: 0.7748 \| LR: 0.000009

Epoch 38/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:16\<00:00,  2.36it/s, loss=0.0182\]    
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.71it/s\]

Train Loss: 0.0347  
Val Loss: 0.0661 \| Val Acc: 0.7815 \| LR: 0.000009

Epoch 39/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:16\<00:00,  2.35it/s, loss=0.00176\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.70it/s\]

Train Loss: 0.0354  
Val Loss: 0.0634 \| Val Acc: 0.7947 \| LR: 0.000009

Epoch 40/40  
\--------------------

Training: 100%\|██████████\| 38/38 \[00:15\<00:00,  2.39it/s, loss=0.00879\]   
Validation: 100%\|██████████\| 10/10 \[00:01\<00:00,  6.53it/s\]

Train Loss: 0.0373  
Val Loss: 0.0614 \| Val Acc: 0.7881 \| LR: 0.000009

Training Complete\!  
Best Validation Accuracy: 0.8278  
Model saved to disk as 'Mpox\_Challenger\_SOTA.pth'

**Cell 6 :**   
import torch  
import numpy as np  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.metrics import classification\_report, confusion\_matrix, roc\_auc\_score  
import torch.nn.functional as F

print("Loading best model weights for final evaluation...")  
\# Ensure the model is in evaluation mode  
model.load\_state\_dict(best\_model\_wts)  
model.eval()

all\_preds \= \[\]  
all\_labels \= \[\]  
all\_probs \= \[\]

\# 1\. Run the Validation Set through the Best Model  
print("Evaluating on Validation Set...")  
with torch.no\_grad():  
    for inputs, labels in val\_loader:  
        inputs \= inputs.to(device)  
        labels \= labels.to(device)  
          
        logits \= model(inputs)  
        probs \= F.softmax(logits, dim=1)  
        \_, preds \= torch.max(logits, 1\)  
          
        all\_probs.append(probs.cpu().numpy())  
        all\_preds.append(preds.cpu().numpy())  
        all\_labels.append(labels.cpu().numpy())

\# Flatten the lists  
all\_probs \= np.concatenate(all\_probs)  
all\_preds \= np.concatenate(all\_preds)  
all\_labels \= np.concatenate(all\_labels)

\# Reverse the label mapping to get actual disease names  
\# (e.g., 5 \-\> 'MKP', 0 \-\> 'CHP')  
inv\_label\_mapping \= {v: k for k, v in label\_mapping.items()}  
target\_names \= \[inv\_label\_mapping\[i\] for i in range(len(inv\_label\_mapping))\]

\# \---------------------------------------------------------  
\# 2\. Print the SOTA Metrics Table (Precision, Recall, F1)  
\# \---------------------------------------------------------  
print("\\n" \+ "="\*50)  
print("🏆 FINAL CHALLENGER MODEL METRICS 🏆")  
print("="\*50)

\# Scikit-learn generates the exact Precision, Recall, and F1 table you saw in the PDF  
report \= classification\_report(all\_labels, all\_preds, target\_names=target\_names, digits=4)  
print(report)

\# Calculate One-vs-Rest AUC for all classes  
try:  
    auc\_score \= roc\_auc\_score(all\_labels, all\_probs, multi\_class='ovr', average='macro')  
    print(f"\\nOverall Macro AUC: {auc\_score:.4f}")  
except ValueError as e:  
    print("\\nAUC Calculation skipped (requires all classes to be present in validation set).")

\# \---------------------------------------------------------  
\# 3\. Plot the Confusion Matrix  
\# \---------------------------------------------------------  
cm \= confusion\_matrix(all\_labels, all\_preds)

plt.figure(figsize=(8, 6))  
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',   
            xticklabels=target\_names, yticklabels=target\_names)  
plt.title('Confusion Matrix \- Mpox Challenger Model')  
plt.ylabel('True Label')  
plt.xlabel('Predicted Label')  
plt.show()  
**Cell 6 Output :**   
**Loading best model weights for final evaluation...**  
**Evaluating on Validation Set...**

**\==================================================**  
**🏆 FINAL CHALLENGER MODEL METRICS 🏆**  
**\==================================================**  
              **precision    recall  f1-score   support**

        **HFMD     0.9394    0.9688    0.9538        32**  
         **MKP     0.8033    0.8033    0.8033        61**  
         **MSL     1.0000    1.0000    1.0000         9**  
     **HEALTHY     0.9545    0.9545    0.9545        22**  
         **CHP     0.4375    0.6364    0.5185        11**  
         **CWP     0.8000    0.5000    0.6154        16**

    **accuracy                         0.8278       151**  
   **macro avg     0.8225    0.8105    0.8076       151**  
**weighted avg     0.8389    0.8278    0.8283       151**

**Overall Macro AUC: 0.9643**

**![][image5]**  
**Cell 7 :**   
import numpy as np  
from sklearn.metrics import f1\_score, classification\_report, accuracy\_score

print("Starting Probability Weight Tuning...")

\# Baseline metrics from Cell 6  
baseline\_preds \= np.argmax(all\_probs, axis=1)  
best\_f1 \= f1\_score(all\_labels, baseline\_preds, average='macro')  
best\_acc \= accuracy\_score(all\_labels, baseline\_preds)  
best\_weights \= np.ones(6)  
best\_preds \= baseline\_preds.copy()

\# \---------------------------------------------------------  
\# Random Search for Optimal Decision Boundaries  
\# \---------------------------------------------------------  
np.random.seed(42)  
iterations \= 10000

for \_ in range(iterations):  
    \# Generate random class multipliers between 0.5 and 1.5  
    weights \= np.random.uniform(0.5, 1.5, size=6)  
      
    \# Apply weights to the raw probabilities  
    adjusted\_probs \= all\_probs \* weights  
      
    \# Get new predictions based on the weighted probabilities  
    preds \= np.argmax(adjusted\_probs, axis=1)  
      
    \# Calculate the new Macro F1 Score  
    macro\_f1 \= f1\_score(all\_labels, preds, average='macro')  
      
    \# If we found a better combination of thresholds, save it\!  
    if macro\_f1 \> best\_f1:  
        best\_f1 \= macro\_f1  
        best\_weights \= weights  
        best\_preds \= preds

\# \---------------------------------------------------------  
\# Print the Optimized Results  
\# \---------------------------------------------------------  
print("\\nOptimization Complete\!")  
print("-" \* 30\)  
print(f"Original Accuracy:  {best\_acc:.4f}")  
print(f"Optimized Accuracy: {accuracy\_score(all\_labels, best\_preds):.4f}")  
print(f"Original Macro F1:  {f1\_score(all\_labels, baseline\_preds, average='macro'):.4f}")  
print(f"Optimized Macro F1: {best\_f1:.4f}")

print("\\n" \+ "="\*50)  
print("🏆 POST-TUNING CALIBRATED METRICS 🏆")  
print("="\*50)  
print(classification\_report(all\_labels, best\_preds, target\_names=target\_names, digits=4))

\# Show the exact multipliers the model needed to fix its hesitation  
print("\\nOptimal Class Multipliers (Threshold Adjustments):")  
for name, weight in zip(target\_names, best\_weights):  
    print(f"{name:\>10}: {weight:.4f}")  
**Cell 7 Output :**   
Starting Probability Weight Tuning...

Optimization Complete\!  
\------------------------------  
Original Accuracy:  0.8278  
Optimized Accuracy: 0.8344  
Original Macro F1:  0.8076  
Optimized Macro F1: 0.8342

\==================================================  
🏆 POST-TUNING CALIBRATED METRICS 🏆  
\==================================================  
              precision    recall  f1-score   support

        HFMD     1.0000    0.9062    0.9508        32  
         MKP     0.8136    0.7869    0.8000        61  
         MSL     1.0000    1.0000    1.0000         9  
     HEALTHY     1.0000    0.9091    0.9524        22  
         CHP     0.4444    0.7273    0.5517        11  
         CWP     0.7500    0.7500    0.7500        16

    accuracy                         0.8344       151  
   macro avg     0.8347    0.8466    0.8342       151  
weighted avg     0.8577    0.8344    0.8427       151

Optimal Class Multipliers (Threshold Adjustments):  
      HFMD: 0.5175  
       MKP: 0.7132  
       MSL: 0.9250  
   HEALTHY: 0.5449  
       CHP: 0.8277  
       CWP: 1.3870

**CODE 8 :**   
import os  
import time  
import json  
import random  
import glob  
import zipfile  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from PIL import Image

\# Google Colab specific  
from google.colab import drive

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  
import torchvision.transforms as transforms  
import torchvision.models as models  
from torchvision.models import ResNet50\_Weights  
from torchvision.ops import StochasticDepth  
from torch.amp import autocast, GradScaler  
from torch.optim.swa\_utils import AveragedModel, get\_ema\_multi\_avg\_fn

from sklearn.model\_selection import train\_test\_split  
from sklearn.metrics import (  
    accuracy\_score, precision\_score, recall\_score,  
    f1\_score, confusion\_matrix, roc\_auc\_score  
)

\# \==========================================  
\# 0\. DRIVE MOUNT & DATASET EXTRACTION  
\# \==========================================  
if not os.path.exists('/content/drive'):  
    drive.mount('/content/drive')

ZIP\_PATH \= '/content/drive/MyDrive/Colab Data/MSLD v2.0.zip'  
EXTRACT\_TO \= '/content/dataset'

if os.path.exists(ZIP\_PATH):  
    if not os.path.exists(os.path.join(EXTRACT\_TO, 'Original Images')):  
        print("Unzipping dataset to local SSD for maximum I/O speed...")  
        with zipfile.ZipFile(ZIP\_PATH, 'r') as zip\_ref:  
            zip\_ref.extractall(EXTRACT\_TO)  
        print("Extraction complete\!")  
    else:  
        print("Dataset already unzipped.")  
else:  
    print(f"⚠️ ERROR: Could not find zip file at {ZIP\_PATH}. Check Drive path.")

\# \==========================================  
\# 1\. CONFIGURATION & SEEDING  
\# \==========================================  
class Config:  
    SEED \= 42  
    NUM\_CLASSES \= 2  \# Binary: Mpox (1) vs Others (0)  
    BATCH\_SIZE \= 8   \# Prevents CUDA OOM on T4 GPU  
    IMG\_SIZE \= 384

    \# Phase 1: Warmup  
    PHASE1\_EPOCHS \= 5  
    LR\_PHASE1 \= 1e-3  
    WD\_PHASE1 \= 1e-4

    \# Phase 2: Full Fine-Tuning  
    PHASE2\_EPOCHS \= 55  
    WD\_PHASE2 \= 5e-4  
    PATIENCE \= 10

    DEVICE \= torch.device("cuda" if torch.cuda.is\_available() else "cpu")  
    SAVE\_DIR \= "/content/drive/MyDrive/Colab Data/msld\_results"

os.makedirs(Config.SAVE\_DIR, exist\_ok=True)

def seed\_everything(seed\=42):  
    random.seed(seed)  
    os.environ\['PYTHONHASHSEED'\] \= str(seed)  
    np.random.seed(seed)  
    torch.manual\_seed(seed)  
    torch.cuda.manual\_seed(seed)  
    torch.cuda.manual\_seed\_all(seed)  
    torch.backends.cudnn.deterministic \= True  
    torch.backends.cudnn.benchmark \= False

seed\_everything(Config.SEED)

\# \==========================================  
\# 2\. DATA HANDLING & AUGMENTATION  
\# \==========================================  
class MedicalDataset(Dataset):  
    def \_\_init\_\_(self, file\_paths, labels, transform\=None):  
        self.file\_paths \= file\_paths  
        self.labels \= labels  
        self.transform \= transform

    def \_\_len\_\_(self):  
        return len(self.file\_paths)

    def \_\_getitem\_\_(self, idx):  
        img \= Image.open(self.file\_paths\[idx\]).convert('RGB')  
        if self.transform:  
            img \= self.transform(img)  
        label \= torch.tensor(self.labels\[idx\], dtype=torch.long)  
        return img, label

def get\_transforms():  
    normalize \= transforms.Normalize(mean=\[0.485, 0.456, 0.406\], std=\[0.229, 0.224, 0.225\])

    train\_transform \= transforms.Compose(\[  
        transforms.RandomResizedCrop(Config.IMG\_SIZE, scale=(0.7, 1.0)),  
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(15),  
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),  
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),  
        transforms.ToTensor(),  
        normalize,  
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)  
    \])

    val\_test\_transform \= transforms.Compose(\[  
        transforms.Resize((400, 400)),  
        transforms.CenterCrop(Config.IMG\_SIZE),  
        transforms.ToTensor(),  
        normalize  
    \])

    return train\_transform, val\_test\_transform

from sklearn.model\_selection import GroupShuffleSplit

\# ... (Keep your get\_transforms and other functions above this) ...

def prepare\_dataloaders():  
    base\_path \= "/content/dataset/Original Images"

    \# Grab literally every file in that directory and its subdirectories  
    all\_files \= glob.glob(os.path.join(base\_path, "\*\*/\*.\*"), recursive=True)

    \# Filter for just the image files  
    valid\_exts \= ('.jpg', '.jpeg', '.png')  
    image\_paths \= \[f for f in all\_files if f.lower().endswith(valid\_exts)\]

    if not image\_paths:  
        raise ValueError(f"Zero images found inside {base\_path}.")

    paths, labels, patient\_ids \= \[\], \[\], \[\]  
    found\_folders \= set()

    for img\_path in image\_paths:  
        parent\_folder \= os.path.basename(os.path.dirname(img\_path))  
        filename \= os.path.basename(img\_path)  
        found\_folders.add(parent\_folder)

        \# 1\. Label Assignment  
        if any(keyword in parent\_folder.lower() for keyword in \['mkp', 'mpox', 'monkeypox'\]):  
            labels.append(1)  
        else:  
            labels.append(0)

        \# 2\. Extract Patient ID  
        \# Filenames follow: CLASS\_PATIENTID\_IMAGEID.jpg (e.g., MKP\_73\_03.jpg or HEALTHY\_19\_01.jpg)  
        parts \= filename.split('\_')  
        if len(parts) \>= 2:  
            \# Combine Class and Patient ID to ensure uniqueness (e.g., 'MKP\_73')  
            patient\_id \= f"{parts\[0\]}\_{parts\[1\]}"  
        else:  
            \# Fallback just in case a file is named differently  
            patient\_id \= filename

        paths.append(img\_path)  
        patient\_ids.append(patient\_id)

    print(f"Detected subfolders: {found\_folders}")  
    print(f"Total Images: {len(paths)} \| Mpox: {sum(labels)} \| Others: {len(labels)-sum(labels)}")  
    print(f"Unique Patients: {len(set(patient\_ids))}")

    \# Ensure both classes are present  
    if sum(labels) \== 0 or sum(labels) \== len(labels):  
        raise ValueError("Could not find both classes\!")

    \# 3\. Patient-Independent Splitting  
    \# Split 1: Train (70%) vs Temp (Val \+ Test \= 30%)  
    gss1 \= GroupShuffleSplit(n\_splits=1, train\_size=0.70, random\_state=Config.SEED)  
    train\_idx, temp\_idx \= next(gss1.split(paths, labels, groups=patient\_ids))

    X\_train \= \[paths\[i\] for i in train\_idx\]  
    y\_train \= \[labels\[i\] for i in train\_idx\]

    X\_temp \= \[paths\[i\] for i in temp\_idx\]  
    y\_temp \= \[labels\[i\] for i in temp\_idx\]  
    temp\_groups \= \[patient\_ids\[i\] for i in temp\_idx\]

    \# Split 2: Validation (15%) vs Test (15%) \-\> Split the Temp group 50/50  
    gss2 \= GroupShuffleSplit(n\_splits=1, train\_size=0.50, random\_state=Config.SEED)  
    val\_idx, test\_idx \= next(gss2.split(X\_temp, y\_temp, groups=temp\_groups))

    X\_val \= \[X\_temp\[i\] for i in val\_idx\]  
    y\_val \= \[y\_temp\[i\] for i in val\_idx\]

    X\_test \= \[X\_temp\[i\] for i in test\_idx\]  
    y\_test \= \[y\_temp\[i\] for i in test\_idx\]

    print(f"Split sizes \-\> Train: {len(X\_train)} \| Val: {len(X\_val)} \| Test: {len(X\_test)}")

    train\_tf, val\_tf \= get\_transforms()

    train\_loader \= DataLoader(MedicalDataset(X\_train, y\_train, train\_tf), batch\_size=Config.BATCH\_SIZE, shuffle=True, num\_workers=2)  
    val\_loader \= DataLoader(MedicalDataset(X\_val, y\_val, val\_tf), batch\_size=Config.BATCH\_SIZE, shuffle=False, num\_workers=2)  
    test\_loader \= DataLoader(MedicalDataset(X\_test, y\_test, val\_tf), batch\_size=Config.BATCH\_SIZE, shuffle=False, num\_workers=2)

    return train\_loader, val\_loader, test\_loader, MedicalDataset(X\_test, y\_test, transform=None)

\# \==========================================  
\# 3\. ADVANCED ARCHITECTURE COMPONENTS  
\# \==========================================  
class FocalLoss(nn.Module):  
    \# gamma reduced from 2 to 1.5 to prevent gradient spikes with OneCycleLR  
    def \_\_init\_\_(self, alpha\=1, gamma\=1.5, reduction\='mean'):  
        super().\_\_init\_\_()  
        self.alpha \= alpha  
        self.gamma \= gamma  
        self.reduction \= reduction

    def forward(self, inputs, targets):  
        ce\_loss \= F.cross\_entropy(inputs, targets, reduction='none')  
        pt \= torch.exp(-ce\_loss)  
        focal\_loss \= self.alpha \* (1 \- pt) \*\* self.gamma \* ce\_loss  
        if self.reduction \== 'mean': return focal\_loss.mean()  
        return focal\_loss.sum() if self.reduction \== 'sum' else focal\_loss

\# Safe DropPath wrapper protecting against torchvision internal changes  
class SafeDropPathBottleneck(nn.Module):  
    def \_\_init\_\_(self, original\_bottleneck, drop\_prob):  
        super().\_\_init\_\_()  
        self.block \= original\_bottleneck  
        self.stochastic\_depth \= StochasticDepth(p=drop\_prob, mode="row")

    def forward(self, x):  
        identity \= self.block.downsample(x) if self.block.downsample is not None else x

        out \= self.block.conv1(x); out \= self.block.bn1(out); out \= self.block.relu(out)  
        out \= self.block.conv2(out); out \= self.block.bn2(out); out \= self.block.relu(out)  
        out \= self.block.conv3(out); out \= self.block.bn3(out)

        \# Apply DropPath only to the residual branch  
        out \= self.stochastic\_depth(out)  
        out \+= identity  
        out \= self.block.relu(out)  
        return out

def inject\_droppath(model, drop\_prob\=0.1):  
    for name, module in list(model.layer4.named\_children()):  
        if isinstance(module, models.resnet.Bottleneck):  
            setattr(model.layer4, name, SafeDropPathBottleneck(module, drop\_prob))  
    return model

def freeze\_batchnorm\_stats(model):  
    for module in model.modules():  
        if isinstance(module, nn.BatchNorm2d):  
            module.eval() \# Freezes stats but allows affine scaling (ideal for high-res)

def build\_advanced\_model():  
    model \= models.resnet50(weights=ResNet50\_Weights.DEFAULT)  
    num\_ftrs \= model.fc.in\_features  
    model.fc \= nn.Sequential(  
        nn.Linear(num\_ftrs, num\_ftrs // 2),  
        nn.GELU(),  
        nn.Dropout(0.5),  
        nn.Linear(num\_ftrs // 2, Config.NUM\_CLASSES)  
    )  
    model \= inject\_droppath(model, drop\_prob=0.1)  
    return model.to(Config.DEVICE)

\# \==========================================  
\# 4\. TRAINING ENGINE  
\# \==========================================  
class EarlyStopping:  
    def \_\_init\_\_(self, patience\=10, path\='best\_model.pth'):  
        self.patience \= patience  
        self.counter \= 0  
        self.best\_score \= None  
        self.early\_stop \= False  
        self.val\_max\_acc \= \-np.inf  
        self.path \= path

    def \_\_call\_\_(self, val\_acc, model):  
        if self.best\_score is None or val\_acc \> self.best\_score:  
            self.best\_score \= val\_acc  
            self.val\_max\_acc \= val\_acc  
            \# Clean save of the unwrapped model state dict  
            torch.save(model.state\_dict(), self.path)  
            self.counter \= 0  
        else:  
            self.counter \+= 1  
            if self.counter \>= self.patience: self.early\_stop \= True

def train\_phase1\_epoch(model, loader, criterion, optimizer, scaler):  
    model.train()  
    running\_loss, correct, total \= 0.0, 0, 0  
    for inputs, labels in loader:  
        inputs, labels \= inputs.to(Config.DEVICE), labels.to(Config.DEVICE)  
        optimizer.zero\_grad()  
        with autocast(device\_type="cuda"):  
            outputs \= model(inputs)  
            loss \= criterion(outputs, labels)

        scaler.scale(loss).backward()  
        scaler.step(optimizer)  
        scaler.update()

        running\_loss \+= loss.item() \* inputs.size(0)  
        \_, preds \= torch.max(outputs, 1)  
        correct \+= (preds \== labels).sum().item(); total \+= labels.size(0)  
    return running\_loss / total, correct / total

def train\_phase2\_epoch(model, ema\_model, loader, criterion, optimizer, scheduler, scaler):  
    model.train()  
    freeze\_batchnorm\_stats(model)

    running\_loss, correct, total \= 0.0, 0, 0  
    for inputs, labels in loader:  
        inputs, labels \= inputs.to(Config.DEVICE), labels.to(Config.DEVICE)

        optimizer.zero\_grad()  
        with autocast(device\_type="cuda"):  
            outputs \= model(inputs)  
            loss \= criterion(outputs, labels)

        scaler.scale(loss).backward()  
        scaler.unscale\_(optimizer)  
        torch.nn.utils.clip\_grad\_norm\_(model.parameters(), max\_norm=1.0)  
        scaler.step(optimizer)  
        scaler.update()

        ema\_model.update\_parameters(model)  
        scheduler.step()

        running\_loss \+= loss.item() \* inputs.size(0)  
        \_, preds \= torch.max(outputs, 1)  
        correct \+= (preds \== labels).sum().item(); total \+= labels.size(0)  
    return running\_loss / total, correct / total

def validate(model, loader, criterion):  
    model.eval()  
    running\_loss, correct, total \= 0.0, 0, 0  
    all\_preds, all\_labels, all\_probs \= \[\], \[\], \[\]

    with torch.no\_grad():  
        for inputs, labels in loader:  
            inputs, labels \= inputs.to(Config.DEVICE), labels.to(Config.DEVICE)  
            outputs \= model(inputs)  
            loss \= criterion(outputs, labels)

            running\_loss \+= loss.item() \* inputs.size(0)  
            probs \= torch.softmax(outputs, dim=1)  
            \_, preds \= torch.max(outputs, 1)

            correct \+= (preds \== labels).sum().item(); total \+= labels.size(0)  
            all\_preds.extend(preds.cpu().numpy())  
            all\_labels.extend(labels.cpu().numpy())

            if Config.NUM\_CLASSES \== 2: all\_probs.extend(probs\[:, 1\].cpu().numpy())  
            else: all\_probs.extend(probs.cpu().numpy())

    return running\_loss / total, correct / total, all\_labels, all\_preds, all\_probs

\# \==========================================  
\# 5\. TTA & CALIBRATION  
\# \==========================================  
class ModelWithTemperature(nn.Module):  
    def \_\_init\_\_(self, model):  
        super().\_\_init\_\_()  
        self.model \= model  
        self.temperature \= nn.Parameter(torch.ones(1) \* 1.5)

    def forward(self, input):  
        return self.model(input) / self.temperature

    def set\_temperature(self, valid\_loader):  
        self.to(Config.DEVICE)  
        nll\_criterion \= nn.CrossEntropyLoss().to(Config.DEVICE)  
        optimizer \= optim.LBFGS(\[self.temperature\], lr=0.01, max\_iter=50)

        logits\_list, labels\_list \= \[\], \[\]  
        with torch.no\_grad():  
            for input, label in valid\_loader:  
                logits\_list.append(self.model(input.to(Config.DEVICE)))  
                labels\_list.append(label.to(Config.DEVICE))

        logits \= torch.cat(logits\_list)  
        labels \= torch.cat(labels\_list)

        def eval():  
            optimizer.zero\_grad()  
            loss \= nll\_criterion(logits / self.temperature, labels)  
            loss.backward()  
            return loss  
        optimizer.step(eval)  
        print(f"Optimal Temperature: {self.temperature.item():.3f}")

def run\_tta(model, raw\_test\_dataset):  
    model.eval()  
    normalize \= transforms.Normalize(mean=\[0.485, 0.456, 0.406\], std=\[0.229, 0.224, 0.225\])

    t\_orig \= transforms.Compose(\[transforms.Resize((400, 400)), transforms.CenterCrop(Config.IMG\_SIZE), transforms.ToTensor(), normalize\])  
    t\_hflip \= transforms.Compose(\[transforms.Resize((400, 400)), transforms.CenterCrop(Config.IMG\_SIZE), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), normalize\])  
    t\_5crop \= transforms.Compose(\[transforms.Resize((400, 400)), transforms.FiveCrop(Config.IMG\_SIZE), transforms.Lambda(lambda crops: torch.stack(\[normalize(transforms.ToTensor()(crop)) for crop in crops\]))\])

    all\_labels, final\_probs \= \[\], \[\]

    with torch.no\_grad():  
        for i in range(len(raw\_test\_dataset)):  
            img, label \= raw\_test\_dataset\[i\]  
            all\_labels.append(label.item())

            p\_orig \= torch.softmax(model(t\_orig(img).unsqueeze(0).to(Config.DEVICE)), dim=1).cpu().numpy()\[0\]  
            p\_hflip \= torch.softmax(model(t\_hflip(img).unsqueeze(0).to(Config.DEVICE)), dim=1).cpu().numpy()\[0\]  
            crops \= t\_5crop(img).to(Config.DEVICE)  
            p\_crops \= torch.softmax(model(crops), dim=1).mean(dim=0).cpu().numpy()

            p\_final \= (p\_orig \+ p\_hflip \+ p\_crops \* 5) / 7.0  
            final\_probs.append(p\_final)

    preds \= np.argmax(np.array(final\_probs), axis=1)  
    return accuracy\_score(all\_labels, preds)

def calculate\_ece(probs, labels, n\_bins\=10):  
    bin\_boundaries \= np.linspace(0, 1, n\_bins \+ 1)  
    confidences \= np.max(probs, axis=1)  
    predictions \= np.argmax(probs, axis=1)  
    accuracies \= predictions \== labels

    ece \= 0.0  
    for i in range(n\_bins):  
        in\_bin \= (confidences \> bin\_boundaries\[i\]) & (confidences \<= bin\_boundaries\[i+1\])  
        prop\_in\_bin \= in\_bin.mean()  
        if prop\_in\_bin \> 0:  
            ece \+= np.abs(confidences\[in\_bin\].mean() \- accuracies\[in\_bin\].mean()) \* prop\_in\_bin  
    return ece

\# \==========================================  
\# 6\. MAIN EXECUTION  
\# \==========================================  
def main():  
    start\_time \= time.time()  
    train\_loader, val\_loader, test\_loader, raw\_test\_dataset \= prepare\_dataloaders()

    model \= build\_advanced\_model()  
    criterion \= FocalLoss(gamma=1.5)  
    scaler \= GradScaler()  
    history \= {'train\_loss': \[\], 'val\_loss': \[\], 'train\_acc': \[\], 'val\_acc': \[\], 'fc\_lr': \[\]}

    \# \------------------------------------------  
    \# PHASE 1: WARMUP (Head Only)  
    \# \------------------------------------------  
    print("\\n--- Phase 1: Warmup (FC Head) \---")  
    for param in model.parameters(): param.requires\_grad \= False  
    for param in model.fc.parameters(): param.requires\_grad \= True

    optimizer1 \= optim.AdamW(model.fc.parameters(), lr=Config.LR\_PHASE1, weight\_decay=Config.WD\_PHASE1)  
    for epoch in range(Config.PHASE1\_EPOCHS):  
        tl, ta \= train\_phase1\_epoch(model, train\_loader, criterion, optimizer1, scaler)  
        vl, va, \_, \_, \_ \= validate(model, val\_loader, criterion)  
        print(f"Epoch {epoch+1}/{Config.PHASE1\_EPOCHS} \| Train Acc: {ta:.4f} \| Val Acc: {va:.4f}")

    \# \------------------------------------------  
    \# PHASE 2: FULL FINE-TUNING  
    \# \------------------------------------------  
    print("\\n--- Phase 2: Full Fine-Tuning \---")  
    for param in model.parameters(): param.requires\_grad \= True

    param\_groups \= \[  
        {'params': model.conv1.parameters(), 'lr': 1e-5},  
        {'params': model.bn1.parameters(),  'lr': 1e-5},  
        {'params': model.layer1.parameters(), 'lr': 1e-5},  
        {'params': model.layer2.parameters(), 'lr': 3e-5},  
        {'params': model.layer3.parameters(), 'lr': 1e-4},  
        {'params': model.layer4.parameters(), 'lr': 3e-4},  
        {'params': model.fc.parameters(),     'lr': 1e-3}  
    \]

    optimizer2 \= optim.AdamW(param\_groups, weight\_decay=Config.WD\_PHASE2)  
    scheduler \= optim.lr\_scheduler.OneCycleLR(optimizer2, max\_lr=\[g\['lr'\] for g in param\_groups\],  
                                              steps\_per\_epoch=len(train\_loader), epochs=Config.PHASE2\_EPOCHS, pct\_start=0.1)

    \# Accelerated decay for smaller medical dataset  
    ema\_avg\_fn \= get\_ema\_multi\_avg\_fn(decay=0.995)  
    ema\_model \= AveragedModel(model, multi\_avg\_fn=ema\_avg\_fn)  
    early\_stopping \= EarlyStopping(patience=Config.PATIENCE, path=f"{Config.SAVE\_DIR}/best\_ema\_resnet50.pth")

    best\_epoch \= 0  
    for epoch in range(Config.PHASE2\_EPOCHS):  
        tl, ta \= train\_phase2\_epoch(model, ema\_model, train\_loader, criterion, optimizer2, scheduler, scaler)  
        vl, va, \_, \_, \_ \= validate(ema\_model, val\_loader, criterion)

        lr\_backbone \= optimizer2.param\_groups\[0\]\['lr'\]  
        lr\_fc \= optimizer2.param\_groups\[\-1\]\['lr'\]

        history\['train\_loss'\].append(tl); history\['val\_loss'\].append(vl)  
        history\['train\_acc'\].append(ta); history\['val\_acc'\].append(va); history\['fc\_lr'\].append(lr\_fc)

        print(f"Epoch {epoch+1:02d}/{Config.PHASE2\_EPOCHS} \| Train Acc: {ta:.4f} \| Val Acc (EMA): {va:.4f} \| Backbone LR: {lr\_backbone:.6f} \| FC LR: {lr\_fc:.6f}")

        \# Save the UNDERLYING module, not the EMA wrapper  
        early\_stopping(va, ema\_model.module)

        if va \== early\_stopping.val\_max\_acc: best\_epoch \= epoch \+ 1  
        if early\_stopping.early\_stop:  
            print("Early stopping triggered."); break

    \# \------------------------------------------  
    \# EVALUATION  
    \# \------------------------------------------

    print("\\n--- Running Deep Evaluation \---")

    \# 1\. Build fresh model and load clean state dict (version-safe)  
    best\_model \= build\_advanced\_model()  
    best\_model.load\_state\_dict(torch.load(f"{Config.SAVE\_DIR}/best\_ema\_resnet50.pth", map\_location=Config.DEVICE))  
    best\_model.eval()

    \# 2\. Evaluate  
    \_, test\_acc\_raw, y\_true, y\_pred, y\_probs \= validate(best\_model, test\_loader, criterion)  
    test\_acc\_tta \= run\_tta(best\_model, raw\_test\_dataset)

    calibrated\_model \= ModelWithTemperature(best\_model)  
    calibrated\_model.set\_temperature(val\_loader)  
    \_, test\_acc\_calib, \_, \_, \_ \= validate(calibrated\_model, test\_loader, criterion)

    raw\_probs\_matrix \= np.column\_stack(\[1 \- np.array(y\_probs), np.array(y\_probs)\]) if Config.NUM\_CLASSES \== 2 else np.array(y\_probs)  
    ece\_before \= calculate\_ece(raw\_probs\_matrix, y\_true)

    \# Save & Plot  
    pd.DataFrame(history).to\_csv(f"{Config.SAVE\_DIR}/training\_logs.csv", index=False)  
    plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1); plt.plot(history\['train\_loss'\], label='Train Loss'); plt.plot(history\['val\_loss'\], label='Val Loss')  
    plt.legend(); plt.subplot(1, 2, 2); plt.plot(history\['train\_acc'\], label='Train Acc'); plt.plot(history\['val\_acc'\], label='Val Acc')  
    plt.legend(); plt.savefig(f"{Config.SAVE\_DIR}/learning\_curves.png")

    plt.figure(figsize=(6, 5)); sns.heatmap(confusion\_matrix(y\_true, y\_pred), annot=True, fmt='d', cmap='Blues')  
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.savefig(f"{Config.SAVE\_DIR}/confusion\_matrix.png")

    with open(f"{Config.SAVE\_DIR}/test\_metrics.json", "w") as f:  
        json.dump({"Accuracy": test\_acc\_raw, "ROC\_AUC": roc\_auc\_score(y\_true, y\_probs) if Config.NUM\_CLASSES \== 2 else "N/A"}, f, indent=4)

    print("\\n" \+ "="\*40 \+ "\\n      🎯 RESEARCH PIPELINE SUMMARY\\n" \+ "="\*40)  
    print(f"Total Time           : {(time.time() \- start\_time) / 60:.2f} mins")  
    print(f"Peak GPU Memory      : {torch.cuda.max\_memory\_allocated() / (1024\*\*3) if torch.cuda.is\_available() else 0:.2f} GB")  
    print(f"Best Epoch           : {best\_epoch}")  
    print("-" \* 40)  
    print(f"Best Val Acc (EMA)   : {early\_stopping.val\_max\_acc:.4f}")  
    print(f"Overfitting Gap      : {max(history\['train\_acc'\]) \- early\_stopping.val\_max\_acc:.4f} (Train \- Val)")  
    print("-" \* 40)  
    print(f"Test Accuracy (Raw)  : {test\_acc\_raw:.4f}")  
    print(f"Test Accuracy (TTA)  : {test\_acc\_tta:.4f}")  
    print(f"Test Acc (Calibrated): {test\_acc\_calib:.4f}")  
    print(f"ECE (Before Calib)   : {ece\_before:.4f}")  
    print("-" \* 40)  
    print(f"Target (\>0.90)       : {'YES ✅' if test\_acc\_tta \> 0.90 else 'NO ❌'}")  
    print("="\*40)

if \_\_name\_\_ \== "\_\_main\_\_":  
    main()  
**CODE 8 Output :**   
Dataset already unzipped.  
Detected subfolders: {'Healthy', 'Cowpox', 'Monkeypox', 'Measles', 'Chickenpox', 'HFMD'}  
Total Images: 3775 \| Mpox: 1420 \| Others: 2355  
Unique Patients: 524  
Split sizes \-\> Train: 2545 \| Val: 600 \| Test: 630

\--- Phase 1: Warmup (FC Head) \---  
Epoch 1/5 \| Train Acc: 0.7297 \| Val Acc: 0.7000  
Epoch 2/5 \| Train Acc: 0.7560 \| Val Acc: 0.6667  
Epoch 3/5 \| Train Acc: 0.7929 \| Val Acc: 0.7083  
Epoch 4/5 \| Train Acc: 0.8004 \| Val Acc: 0.6750  
Epoch 5/5 \| Train Acc: 0.8094 \| Val Acc: 0.6917

\--- Phase 2: Full Fine-Tuning \---  
Epoch 01/55 \| Train Acc: 0.8821 \| Val Acc (EMA): 0.7000 \| Backbone LR: 0.000001 \| FC LR: 0.000116  
Epoch 02/55 \| Train Acc: 0.8951 \| Val Acc (EMA): 0.7333 \| Backbone LR: 0.000003 \| FC LR: 0.000321  
Epoch 03/55 \| Train Acc: 0.8731 \| Val Acc (EMA): 0.7500 \| Backbone LR: 0.000006 \| FC LR: 0.000589  
Epoch 04/55 \| Train Acc: 0.8868 \| Val Acc (EMA): 0.7333 \| Backbone LR: 0.000008 \| FC LR: 0.000835  
Epoch 05/55 \| Train Acc: 0.9061 \| Val Acc (EMA): 0.7667 \| Backbone LR: 0.000010 \| FC LR: 0.000981  
Epoch 06/55 \| Train Acc: 0.9411 \| Val Acc (EMA): 0.7583 \| Backbone LR: 0.000010 \| FC LR: 0.001000  
Epoch 07/55 \| Train Acc: 0.9646 \| Val Acc (EMA): 0.7500 \| Backbone LR: 0.000010 \| FC LR: 0.000998  
Epoch 08/55 \| Train Acc: 0.9623 \| Val Acc (EMA): 0.7583 \| Backbone LR: 0.000010 \| FC LR: 0.000994  
Epoch 09/55 \| Train Acc: 0.9650 \| Val Acc (EMA): 0.7583 \| Backbone LR: 0.000010 \| FC LR: 0.000988  
Epoch 10/55 \| Train Acc: 0.9690 \| Val Acc (EMA): 0.7583 \| Backbone LR: 0.000010 \| FC LR: 0.000980  
Epoch 11/55 \| Train Acc: 0.9780 \| Val Acc (EMA): 0.7583 \| Backbone LR: 0.000010 \| FC LR: 0.000970  
Epoch 12/55 \| Train Acc: 0.9847 \| Val Acc (EMA): 0.7250 \| Backbone LR: 0.000010 \| FC LR: 0.000958  
Epoch 13/55 \| Train Acc: 0.9862 \| Val Acc (EMA): 0.7500 \| Backbone LR: 0.000009 \| FC LR: 0.000944  
Epoch 14/55 \| Train Acc: 0.9882 \| Val Acc (EMA): 0.7583 \| Backbone LR: 0.000009 \| FC LR: 0.000929  
Epoch 15/55 \| Train Acc: 0.9807 \| Val Acc (EMA): 0.7583 \| Backbone LR: 0.000009 \| FC LR: 0.000912  
Early stopping triggered.

\--- Running Deep Evaluation \---  
Optimal Temperature: 1.770

\========================================  
      🎯 RESEARCH PIPELINE SUMMARY  
\========================================  
Total Time           : 23.26 mins  
Peak GPU Memory      : 1.69 GB  
Best Epoch           : 5  
\----------------------------------------  
Best Val Acc (EMA)   : 0.7667  
Overfitting Gap      : 0.2215 (Train \- Val)  
\----------------------------------------  
Test Accuracy (Raw)  : 0.8095  
Test Accuracy (TTA)  : 0.8095  
Test Acc (Calibrated): 0.8095  
ECE (Before Calib)   : 0.0836  
\----------------------------------------  
Target (\>0.90)       : NO ❌  
\========================================

![][image6]  
![][image7]

[image1]: <data:image/png;base64,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Q80Ym69/2Fqzb4piiRa4cJYN/xutvN8q7AKHX2+vKnC9j3Vnjxjp+p88oJKLBR4IU7Tqi7WudevWfmx7HnmUGzFydDbWuK/svvwYT58xKxvjubfOq9k3NVg0UmAFbv3mbe75ld+55z75zm3dsdOva7e+9oW/3wqcbfv3R99max5+jdTg+pEiW5MPO7xbZddkLBspsgJ3yKGHuvPGjnfnj7swe+A391931eyboliiBS6caUshWLjyBa5nz57u8MMPd3feeWe2zcpa27Zt3dChQ+tsswJnb4sSfknuv/9+f58VuL/97W/u4IMPdtu2bfPb7O++Zs0a//FLL73UqMKKEztl1029vpKLBR4IU9S9Rw8/rmcMG55te/jRJ/y2V994s2Z/2T1YisMYV6EoY9FIleWtL38o3G7B7anB9SNlVV2TsWxUhY3zaUOH1WxPUSyVKXDbt2/PhAL397//3c2cOdPNmDHD7b///tn+t99+u/v3v//t/7/WTZs2+W2hwH322WfumGOOccuWLXPffvvrI8lOnTq5c8891910003utttu89vse9jXt4wdO9ZfB9hQcGKnrKqLBR4IU2S/Ozauky6bnG1btvw9v23eHXfX7C+7bsOmrTVlLYyxChzH+Qve9+vag+98VXNfCG5PDa4fKavqmoxlowo++uw/fpz/Om1mzX0piqXBAmcFKea3Su/evd3y5cszhxxyiHvuuef8P0AodY8//nj25sJWxsIvwpw5c/y2+p5CnTRpUvZxjx49/J8rV670n2+xt1LZuHFjtk8sOLFTVtXFAg+EKQrjet2Uadm2Dz5e5bfNmj2nZn/ZdavXrKspa2GMVeD23PI1P/g1bfO2HTX3mRDcnhpcP1JW1TUZy0bqBp92uh/jlq1a1dyXqlgaLHAxjXlKkRUsXHZ2zF4Ra/8I+WJnBe6TTz7JXiFrT4fa2TlLfQUufw1c165ds4/vuece9/nnn7vnn38+21ZfcGKnrKqLBR4IU9SjZ08/rvmnUBc88pjfpqdQebCshTFWgdt9Vzz9sft+86+Xkfznhy019wchuD01uH6krKprMpaNVL245E136KHt/fh2O6J7zf0pi6XBAocvXEjlRQzr16/3Z8Yw7733nn961GL/5VdRgTv++OOz/S2xAmdn3U466SS3c+evF/M2FJzYKavqYoEHwhRdefW1flxb/fII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>

[image2]: <data:image/png;base64,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zY/qTn5Ub/KjelMY1Z34wVWozzzzjBk9erS5/PLL7VWos2bN4s3Kij5hUTKaUFwYTSjOj+pOflRv8qN6UxjVncrQoEEDs+GGG9reuAEDBvDqsqMAJ4QQQgiRMhTghBBCCCFShgKcEEIIIUTKUIATQgghhEgZCnBrARdccIE544wzeLG9SmnBggWmX79+9nd25syZ3va/+tWvzP777+8tA9j2jTfeyP4d9e3VYUHU/mA/sD/PPfec2XTTTc2ECROy6/BFibiq56GHHgo9onzgve6+++52AurOO+9sTj31VPPhhx/adZ999pk55ZRT7OvjRsUHHHCAueeee7KPPeKII8wVV1yR/fuRRx6x7/Ppp5/OLgPHHHOM/cnlvd1225lJkyZ52wavGX69n376ya5DWTzwwAPe9pgwu8EGG3jL8oH6gNcN7zO46qqr7HKsB99995256KKLzA477GA/e0zQvfbaa73HHH/88fYxvXv3zi677777zK677pr9e8stt8z+DnCVFsoY352EMuL6gfe8atUqc+KJJ5rf/va33mMffvhh+3xTp071yhFl2Lp165xyjPrcUI5BfeJyRBm+//773rIoiinDYBuWuf322816663Hi+1+4msINt54Y7P11lvbidFPPPFEdn3wOUXBrwmDrzR4/fXXbZ3dYostzGabbWb2228/enQ0OEauvvpqe5zwMYLPG597GLQDBx10UPbv8O8AV0p27NjRfjbbb7+9/bxfe+01+9mDv/zlL972999/vz0+QVDv2AD8juf99ttvs8vQ5oTLLPh8ULf32GMPc+utt5off/zRrkPZ33vvvbbO4HlQV373u99lH/vvf//b1kMcIwEom7322sssXbo0u+zngn3jNjKo7/na7E6dOnnbYZ/DDBo0yCun8HOsu+66tj7gy2dnzJjhPU4kl9zWRPziiGoMADcG48aNs410IL7LJsy5555rdtlll5yra/DYcIBbvHhx9jnQ0N92223e84Ko/cF+YH9Aq1atTJMmTbL78Pvf/9428EHjXk6++eYbe9+6V1991ZbByJEj7Yl+n332setxUsfNib/++mu7LU6iOKEG+9KhQ4fstuDss8+25cQn2Jtvvtn+DJc3Gsv+/fvbk2gQ0HCCDV4z/Hpnnnlm9jURcMePH29/X7ZsmX39Nm3aZF6oBrBf2L9wsPrhhx9sg4+Q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>

[image3]: <data:image/png;base64,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zY/qTn5Ub/KjelMY1Z34wVWozzzzjBk9erS5/PLL7VWos2bN4s3Kij5hUTKaUFwYTSjOj+pOflRv8qN6UxjVncrQoEEDs+GGG9reuAEDBvDqsqMAJ4QQQgiRMhTghBBCCCFShgKcEEIIIUTKUIATQgghhEgZCnBrARdccIE544wzeLG9SmnBggWmX79+9nd25syZ3va/+tWvzP777+8tA9j2jTfeyP4d9e3VYUHU/mA/sD/PPfec2XTTTc2ECROy6/BFibiq56GHHgo9onzgve6+++52AurOO+9sTj31VPPhhx/adZ999pk55ZRT7OvjRsUHHHCAueeee7KPPeKII8wVV1yR/fuRRx6x7/Ppp5/OLgPHHHOM/cnlvd1225lJkyZ52wavGX69n376ya5DWTzwwAPe9pgwu8EGG3jL8oH6gNcN7zO46qqr7HKsB99995256KKLzA477GA/e0zQvfbaa73HHH/88fYxvXv3zi677777zK677pr9e8stt8z+DnCVFsoY352EMuL6gfe8atUqc+KJJ5rf/va33mMffvhh+3xTp071yhFl2Lp165xyjPrcUI5BfeJyRBm+//773rIoiinDYBuWuf322816663Hi+1+4msINt54Y7P11lvbidFPPPFEdn3wOUXBrwmDrzR4/fXXbZ3dYostzGabbWb2228/enQ0OEauvvpqe5zwMYLPG597GLQDBx10UPbv8O8AV0p27NjRfjbbb7+9/bxfe+01+9mDv/zlL972999/vz0+QVDv2AD8juf99ttvs8vQ5oTLLPh8ULf32GMPc+utt5off/zRrkPZ33vvvbbO4HlQV373u99lH/vvf//b1kMcIwEom7322sssXbo0u+zngn3jNjKo7/na7E6dOnnbYZ/DDBo0yCun8HOsu+66tj7gy2dnzJjhPU4kl9zWRPziiGoMADcG48aNs410IL7LJsy5555rdtlll5yra/DYcIBbvHhx9jnQ0N92223e84Ko/cF+YH9Aq1atTJMmTbL78Pvf/9428EHjXk6++eYbe9+6V1991ZbByJEj7Yl+n332setxUsfNib/++mu7LU6iOKEG+9KhQ4fstuDss8+25cQn2Jtvvtn+DJc3Gsv+/fvbk2gQ0HCCDV4z/Hpnnnlm9jURcMePH29/X7ZsmX39Nm3aZF6oBrBf2L9wsPrhhx9sg4+Q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>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnAAAAIrCAYAAABvWqSXAACAAElEQVR4XuzdCdyVc/4/fmW7pVCWMCPMGC1CZEZIZRmMvZUwisYytmGQryWFsmXLUBhbmEK0mYhqpIUWUmnRogWlDSXt6vP3/vhf53fu17nPfV+fd9d1Pp/b/Xo9Hs9H51znupc+fc7nvLrOuc7ZxjAMwzAMwzDlKtvgBoZhGIZhGCbssMAxDMMwDMOUs7DAMQzDMAzDlLOwwDEMwzAMw5SzsMAxDMMwDMOUs7DAMQzDMAzDlLOwwDEMwzAMw5SzsMAxzmnXrp3Zf//9cXPeyP6FyAsvvGC22aZwU7pz584F/XkhJBrj+fPn403lNvLvuHz5ctycicz1Qs3hrclLL71kateubbbbbjuz66674s1O+bXObZm38veSecww5T2/vnsoUyyyWPXv37/YtsMOO6zExfmYY47BTSUm6QK3Zs0a8/777+Nm55RV4KIHJbHTTjuZunXrmlWrVuFuJUZ+R/n67CT9ILdo0SL7PT/99FO8KSeHHHIIbipIyipw2WOMQk2hCpzcD3fffXez/fbbm3322ce0bt0ad9mqVK5c2Zx++un23+i1117Dm52S9NwuKfnmxW233Zb3tnwZMmRIzv2zpLDAMb+mxL+HMOUyBx54oPnTn/6UuT506FBTqVIl07Zt26y9jOnTp48ZNGhQsW1JpawHP3nwjLP4lpU4BQ4fqGX/Dz/8sNi2kiJfV9r3TiITJ06M/eASeoHDcQ45Jc2L7Gxtgevevbsdk+uvvx5vMuPHj8dN6pS3cZff99xzzzUDBw7MbBs7dqzd3rJlS6f729VXX13m/hs2bMBNDFOuU/qMZ8p9/vrXv9r/8a9du9Ze79Spk6lfv759umXz5s2Z/WQBXLFiReb6yy+/bIqKikz16tXNeeedZ7788svMbXgETr6uWrVq9mmbiy++2EyePLlYCZH9v/76a3POOeeYnXfe2eyxxx7mp59+srdF/yPOFpW5mTNn2oV8xx13NA0bNswpmNOmTTMnnHCC+c1vfmPuuece89xzz5W6iJf0QC37/+c//7GLu4zNLrvsYqpUqWIaN25s/ve//9l98v2OUVnJjozbkUceaccOx61p06a2eE2fPt00a9bM7LvvvuaBBx6wt8kRSPwZpRW50grc22+/bX9/+XtUrVrVjlMUKRMLFizI2tuY//u//7Nz5LvvvrPXx40bZ0499VQ7Fk2aNDFjxozJ7Lu1BS76e8oRIvl3k3/bE0880cyZMyezz+zZs02LFi3sbbKPjOPKlSszt5c2N6Mxlt9bjrL+/ve/N/369TMjR460/5GRrzv44IMz+0vkd5a5JkfEZB5fd911Zt26dZnbscB9//335h//+If57W9/a3bYYQdz//33F7svZUfudzVq1DB16tTJzPl8+eKLL+zfSX7vo48+2vz3v//N3CbjJmPWtWvXzLhlj5n8jjg/oz+zk/132bhxo+nSpYs56KCD7O943HHHmffee8/ehnN706ZN5u677za/+93v7N/51ltvNevXr8/cLt/3jDPOMKNHj7a/m/zHsXfv3pnbS4p8f1l32rRpk9l21VVXmUMPPTTn548aNcq0atXK/mwZdynD0Zomfx+870jkT5nvjz76qP295ch29hG4pUuX2rVI5syWLVvs18iYyv0m+3dimFCT/9GO+VXk6aeftgtW9BSlPFhefvnlZu7cuWbKlCmZ/Ro0aJC5LA8ScpSuZ8+e5q677rKL3AEHHJC5PbvAyQOXPPV6zTXXmCeeeML8+c9/NocffnhOgZMH1UsvvdT06tXLljL53pIff/zRbmvevLl9YBbye0npkEJYr149+33lAVl+p+jp4G+++cbsueee9gFPFuk//OEPeZ8ajiIPClgsZH85Kinb5Wkt+V0efPBB+1oiKTWS6HeUfbN/R3yQicZNSoX8/aJxkwd8iTxQSGnbb7/9bAGQfwv5eilcS5YssQ+Qcl3+feRnyAN6vpRW4OR3OO2008y//vUvWxB32223TOFauHCh/ftlRx7c5ME3ijxIyr/pww8/bMdUrkdHiuIWuFmzZtkxjURjEBW4I444wj6wSoGQB8zoKLEUaXnwl3F69tln7fz74x//mCmdZc3NaIxvvvlm+/eX+bPtttuavffe2/6sxx57zBagH374IfM18jtLaTjrrLPsXJPfT/7jEwULnIyJPBUqT/U99dRT9veRf8+SIoVIvp/825YW+fevWbOmuf32280jjzxi70PylGgUGTcZM/mPTDRu2UfWBwwYYH+OzNNofsr10gqc/P7yu1922WX231qOyksZleDcjkqSlKgnn3zSXpajZ1Hk+8p9Rv4OMobynxj53tn/ecDI95ACJ4V19erVdpvc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>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,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>

[image7]: <data:image/png;base64,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>