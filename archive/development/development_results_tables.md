**Phase 1: Establishing the Baseline**  
**Table 1: 12-Model CNN Ablation Study (Stratified 5-Fold Cross-Validation)**

|  | Model | Accuracy | Precision | Recall | F1-Score | AUC |
| ----- | :---- | ----: | ----: | ----: | ----: | ----: |
| 1 | RESNET50 | 78.9969 | 0.779 | 0.7913 | 0.7768 | 0.9578 |
| 2 | RESNET101 | 77.6447 | 0.7617 | 0.7876 | 0.7648 | 0.9603 |
| 3 | INCEPTION\_V3 | 76.3382 | 0.7444 | 0.7732 | 0.7457 | 0.9511 |
| 4 | GOOGLENET | 70.6689 | 0.6807 | 0.7303 | 0.6924 | 0.9458 |
| 5 | MOBILENET\_V3\_LARGE | 69.5256 | 0.6941 | 0.7341 | 0.6861 | 0.9442 |
| 6 | VGG19 | 69.2518 | 0.6755 | 0.6941 | 0.6728 | 0.9266 |
| 7 | ALEXNET | 67.4264 | 0.6626 | 0.6884 | 0.6606 | 0.9257 |
| 8 | EFFICIENTNET\_B0 | 63.2965 | 0.6418 | 0.6548 | 0.6127 | 0.9162 |
| 9 | EFFICIENTNET\_B4 | 62.8962 | 0.6311 | 0.6876 | 0.6229 | 0.9126 |
| 10 | DENSENET121 | 61.8558 | 0.6563 | 0.6484 | 0.5909 | 0.9184 |
| 11 | MOBILENET\_V2 | 61.3227 | 0.632 | 0.6639 | 0.5994 | 0.9159 |
| 12 | EFFICIENTNET\_B7 | 59.2037 | 0.608 | 0.6569 | 0.5901 | 0.9046 |

**Phase 2: Optimization and Holdout Testing**  
**Table 2: Patient-Level Dataset Class Distribution (70/10/20 patient-level split utilized)**

| Disease Class | Train | Validation | Test | Total Images |
| :---- | ----- | ----- | ----- | ----- |
| Monkeypox | 187 | 45 | 52 | 284 |
| HFMD | 109 | 15 | 37 | 161 |
| Healthy | 79 | 10 | 25 | 114 |
| Chickenpox | 56 | 6 | 13 | 75 |
| Cowpox | 38 | 6 | 22 | 66 |
| Measles | 40 | 5 | 10 | 55 |
| Total | 509 | 87 | 159 | 755 |

**Table 3: 6-Model CNN Single Split Performance (with TTA and 70/10/20 patient-level split utilized)**

|  | Model Architecture | Accuracy (%) | Precision | Recall | F1-Score | AUC |
| ----- | :---- | ----: | ----: | ----: | ----: | ----: |
| 1 | ResNet50 | **84.91** | 0.8211 | 0.8343 | 0.8267 | 0.9751 |
| 2 | ResNet101 | 81.76 | 0.8485 | 0.799 | 0.8189 | 0.9691 |
| 3 | Inception\_v3 | 79.25 | 0.7973 | 0.7741 | 0.7799 | 0.9509 |
| 4 | GoogLeNet | 74.84 | 0.7609 | 0.7502 | 0.7414 | 0.9482 |
| 5 | MobileNet\_v3\_large | 67.92 | 0.7005 | 0.7029 | 0.6866 | 0.9406 |
| 6 | EfficientNet\_b4 | 54.72 | 0.6081 | 0.6505 | 0.5452 | 0.9209 |

**Phase 3: The Search for Synergies**

**Table 4: Ensemble Strategy Evaluation (70/10/20 patient-level split utilized)**

|  | Ensemble Strategy | Method | Accuracy (%) | Precision | Recall | F1-Score | AUC |
| ----- | :---- | :---- | ----: | ----: | ----: | ----: | ----: |
| 1 | Ens 3: Stacking (LogReg) | Meta-Learner (ResNet50, ResNet101, Inception, GoogLeNet) | **83.65** | 0.8345 | 0.8435 | 0.836 | 0.9733 |
| 2 | Ens 1: Trio Soft Voting | Top 3 (ResNet50, Inception\_v3, GoogLeNet) | 83.02 | 0.8262 | 0.819 | 0.8215 | 0.9725 |
| 3 | Ens 3: Stacking (XGBoost) | Meta-Learner (ResNet50, ResNet101, Inception, GoogLeNet) | 81.76 | 0.8633 | 0.7773 | 0.8105 | 0.9482 |
| 4 | Ens 2: Diverse Voting | Structural Variety (ResNet50, MobileNet\_v3, Inception\_v3) | 81.13 | 0.813 | 0.8021 | 0.8003 | 0.9691 |
| 5 | Ens 4: Disagreement Voting | Max Val Disagreement (ResNet50, GoogLeNet, MobileNet) | 80.5 | 0.7957 | 0.7996 | 0.7952 | 0.9686 |

**Phase 4: Paradigm Shift to Vision Transformers (New)**

**Table 5: Vision Transformer & CNN Hybrid Fusion Evaluation (5-Fold CV)**

|  | Architecture | Avg Val Acc (%) | Avg Test Acc (%) | Avg Macro F1 | Avg AUC | Avg ECE (Calibration) |
| ----- | :---- | ----- | ----: | ----: | ----: | ----: |
| 1 | Fusion \+ SWA (Stochastic Weight Averaging) | \- | 83.96 | 0.8157 | 0.9647 | 0.0717 |
| 2 | Stacking (Random Forest Meta-Learner) | \- | 83.78 | 0.8126 | 0.9538 | 0.0956 |
| 3 | Base Hybrid Fusion | **85.29** | 82.96 | 0.804 | 0.9648 | 0.0985 |
| 4 | Swin Transformer (Backbone Only) | 82.45 | 82.78 | 0.8058 | 0.959 | 0.0771 |
| 5 | EfficientNetV2 (Backbone Only) | 79.28 | 78.1 | 0.7597 | 0.9538 | 0.131 |
| 6 | Fusion \+ Temperature Scaling | \- | 49.71 | 0.4715 | 0.5904 | 0.1448 |

**Note:** *Validation metrics are intentionally omitted for Stacking and Temperature Scaling variants to prevent data leakage, as these methods inherently utilize the validation set for meta-fitting and calibration.*

**Phase 5: Real-World Clinical Application (Patient-Level Diagnosis)**  
**Table 6: Clinical Patient-Level Aggregation (TTA & Weighted Ensemble)**

|  | Model / Ensemble Strategy | Patient Accuracy (%) | Patient Macro F1 | Patient ROC-AUC | Patient ECE (Calibration) |
| ----- | :---- | ----- | ----- | ----- | ----- |
| 1 | Swin-Tiny (Vision Transformer \+ TTA) | **86.83** | 0.856 | 0.976 | 0.0436 |
| 2 | Weighted Ensemble (75% Swin / 25% EffNet) | **86.07** | 0.845 | 0.9763 | 0.0344 |
| 3 | EfficientNetV2-S (CNN \+ TTA) | 79.01 | 0.7709 | 0.9465 | 0.0809 |

**Table 7 : Performance Optimization of Patient-Level Ensemble via Stochastic Threshold Tuning**

| Method | Baseline Accuracy | Optimized Accuracy | Improvement |
| :---- | ----- | ----- | :---- |
| EfficientNetV2-S (Baseline) | 79.01% | \- | \- |
| Swin-Tiny (Baseline) | **86.83%** | **87.60%** | **\+0.77%** |
| Swin \+ EffNet Ensemble | **86.07%** | **87.98%** | **\+1.91%** |

**Note:** *Post-processing threshold optimization was intentionally omitted for the standalone EfficientNetV2-S model. Following standard ablation methodology, this architecture was pruned from the final hyper-tuning phase because its baseline performance (79.01%) was significantly outpaced by the top-tier candidates. Computational resources for stochastic threshold tuning were exclusively reserved for the highest-scoring models (Swin-Tiny and the Hybrid Ensemble) to determine the study's absolute peak accuracy.*

### **Phase 6: State-of-the-Art Methodological Refinement (V8 Pipeline)**

*To maximize clinical reliability and eliminate potential data leakage from static augmentations, the final pipeline transitioned to a rigorously controlled 5-Fold Stratified Group Cross-Validation strategy exclusively utilizing the 756 original clinical images. Advanced regularization techniques, including MixUp fade-out, Cosine Annealing scheduling, and confidence-weighted patient aggregation, were applied to four structurally diverse architectures.*

**Table 8: Performance of Isolated Advanced Architectures (Patient-Level, 5-Fold CV)**

| Architecture | Family | Patient Accuracy | Precision | Recall | Macro F1 | ROC-AUC | ECE (Calibration) |
| :---- | :---- | ----: | ----: | ----: | ----: | ----: | ----: |
| Swin-Small | Local ViT | **88.57%** | 0.877 | 0.8766 | 0.8751 | 0.9825 | 0.1247 |
| ConvNeXt-Tiny | Modern CNN | 87.43% | 0.8543 | 0.8707 | 0.8599 | 0.9733 | 0.1323 |
| BEiT-Base | Global ViT (MIM) | 85.14% | 0.8406 | 0.8259 | 0.832 | 0.9704 | 0.0825 |
| EfficientNetV2-S | Baseline CNN | 72.76% | 0.7255 | 0.7385 | 0.7141 | 0.9411 | 0.061 |

**Table 9: Champion Ensemble Calibration and Weight Distribution**  
*Rather than relying on basic soft-voting or manual threshold tuning (as seen in earlier phases), the top three architectures were pruned and mathematically calibrated. Temperature Scaling (T) was optimized via L-BFGS-B to correct model overconfidence, and final ensemble weights were dynamically assigned based on log-loss minimization.*

| Pruned Architecture | L-BFGS-B Temperature (T) | Optimized Ensemble Weight |
| :---- | ----- | ----- |
| Swin-Small | 0.71 | 77.49% |
| ConvNeXt-Tiny | 0.75 | 15.23% |
| BEiT-Base | 0.84 | 7.28% |

**Table 10: Final Clinical Patient-Level Evaluation (Pruned Calibrated Ensemble)**  
*The final evaluation demonstrates the critical clinical value of temperature-scaled ensembling. The ensemble provided targeted gains in Accuracy (reaching 88.76% with a 95% CI of 86.10% \-- 91.43%) and Macro F1, but its most significant contribution was plummeting the Expected Calibration Error (ECE) to near-zero.*

| Configuration | Patient Accuracy (95% CI) | Precision | Recall | Macro F1 | ROC-AUC | ECE (Calibration) |
| :---- | :---- | ----- | ----- | ----- | ----- | ----- |
| Best Single Model (Swin) | 88.57% | 0.877 | 0.8766 | 0.8751 | 0.9825 | 12.47% |
| Champion Calibrated Ensemble | **88.76%** \[86.10%, 91.43%\] | 0.879 | 0.8794 | 0.8772 | 0.9831 | **3.70%** |

### 

### 

### 

### 

### **Table 11: Controlled Leakage Audit (Identical Training Budget and Architecture)**

*An ablation study isolating the impact of augmentation timing on test accuracy. Both pipelines utilized the identical ViT-Tiny architecture and a \~6,000 image training budget. Pipeline A replicates the methodology of Pal et al. (2025), exposing a severe artificial inflation of metrics.*

| Factor | Pipeline A: Augment-Then-Split (Pal et al. Method) | Pipeline B: Split-Then-Augment (Strict Isolation) |
| :---: | ----- | ----- |
| Original images | 755 | 755 |
| Augmentation multiplier | 10× (before split) | 10× (training set only) |
| Training samples | \~6,040 | \~6,040 |
| Test samples | \~1,510 (contain clones of training patients) | \~151 (unseen patients only) |
| Architecture | ViT-Tiny (identical) | ViT-Tiny (identical) |
| Training epochs | 100 (identical) | 100 (identical) |
| Test Accuracy | **89.47%** | **62.14%** |
| Accuracy Gap | **\~ 27.33**  | **percentage points** |
| Interpretation | Inflated by patient-clone memorization | Clinically honest generalization |

