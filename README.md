# Augmentation leakage and duplicate-aware evaluation of Mpox skin lesion classifiers

This project audits augmentation leakage in the Mpox Skin Lesion Dataset v2.0 (MSLD v2.0) and evaluates six-class image classifiers under duplicate-aware splits. **Protocol A** is the primary full-coverage benchmark. **Protocol B** is a secondary benchmark based on the published workbook assignments. Controlled exposure experiments quantify how the training and test partition affects the reported accuracy.

## Data and evaluation units

- The clinical inventory contains **755 original images**, **524 filename identifiers**, and **520 known groups** after linking four confirmed cross-identifier duplicate pairs. A known group is a connected component defined using filename identifiers, exact matches, and confirmed duplicate links. Clinical identities were unavailable.
- The model output order is Healthy, Cowpox, Mpox, Measles, Chickenpox, and hand, foot, and mouth disease (HFMD). The frozen model and prediction records use the label `Monkeypox` for Mpox.
- The first duplicate screen adjudicated all 46 candidates. The broader search produced 2,127 candidates, of which 103 were visually reviewed. It confirmed no additional links and left 2,024 lower-priority pairs unadjudicated.
- All **11,325 images** in version 1 of the Mendeley MCVSLD record matched MSLD v2.0 files byte for byte. This release provided no independent external validation images. [MCVSLD version 1](https://data.mendeley.com/datasets/dfztdtfsxz/1)
- Obtain the clinical originals, released augmentation folders, and `datalog.xlsx` from the dataset distributor. The frozen inventory specifies the archive and eligible inputs for each experiment. [MSLD v2.0 on Kaggle](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20) · [MSLD v2.0 dataset paper](https://doi.org/10.1016/j.bspc.2024.106742)

## Main evaluation protocols

| Design | Split and training | Held-out evaluation |
| --- | --- | --- |
| **A · Full coverage** | Five stratified outer group folds, with disjoint inner checkpoint-selection and calibration groups. Original training images receive online augmentation. The primary outer split seed is 42. | Each of the 755 original images and 520 known groups is tested once. Temperatures and mixture weights are fitted within each fold. |
| **B · Official assignments** | Five workbook folds. Within each fold, overlapping groups are assigned by priority in the order test, calibration, selection, then training. Released offline augmentations are restricted to sources that remain eligible for training. Training also uses online transforms. | 364 test-image appearances represent **297 distinct images (39.34%)**. The 264 test-group appearances represent **215 distinct groups (41.35%)**. Repeated held-out predictions are pooled by group. Removing 79 pooled test groups from 217 calibration candidates leaves **138 calibration groups**. |

### Backbone configurations

All four main backbones use ImageNet-pretrained weights, a six-class head, and 384 × 384 inputs. ViT-Tiny is used in the leakage experiments alongside ConvNeXt-Tiny.

| Backbone | `timm` identifier | Input (px) | Parameters (M) | Use |
| --- | --- | ---: | ---: | --- |
| EfficientNetV2-S | `tf_efficientnetv2_s.in21k_ft_in1k` | 384 | 20.2 | A, B, configuration sensitivity |
| ConvNeXt-Tiny | `convnext_tiny.in12k_ft_in1k_384` | 384 | 27.8 | A, B, leakage experiments |
| Swin-Small | `swin_small_patch4_window7_224.ms_in22k_ft_in1k` | 384 | 48.8 | A, B |
| BEiT-Base | `beit_base_patch16_384.in22k_ft_in22k_in1k` | 384 | 86.0 | A, B, precision sensitivity |
| ViT-Tiny | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` | 128 | 5.5 | Leakage experiments |

Both main protocols train for **25 epochs** at batch size **16** using mixed precision. AdamW uses a head learning rate of **0.001** for epochs 1–3, then **0.0001** for full fine-tuning, weight decay **0.01**, and a cosine schedule with two warm-up epochs and minimum learning rate **0.000001**. Main training transformations use a random resized crop (area scale 0.5–1.0), horizontal flips (probability 0.5), rotations up to 15°, brightness/contrast/saturation jitter of 0.4, hue jitter of 0.1, 3 × 3 Gaussian blur, and ImageNet normalization.

| Setting | Protocol A | Protocol B |
| --- | --- | --- |
| Mixing, epochs 1–10 | Mixup α = 0.2 and CutMix α = 1.0 | Mixup α = 0.1 and CutMix disabled |
| Hard-label loss, epochs 11–25 | Inverse-frequency weighted cross-entropy | Effective-number weighted cross-entropy, β = 0.999 |
| Label smoothing | 0.1 | 0.1 |
| Checkpoint selection | Two lowest selection-group log-loss epochs per fold and model | Same rule |
| Test views | Identity, horizontal flip, and 0.9× and 1.1× rescaling | Identity, horizontal flip, and 90° and 270° rotations |
| Probability aggregation | Average view logits before softmax, average checkpoint probabilities per image, then form a confidence-weighted group mean | Average view probabilities, form a normalized geometric group mean per checkpoint, then average the two checkpoint vectors |
| Final calibration | Per outer fold | Once on the pooled eligible calibration cohort |

Each backbone’s temperature is fitted by minimizing calibration-group log-loss with a bound of **[0.1, 5.0]**. Softmax-parameterized mixture weights minimize log-loss on the calibration groups. Main classification uses the maximum class probability. The manifests pin class order, split membership, pretrained revisions, and job configurations.

### Main results

| Ensemble | Correct / evaluated | Accuracy (%) | Macro F1 | Macro AUC | Ten-bin ECE (%) | Log-loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **A · Original images** | 611 / 755 | **80.93** | 0.7959 | 0.9674 | 5.27 | 0.5536 |
| **A · Known groups** | 442 / 520 | **85.00** | 0.8333 | 0.9748 | 2.76 | 0.4509 |
| B · Pooled official test groups | 189 / 215 | 87.91 | 0.8627 | 0.9794 | 6.25 | 0.4195 |

| Raw backbone | Protocol A image accuracy (%) | Protocol A group accuracy (%) | Protocol B group accuracy (%) |
| --- | ---: | ---: | ---: |
| EfficientNetV2-S | 62.38 | 69.42 | 82.79 |
| ConvNeXt-Tiny | 81.72 | 85.77 | 86.05 |
| Swin-Small | 80.93 | 85.00 | 85.12 |
| BEiT-Base | 71.79 | 76.15 | 85.58 |

AUC is the macro one-versus-rest ROC area. Group-accuracy Wilson 95% CIs for the ensemble are **81.67%–87.81%** for A and **82.87%–91.61%** for B. The cohorts and evaluation units differ across designs. Protocol A preserves the initial BEiT-Base fold-2 fit. Substituting the repeat and refitting only that fold’s ensemble left overall group accuracy at 442/520.

## Leakage experiments

The released-pool comparison uses **32,777 source–pixel combinations** (755 originals and 32,022 distinct within-source augmentation items), derived from the released folders by source filename and decoded-pixel digest. Every item-random test item shared a source with training. Reconstructed splits of the 7,532-entry folders were audited separately for ancestry.

| Backbone | Item-random accuracy (%) | Group-disjoint accuracy (%) |
| --- | ---: | ---: |
| ViT-Tiny, 128 px | 95.85 | 53.89 |
| ConvNeXt-Tiny, 384 px | 99.88 | 83.60 |

The released-pool values are means over five splits on all held-out items. In the separate matched-exposure study, each pair shared the original test cohort, initialization, and scheduled training budget. Inserting eight same-source relatives per test original into same-class training slots increased original-image accuracy by **38.09 percentage points for ViT-Tiny** and **16.08 points for ConvNeXt-Tiny** over ten seeds. Both exact two-sided sign tests were reported as **p = 0.0020**. The zero- and eight-relative endpoints use seeds 0–9. The two- and four-relative ConvNeXt-Tiny analyses use seeds 0–4.

ViT-Tiny uses 100 epochs, batch size 32, Adam at 0.001, and a single evaluation view. ConvNeXt-Tiny uses 25 epochs, batch size 16, the staged Protocol A optimizer and loss recipe, and a single evaluation view. Both exposure experiments evaluate the final epoch.

## Additional analyses

| Analysis | Configuration and reported result |
| --- | --- |
| **Metadata and border features** | On the frozen Protocol A folds, separate logistic regressions and random forests used non-pixel file/geometry features (Tier 1A) or the same features plus statistics from the outer 12-pixel band and corners (Tier 1B). The Tier 1A random forest reached **55.77%** group accuracy versus **26.92%** for the group majority baseline. The Tier 1B random forest reached **57.50%**. |
| **EfficientNetV2-S configuration sensitivity** | On the same 215 official test groups, A-style training and A inference gave **148/215 (68.84%)**. Replicating the scheduled B budget with freshly shuffled passes over the original training images gave **179/215 (83.26%)** under A inference, a paired gain of **0.1442** (95% CI 0.0791 to 0.2093, Holm-adjusted McNemar **p = 0.0002**). Folds 1–5 used 919, 868, 910, 886, and 914 mini-batches of 16 per epoch. The full supplement includes loss/mixing and inference-recipe crossovers. |
| **Mpox operating points** | Each fold’s disjoint calibration set selects the Youden’s J, 85%, or 90% sensitivity rule. The 95% minimum-positive rule is secondary. At the 85% and 90% targets, primary held-out ensemble sensitivity was **77.44%** for both and specificity was **92.51%** and **87.86%**, respectively. Stability is reported across six Protocol A split seeds. |
| **Partial exposure** | ConvNeXt-Tiny uses one relative per exposed original. We shuffle test groups within each class with random seed 615000 plus the experiment seed. A single class-proportional order supplies nested sets of 10, 26, 52, or 104 of the 104 test groups. At nominal 50% exposure, accuracy was **99.23%** on exposed images and **85.75%** on unexposed images. The five-seed clean anchor was **83.03%** overall. |
| **BEiT-Base precision** | On Protocol A fold 2 at seeds 344–348, four microbatches of four preserved an effective batch of 16. The clipped FP16 arm capped the finite global L2 norm of accumulated, unscaled gradients at **1.0** before the optimizer step. Mean accuracy ± SD was **0.8400 ± 0.0395** under unclipped FP16, **0.8019 ± 0.1253** under clipped FP16, and **0.8381 ± 0.0067** under FP32. |
| **Stability and residual similarity** | Five additional Protocol A split seeds are 17, 73, 101, 137, and 211. The frozen-prediction residual-similarity analysis used an eight-bit distance cutoff and flagged four test groups. Excluding these groups yielded **438/516 (84.88%)**, compared with the primary 442/520. |
| **Ensemble attribution and perturbation** | The supplementary Protocol B analysis uses all 40 retained checkpoints. The true-class probability contrast between border masking and equal-area internal controls was **0.0351** with mean-color fill and **0.0036** with blur fill. The supplement includes parameter-randomization checks and representative ensemble Grad-CAM maps. |

Independent external validation and manually annotated tests of eye bars, watermarks, and other source marks were outside the study. Border perturbations and Grad-CAM provide model diagnostics. Clinical localization and artifact-specific reliance require independent evidence.

## Reproduction

Start with the **saved-prediction** workflow to rebuild metrics, calibration summaries, and tables. Use the **training** and **attribution** notebooks when checkpoint generation or image-based reevaluation is required. Within each frozen bundle, preserve its original internal paths and manifest checks. Use the [reproduction runbook](docs/reproduce.md), [artifact registry](reproducibility/artifacts.tsv), and [results index](results/README.md) to locate the inputs and saved outputs.

| Workflow | Entry point | Required material |
| --- | --- | --- |
| Protocol A preflight and training | `notebooks/protocol_a/Protocol_A_755_Revised_Run.ipynb` | Distributor’s MSLD v2.0 release, the extracted Protocol A design, and the release code dataset. GPU for training. |
| Protocol B publication analysis | `notebooks/protocol_b/01_CPU_Final_Publication.ipynb` | `PB_Final_Verified_Inputs.zip` with all 20 completed jobs, plus the release code dataset. CPU. |
| Released-pool and matched-exposure tables | `notebooks/leakage/04_CPU_Final_Report.ipynb` | Frozen leakage input bundle and the completed jobs required by its merger, including the seed 0–4 baseline records. CPU. |
| Five additional analyses | `notebooks/extensions/04_CPU_Consolidate.ipynb` | `MSLD_Final_CPU_Inputs.zip` with the sealed design and four completed result sets. CPU for consolidation. |
| Ensemble attribution | `notebooks/attribution/02_GPU_Ensemble_GradCAM.ipynb` | Original clinical images, the verified B inputs, the 40 selected checkpoint tensors, and the release code dataset. Use the quantitative GPU validation entry point and its supplemental inputs for the perturbation and randomization tables. |

| Location | Contents |
| --- | --- |
| `src/` | Inventory, grouping, training, aggregation, calibration, leakage experiments, and supplementary analyses. |
| `notebooks/` | CPU table reproduction and GPU training or attribution entry points. |
| `reproducibility/` and `environments/` | Artifact checksums, frozen design and weight identifiers, and run-specific dependency records. |
| `results/` and `figures/` | Canonical results tables and the figures used in the manuscript and supplement. |
| `docs/` | Reproduction instructions, data provenance, and artifact-to-manuscript mapping. |

The saved run records pin the environment per job. For example, the Protocol A training records report Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, and timm 1.0.29. Trained tensors and resumable training state are stored with saved training outputs. The CPU workflow uses archived probabilities and verified split records.

## Citation and license

Cite *Augmentation leakage and duplicate-aware evaluation of Mpox skin lesion classifiers* and the dataset sources above. The research code is available under the repository’s [MIT license](LICENSE). Source images and derived image displays follow the respective providers’ terms.
