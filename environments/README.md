# Recorded execution environments

Training and Grad-CAM ran on Kaggle. This repository does not claim that the laptop has the Kaggle GPU environment installed.

| Run | Recorded environment | Record |
|---|---|---|
| Original Protocol A/B analyses | Python 3.12, PyTorch 2.11, timm 1.0.29, NumPy 2.1.3, SciPy 1.16.3, scikit-learn 1.6.1 | Original logs and root requirements.txt |
| V34-W ConvNeXt re-run | Python 3.12.12, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, timm 1.0.25, NumPy 2.0.2, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.6.1, Tesla T4 | results/convnext_v34w/run_info.json |
| Grad-CAM verification session | PyTorch 2.11.0+cu128, timm 1.0.29, CUDA device | Saved notebook console and results/gradcam/weight_verification.csv |

Both sets of comparisons reproduced the relevant saved predictions exactly. Version records describe the executed runs; installing the root requirements is not a claim to recreate every recorded environment.

To rebuild the publication panels on Windows with the original Arial fonts:

```powershell
python src/figures/make_gradcam_figures.py --gradcam results/gradcam --out figures/gradcam_rebuilt --font-regular C:/Windows/Fonts/arial.ttf --font-bold C:/Windows/Fonts/arialbd.ttf
```

This CPU layout step does not retrain models or recompute Grad-CAM maps. Checkpoints from the original runs were never saved. The ten archived `.pt` files are from the verified V34-W ConvNeXt re-run.
