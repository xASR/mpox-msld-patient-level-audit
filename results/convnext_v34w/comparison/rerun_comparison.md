ConvNeXt-Tiny re-run vs original, 216 pooled test patients: reproduced exactly (maximum absolute probability difference <= 1e-6).

- Top-1 agreement: 216 of 216 (100.0%), Cohen's kappa 1.000
- Absolute probability difference: mean 0.0000, maximum 0.0000
- Original: accuracy 87.50% (189/216), macro F1 0.8633, macro AUC 0.9835, ECE 12.90%, log-loss 0.4563
- Re-run:   accuracy 87.50% (189/216), macro F1 0.8633, macro AUC 0.9835, ECE 12.90%, log-loss 0.4563
- McNemar (exact): original-only correct 0, re-run-only correct 0, p = 1.000
- Calibration-half patients (217): top-1 agreement 100.0%, max abs difference 0.0000
