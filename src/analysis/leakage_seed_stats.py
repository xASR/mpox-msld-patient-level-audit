"""Recompute the five-seed leakage-ablation statistics from ablation_runs.csv (no training).
The comparison is paired by SEED: within a seed, 1A and 1B are scored on different test images.
Usage: python leakage_seed_stats.py ablation_runs.csv
"""
import sys, json
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "ablation_runs.csv")
out = {}
for name, g in d.groupby("pipeline"):
    a = g.sort_values("seed")["acc"].to_numpy() * 100
    f = g.sort_values("seed")["macro_f1"].to_numpy()
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    out[name] = {"n_runs": int(len(a)), "acc_mean": a.mean(), "acc_sd": a.std(ddof=1), "acc_t95": [a.mean() - h, a.mean() + h],
                 "acc_runs": a.round(2).tolist(), "f1_mean": f.mean(), "f1_sd": f.std(ddof=1),
                 "n_test_range": [int(g["n_test"].min()), int(g["n_test"].max())],
                 "n_train_range": [int(g["n_train"].min()), int(g["n_train"].max())]}
p = d.pivot(index="seed", columns="pipeline", values="acc") * 100
gap = (p["1A_augment_then_split"] - p["1B_split_then_augment"]).to_numpy()
h = stats.t.ppf(0.975, len(gap) - 1) * gap.std(ddof=1) / np.sqrt(len(gap))
n_pos = int((gap > 0).sum())
out["gap"] = {"per_seed": gap.round(2).tolist(), "mean": gap.mean(), "sd": gap.std(ddof=1), "t95": [gap.mean() - h, gap.mean() + h],
              "paired_t": float(stats.ttest_rel(p["1A_augment_then_split"], p["1B_split_then_augment"]).pvalue),
              "seeds_favouring_1A": n_pos,
              "exact_sign_test_two_sided": float(stats.binomtest(n_pos, len(gap), 0.5).pvalue),
              "wilcoxon_exact_two_sided": float(stats.wilcoxon(gap).pvalue),
              "note": "paired by seed, not by patient; test populations differ between 1A and 1B; no per-image predictions saved"}
print(json.dumps(out, indent=2, default=float))
