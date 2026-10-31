"""Compare the re-trained ConvNeXt-Tiny (V34-W) pooled predictions with the ORIGINAL saved predictions.

Pre-specified before the re-run finished (see KAGGLE_RUN_RECOVERY.md). It does not change the ensemble.

Usage:
    python compare_convnext_rerun.py --orig <folder with original convnext_tiny_*.npy> \
                                     --new  <folder with re-run convnext_tiny_*.npy> \
                                     --out  <output folder>

Both folders must contain convnext_tiny_{test,val}_{probs,labels,pids}.npy (the six pooled arrays).
The original arrays are in the repository at results/predictions/convnext_tiny/.
Outputs: rerun_comparison.json, rerun_comparison.md (a ready-to-paste summary), per_patient_test.csv.
"""
import argparse, json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, log_loss, cohen_kappa_score,
                             precision_score, recall_score)

C = 6
NAMES = ["Healthy", "Cowpox", "Mpox", "Measles", "Chickenpox", "HFMD"]   # V34 class order (index 0..5)


def load(folder, split):
    a = {}
    for k in ("probs", "labels", "pids"):
        path = os.path.join(folder, f"convnext_tiny_{split}_{k}.npy")
        if not os.path.exists(path):                       # allow nested folders
            hits = [os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs
                    if f == f"convnext_tiny_{split}_{k}.npy"]
            assert hits, f"missing {path}"
            path = hits[0]
        a[k] = np.load(path, allow_pickle=(k == "pids"))
    p = a["probs"].astype(np.float64)
    a["probs"] = p / p.sum(1, keepdims=True)
    a["labels"] = a["labels"].astype(int)
    a["pids"] = np.array([str(x) for x in a["pids"]])
    return a


def ece(p, y, n_bins=10):
    b = np.linspace(0, 1, n_bins + 1); conf = p.max(1); acc = p.argmax(1) == y; e = 0.0
    for i in range(n_bins):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.mean() > 0:
            e += abs(conf[m].mean() - acc[m].mean()) * m.mean()
    return float(e)


def metrics(p, y):
    pr = p.argmax(1)
    return {"accuracy": accuracy_score(y, pr), "macro_precision": precision_score(y, pr, average="macro", zero_division=0),
            "macro_recall": recall_score(y, pr, average="macro", zero_division=0),
            "macro_f1": f1_score(y, pr, average="macro"),
            "macro_auc": roc_auc_score(y, p, multi_class="ovr", labels=list(range(C))),
            "ece": ece(p, y), "log_loss": log_loss(y, p, labels=list(range(C))), "correct": int((pr == y).sum()),
            "n": int(len(y))}


def compare(o, n, split):
    out = {"split": split}
    out["same_patient_order"] = bool(np.array_equal(o["pids"], n["pids"]))
    out["same_labels"] = bool(np.array_equal(o["labels"], n["labels"])) if out["same_patient_order"] else False
    if not (out["same_patient_order"] and out["same_labels"]):
        so, sn = set(o["pids"]), set(n["pids"])
        out["only_in_original"] = sorted(so - sn)[:50]
        out["only_in_rerun"] = sorted(sn - so)[:50]
        out["STOP"] = "Patient sets or labels differ: the folds were not reproduced. Do not compare further."
        return out, None
    po, pn, y = o["probs"], n["probs"], o["labels"]
    d = np.abs(po - pn)
    agree = po.argmax(1) == pn.argmax(1)
    out.update({
        "n_patients": int(len(y)),
        "top1_agreement_n": int(agree.sum()), "top1_agreement_pct": float(agree.mean() * 100),
        "cohen_kappa": float(cohen_kappa_score(po.argmax(1), pn.argmax(1))),
        "max_abs_prob_diff": float(d.max()), "mean_abs_prob_diff": float(d.mean()),
        "mean_abs_true_class_prob_diff": float(np.abs(po[np.arange(len(y)), y] - pn[np.arange(len(y)), y]).mean()),
        "exactly_reproduced": bool(d.max() <= 1e-6),
        "original": metrics(po, y), "rerun": metrics(pn, y),
    })
    co, cn = po.argmax(1) == y, pn.argmax(1) == y
    b_, c_ = int((co & ~cn).sum()), int((~co & cn).sum())
    out["mcnemar"] = {"original_only_correct": b_, "rerun_only_correct": c_,
                      "exact_p": float(binomtest(b_, b_ + c_, 0.5).pvalue) if b_ + c_ else 1.0}
    per = pd.DataFrame({"pid": o["pids"], "label": [NAMES[i] for i in y],
                        "orig_pred": [NAMES[i] for i in po.argmax(1)], "rerun_pred": [NAMES[i] for i in pn.argmax(1)],
                        "orig_conf": po.max(1), "rerun_conf": pn.max(1), "agree": agree,
                        "max_abs_diff": d.max(1)})
    return out, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True); ap.add_argument("--new", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    res = {}
    for split in ("test", "val"):
        r, per = compare(load(a.orig, split), load(a.new, split), split)
        res[split] = r
        if per is not None and split == "test":
            per.to_csv(os.path.join(a.out, "per_patient_test.csv"), index=False)
    json.dump(res, open(os.path.join(a.out, "rerun_comparison.json"), "w"), indent=2)
    t = res["test"]
    if "STOP" in t:
        md = f"STOP: {t['STOP']}\n"
    else:
        o, n = t["original"], t["rerun"]
        verdict = ("reproduced exactly (maximum absolute probability difference <= 1e-6)" if t["exactly_reproduced"]
                   else "re-trained with identical code and configuration; predictions are not identical")
        md = (f"ConvNeXt-Tiny re-run vs original, {t['n_patients']} pooled test patients: {verdict}.\n\n"
              f"- Top-1 agreement: {t['top1_agreement_n']} of {t['n_patients']} ({t['top1_agreement_pct']:.1f}%), "
              f"Cohen's kappa {t['cohen_kappa']:.3f}\n"
              f"- Absolute probability difference: mean {t['mean_abs_prob_diff']:.4f}, maximum {t['max_abs_prob_diff']:.4f}\n"
              f"- Original: accuracy {o['accuracy']*100:.2f}% ({o['correct']}/{o['n']}), macro F1 {o['macro_f1']:.4f}, "
              f"macro AUC {o['macro_auc']:.4f}, ECE {o['ece']*100:.2f}%, log-loss {o['log_loss']:.4f}\n"
              f"- Re-run:   accuracy {n['accuracy']*100:.2f}% ({n['correct']}/{n['n']}), macro F1 {n['macro_f1']:.4f}, "
              f"macro AUC {n['macro_auc']:.4f}, ECE {n['ece']*100:.2f}%, log-loss {n['log_loss']:.4f}\n"
              f"- McNemar (exact): original-only correct {t['mcnemar']['original_only_correct']}, "
              f"re-run-only correct {t['mcnemar']['rerun_only_correct']}, p = {t['mcnemar']['exact_p']:.3f}\n")
        v = res["val"]
        if "STOP" not in v:
            md += (f"- Calibration-half patients ({v['n_patients']}): top-1 agreement {v['top1_agreement_pct']:.1f}%, "
                   f"max abs difference {v['max_abs_prob_diff']:.4f}\n")
    open(os.path.join(a.out, "rerun_comparison.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
