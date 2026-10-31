"""Protocol B sensitivity analysis: ensemble metrics without the test patients whose photograph has an
identical copy (under another patient ID) in a training partition of the official folds.
Re-runs the V34-MERGE (merge_ensemble.py) on the saved arrays, then excludes the affected patients,
keeping the fitted temperatures and weights unchanged.
Usage: python duplicate_sensitivity.py <predictions dir> <out json>
"""
import sys, json, os
import numpy as np
AFFECTED = ["CHP_26", "HFMD_130", "HEALTHY_21", "HEALTHY_77"]   # from msld_cross_photo_pairs_in_datalog.csv
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "../protocol_b/merge_ensemble.py")).read()
cut = src.index("fixed, leaky = run(True), run(False)") + len("fixed, leaky = run(True), run(False)")
out_json = sys.argv[2] if len(sys.argv) > 2 else "duplicate_sensitivity.json"
sys.argv = ["merge", sys.argv[1], "/tmp/_merge_tmp"]
ns = {"__name__": "merge"}
exec(compile(src[:cut], "merge_ensemble", "exec"), ns)
fixed, metrics = ns["fixed"], ns["metrics"]
pids, y, ens = np.array([str(p) for p in fixed["pids"]]), fixed["y"], fixed["ens"]
cnx = fixed["singles_cal"][list(ns["MODELS"]).index("convnext_tiny")]
keep = ~np.isin(pids, AFFECTED)
res = {"affected_present": [p for p in AFFECTED if p in set(pids)],
       "affected_predictions": {p: {"label": int(y[pids == p][0]), "ensemble": int(ens[pids == p][0].argmax()),
                                    "convnext_scaled": int(cnx[pids == p][0].argmax())} for p in AFFECTED if p in set(pids)},
       "ensemble_all": {**{k: float(v) for k, v in metrics(ens, y).items()}, "n": int(len(y)), "correct": int((ens.argmax(1) == y).sum())},
       "ensemble_excluding": {**{k: float(v) for k, v in metrics(ens[keep], y[keep]).items()}, "n": int(keep.sum()),
                              "correct": int((ens[keep].argmax(1) == y[keep]).sum())},
       "convnext_scaled_excluding": {**{k: float(v) for k, v in metrics(cnx[keep], y[keep]).items()},
                                     "correct": int((cnx[keep].argmax(1) == y[keep]).sum())}}
json.dump(res, open(out_json, "w"), indent=2)
print(json.dumps(res, indent=2))
