# %% [markdown]
# # Grad-CAM for the re-trained ConvNeXt-Tiny (V34-W weights)
# Explains ConvNeXt-Tiny only, not the four-backbone ensemble. Qualitative: MSLD v2.0 has no lesion masks.
# Kaggle settings: Accelerator GPU T4 (CPU also works, slower). Internet can be off.
# Inputs: (1) MSLD v2.0 (joydippaul); (2) your private dataset made from the V34-W notebook output
#         (must contain outputs/weights/*.pt, *_meta.json, *_assignment.csv and outputs/convnext_tiny_fold*_test_*.npy).
# Run the cells in order. Cell 2 must pass before anything else is trusted.

# %% CELL 1: configuration, inputs, checks
import os, re, json, glob, gc
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
import timm
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

INPUT_ROOT = Path(os.environ.get("GC_INPUT_ROOT", "/kaggle/input"))
OUT = Path(os.environ.get("GC_WORK", "/kaggle/working")) / "gradcam"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMM_NAME = "convnext_tiny.in12k_ft_in1k_384"
CLASS_NAMES = ["Healthy", "Cowpox", "Monkeypox", "Measles", "Chickenpox", "HFMD"]   # V34 index order
IMG_SIZE = 384
VAL_TF = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])   # exact V34 evaluation transform
print("torch", torch.__version__, "| timm", timm.__version__, "| device", DEVICE)

w1 = sorted(INPUT_ROOT.rglob("convnext_tiny_fold1_rank1.pt"))
assert w1, "V34-W weights not found: attach the dataset made from the V34-W notebook output"
WDIR = w1[0].parent
weights = {k: [WDIR / f"convnext_tiny_fold{k}_rank{r}.pt" for r in (1, 2)] for k in range(1, 6)}
missing = [str(p) for v in weights.values() for p in v if not p.exists()]
assert not missing, f"missing weight files: {missing}"
meta = {k: json.load(open(WDIR / f"convnext_tiny_fold{k}_meta.json")) for k in range(1, 6)}
assign = {k: pd.read_csv(WDIR / f"convnext_tiny_fold{k}_assignment.csv") for k in range(1, 6)}
for k in range(1, 6):
    assert meta[k]["class_names"] == CLASS_NAMES and meta[k]["img_size"] == IMG_SIZE, f"fold {k} metadata mismatch"
    print(f"fold {k}: epochs kept {[s['epoch'] for s in meta[k]['snapshots']]}, "
          f"test images {int((assign[k]['role'] == 'test').sum())}, "
          f"weight files {[round(p.stat().st_size / 2**20, 1) for p in weights[k]]} MiB")

orig_dir = min((p for p in INPUT_ROOT.rglob("*") if p.is_dir() and p.name.lower() == "original images"),
               key=lambda p: len(p.parts))
stem2path = {}
for p in sorted(orig_dir.rglob("*")):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        stem2path.setdefault(p.stem.strip().upper(), p)
print("MSLD original photographs found:", len(stem2path))
assert len(stem2path) == 755, "expected 755 MSLD original photographs"


def load_model(path):
    m = timm.create_model(TIMM_NAME, pretrained=False, num_classes=len(CLASS_NAMES))
    sd = torch.load(path, map_location="cpu")
    m.load_state_dict(sd, strict=True)                    # strict: architecture must match exactly
    return m.to(DEVICE).eval()


def image_path(fname):
    key = Path(str(fname)).stem.strip().upper()
    assert key in stem2path, f"image {fname} not found among MSLD originals"
    return stem2path[key]


# %% CELL 2: verify the weights reproduce the re-run's own saved test predictions (V34 pipeline)
@torch.no_grad()
def tta_logprob(model, x):                                 # identical to V34 tta_predict(use_tta=True)
    views = [x, torch.flip(x, [3]), torch.rot90(x, 1, [2, 3]), torch.rot90(x, 3, [2, 3])]
    probs = torch.stack([F.softmax(model(v), dim=1) for v in views], 0)
    return torch.log(probs.mean(0) + 1e-8)

def patient_probs(logp, pids):                             # identical to V34 get_patient_level_arrays
    df = pd.DataFrame({"pid": pids, "lp": list(logp)})
    agg = df.groupby("pid")["lp"].apply(lambda s: np.mean(np.stack(s.values), 0)).sort_index()
    z = np.stack(agg.values); z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / (e.sum(1, keepdims=True) + 1e-8), agg.index.to_numpy()

verify = []
for k in range(1, 6):
    te = assign[k][assign[k]["role"] == "test"].reset_index(drop=True)
    x_all = torch.stack([VAL_TF(Image.open(image_path(f)).convert("RGB")) for f in te["filename"]])
    per_snap = []
    for wp in weights[k]:
        model = load_model(wp)
        lp = torch.cat([tta_logprob(model, x_all[i:i + 16].to(DEVICE)).cpu() for i in range(0, len(x_all), 16)]).numpy()
        pp, pids = patient_probs(lp, te["patient_id"].astype(str).to_numpy())
        per_snap.append(pp); del model
    mine = np.mean(per_snap, 0)
    saved_p = sorted(INPUT_ROOT.rglob(f"convnext_tiny_fold{k}_test_probs.npy"))
    if saved_p:
        saved = np.load(saved_p[0]); saved_ids = np.load(str(saved_p[0]).replace("_probs", "_pids"), allow_pickle=True)
        same_ids = np.array_equal(np.array([str(s) for s in saved_ids]), pids.astype(str))
        diff = float(np.abs(saved - mine).max()) if same_ids else float("nan")
        verify.append({"fold": k, "patients": len(pids), "same_patient_order": bool(same_ids), "max_abs_diff": diff})
    else:
        verify.append({"fold": k, "patients": len(pids), "same_patient_order": None, "max_abs_diff": None})
verify = pd.DataFrame(verify); print(verify.to_string(index=False))
verify.to_csv(OUT / "weight_verification.csv", index=False)
ok = verify["max_abs_diff"].dropna()
assert len(ok) == 5 and (ok < 1e-3).all(), ("The weights do not reproduce the re-run's saved fold predictions "
                                            "(or the per-fold .npy files are missing). Stop and investigate.")
print("Weights verified: they reproduce the saved per-fold test predictions.")


# %% CELL 3: image-level predictions (identity view, mean of the fold's two checkpoints) and fixed selection
rows = []
for k in range(1, 6):
    te = assign[k][assign[k]["role"] == "test"]
    models = [load_model(wp) for wp in weights[k]]
    for r in te.itertuples():
        x = VAL_TF(Image.open(image_path(r.filename)).convert("RGB"))[None].to(DEVICE)
        with torch.no_grad():
            p = np.mean([F.softmax(m(x), 1).cpu().numpy()[0] for m in models], 0)
        rows.append({"filename": Path(str(r.filename)).name, "patient_id": r.patient_id, "fold": k,
                     "true": int(r.label), "pred": int(p.argmax()), "conf": float(p.max()), **{f"p_{c}": float(v) for c, v in zip(CLASS_NAMES, p)}})
    del models
pred = pd.DataFrame(rows)
# rule fixed in advance: each test image is explained by the LOWEST-numbered fold in which it is a test image
pred = pred.sort_values(["filename", "fold"]).drop_duplicates("filename", keep="first").reset_index(drop=True)
pred["correct"] = pred["true"] == pred["pred"]
pred.to_csv(OUT / "image_predictions_identity_view.csv", index=False)
print(f"Held-out images: {len(pred)} | image-level accuracy (identity view): {pred['correct'].mean():.3f}")

MPX, CHP = CLASS_NAMES.index("Monkeypox"), CLASS_NAMES.index("Chickenpox")
sel = []
for c in range(len(CLASS_NAMES)):
    g = pred[pred["true"] == c].sort_values(["conf", "filename"])
    gc, gi = g[g["correct"]], g[~g["correct"]]
    picks = [("correct, highest confidence", gc.tail(1)), ("correct, lowest confidence", gc.head(1)),
             ("incorrect, highest confidence", gi.tail(1)), ("incorrect, lowest confidence", gi.head(1))]
    for why, d in picks:
        for r in d.itertuples():
            sel.append((r.filename, why))
for r in pred[(pred["true"] == MPX) & (pred["pred"] == CHP)].itertuples():
    sel.append((r.filename, "mpox predicted as chickenpox"))
for r in pred[(pred["true"] == CHP) & (pred["pred"] == MPX)].itertuples():
    sel.append((r.filename, "chickenpox predicted as mpox"))
sel = pd.DataFrame(sel, columns=["filename", "reason"]).groupby("filename")["reason"].apply("; ".join).reset_index()
sel = sel.merge(pred, on="filename").sort_values(["true", "correct", "conf"]).reset_index(drop=True)
sel.to_csv(OUT / "selected_images.csv", index=False)
print(f"Selected images: {len(sel)}")
print(sel[["filename", "fold", "true", "pred", "conf", "reason"]].assign(
    true=lambda d: d["true"].map(dict(enumerate(CLASS_NAMES))), pred=lambda d: d["pred"].map(dict(enumerate(CLASS_NAMES)))).to_string(index=False))


# %% CELL 4: Grad-CAM on the output of the last ConvNeXt stage
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.layer = model.stages[-1]                         # last stage: (B, 768, 12, 12) for 384 px input
        self.h = self.layer.register_forward_hook(self._fwd)
    def _fwd(self, mod, inp, out):
        self.act = out
        out.register_hook(lambda g: setattr(self, "grad", g))
    def __call__(self, x, cls):
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            logits[0, cls].backward()
        w = self.grad[0].mean(dim=(1, 2))                     # channel weights = spatially averaged gradients
        cam = F.relu((w[:, None, None] * self.act[0]).sum(0))
        return cam.detach().cpu().numpy(), logits.detach().cpu().numpy()[0]
    def close(self):                                          # break reference cycles so memory is freed
        self.h.remove()
        self.act = self.grad = None
        self.model = self.layer = None

def to_overlay(img, cam_norm, alpha=0.45):
    heat = (cm.jet(cam_norm)[..., :3] * 255).astype(np.uint8)
    return Image.blend(img.convert("RGB"), Image.fromarray(heat), alpha)

index = []
for k in sorted(sel["fold"].unique()):
    cams = [GradCAM(load_model(wp)) for wp in weights[k]]
    assert all(c.layer is c.model.stages[-1] for c in cams), "target layer check failed"
    for r in sel[sel["fold"] == k].itertuples():
        img = Image.open(image_path(r.filename)).convert("RGB")
        x = VAL_TF(img)[None].to(DEVICE)
        targets = [("true", r.true)] + ([("pred", r.pred)] if r.pred != r.true else [])
        for tag, cls in targets:
            raw, logits = zip(*[c(x, cls) for c in cams])
            raw = np.mean(raw, 0)                              # average the two checkpoints' raw maps
            assert np.isfinite(raw).all(), f"non-finite map {r.filename}"
            nonempty = raw.max() > 0
            norm = raw / raw.max() if nonempty else raw
            up = np.array(Image.fromarray(norm.astype(np.float32), mode="F").resize(img.size, Image.BILINEAR))
            up = np.clip(up, 0, 1)
            assert up.shape == (img.size[1], img.size[0]), "overlay size mismatch"
            stem = f"fold{k}_{Path(r.filename).stem}_{tag}_{CLASS_NAMES[cls]}"
            np.save(OUT / f"{stem}_cam_raw.npy", raw)
            to_overlay(img, up).save(OUT / f"{stem}_overlay.png")
            index.append({"filename": r.filename, "fold": k, "explained": tag, "explained_class": CLASS_NAMES[cls],
                          "true": CLASS_NAMES[r.true], "pred": CLASS_NAMES[r.pred], "conf": r.conf, "correct": r.correct,
                          "reason": r.reason, "map_nonempty": bool(nonempty),
                          "logit_of_explained_class": float(np.mean([l[cls] for l in logits])),
                          "checkpoints": ";".join(p.name for p in weights[k]), "overlay": f"{stem}_overlay.png",
                          "cam_raw": f"{stem}_cam_raw.npy"})
        img.save(OUT / f"fold{k}_{Path(r.filename).stem}_original.png")
    for c in cams:
        c.close()
    del cams; gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
index = pd.DataFrame(index); index.to_csv(OUT / "gradcam_index.csv", index=False)
print(f"Maps written: {len(index)} | empty maps: {(~index['map_nonempty']).sum()}")


# %% CELL 5: gallery figure (original | map for true class | map for predicted class if wrong)
rows_ = sel.to_dict("records")
fig, axes = plt.subplots(len(rows_), 3, figsize=(9, 3 * len(rows_)))
axes = np.atleast_2d(axes)
for i, r in enumerate(rows_):
    stem = f"fold{r['fold']}_{Path(r['filename']).stem}"
    axes[i, 0].imshow(Image.open(OUT / f"{stem}_original.png"))
    axes[i, 0].set_title(f"{Path(r['filename']).stem}\ntrue {CLASS_NAMES[r['true']]}, pred {CLASS_NAMES[r['pred']]} ({r['conf']:.2f})", fontsize=8)
    axes[i, 1].imshow(Image.open(OUT / f"{stem}_true_{CLASS_NAMES[r['true']]}_overlay.png"))
    axes[i, 1].set_title(f"Grad-CAM: {CLASS_NAMES[r['true']]}", fontsize=8)
    if r["pred"] != r["true"]:
        axes[i, 2].imshow(Image.open(OUT / f"{stem}_pred_{CLASS_NAMES[r['pred']]}_overlay.png"))
        axes[i, 2].set_title(f"Grad-CAM: {CLASS_NAMES[r['pred']]}", fontsize=8)
    for a in axes[i]:
        a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "gradcam_gallery.png", dpi=150); plt.close()
print("saved", OUT / "gradcam_gallery.png")


# %% CELL 6 (optional): parameter-randomisation sanity check (Adebayo et al., 2018)
from scipy.stats import spearmanr
torch.manual_seed(0)
check = []
for r in sel.head(12).itertuples():
    k = r.fold
    img = Image.open(image_path(r.filename)).convert("RGB"); x = VAL_TF(img)[None].to(DEVICE)
    base = GradCAM(load_model(weights[k][0])); m0, _ = base(x, r.pred); base.close(); del base; gc.collect()
    for level in ("head", "head+last_stage"):
        m = load_model(weights[k][0])
        for mod in ([m.head] if level == "head" else [m.head, m.stages[-1]]):
            for sub in mod.modules():
                if hasattr(sub, "reset_parameters"):
                    sub.reset_parameters()
        g = GradCAM(m); mr, _ = g(x, r.pred); g.close()
        del g, m; gc.collect()
        rho = spearmanr(m0.ravel(), mr.ravel()).correlation if mr.max() > 0 and m0.max() > 0 else float("nan")
        check.append({"filename": r.filename, "randomised": level, "spearman_vs_trained": rho})
check = pd.DataFrame(check); check.to_csv(OUT / "randomisation_check.csv", index=False)
print(check.groupby("randomised")["spearman_vs_trained"].describe().to_string())
print("\nShare: weight_verification.csv, selected_images.csv, gradcam_index.csv, gradcam_gallery.png, randomisation_check.csv")
