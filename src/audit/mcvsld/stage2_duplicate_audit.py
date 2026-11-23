# %% [markdown]
# # Stage 2: exact and perceptual duplicate audit, MCVSLD against MSLD v2.0
# CPU only (Accelerator: None). Inputs: the MCVSLD copy and MSLD v2.0, both already attached.
# Needs /kaggle/working/stage1/mcvsld_inventory.csv from Stage 1 (same notebook session or re-run Stage 1).
# Nothing is excluded automatically here: this stage measures, and the exclusion manifest is fixed after review.

# %% CELL 1: configuration, paths, helpers
import os, re, json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from PIL import Image
from tqdm.auto import tqdm
try:
    from scipy.fft import dct
except ImportError:
    from scipy.fftpack import dct

INPUT_ROOT = Path(os.environ.get("STAGE2_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("STAGE2_WORK", "/kaggle/working"))
S1, OUT = WORK / "stage1", WORK / "stage2"
OUT.mkdir(parents=True, exist_ok=True)
Image.MAX_IMAGE_PIXELS = None

if (S1 / "archive_verification.json").exists():
    MCVSLD_ROOT = Path(json.loads((S1 / "archive_verification.json").read_text())["mcvsld_root"])
else:   # Stage 1 outputs not in this session: find the MCVSLD folder again
    hits = [p for p in INPUT_ROOT.rglob("*") if p.is_dir() and re.search(r"viral|mcvsld", p.name, re.I)]
    assert hits, "MCVSLD folder not found under /kaggle/input"
    MCVSLD_ROOT = min(hits, key=lambda p: len(p.parts))
msld_hits = sorted((p for p in INPUT_ROOT.rglob("*") if p.is_dir() and p.name.lower() == "original images"),
                   key=lambda p: len(p.parts))
assert msld_hits, "MSLD v2.0 'Original Images' folder not found; attach the joydippaul MSLD v2.0 dataset."
MSLD_ROOT = msld_hits[0].parent                     # dataset root holding Original Images / Augmented Images
MSLD_ORIG = msld_hits[0]
aug_hits = sorted((p for p in MSLD_ROOT.rglob("*") if p.is_dir() and p.name.lower() == "augmented images"),
                  key=lambda p: len(p.parts))
MSLD_AUG = aug_hits[0] if aug_hits else None
print("MCVSLD root  :", MCVSLD_ROOT)
print("MSLD originals:", MSLD_ORIG)
print("MSLD augmented:", MSLD_AUG)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_DIRS = {"MONKEYPOX": "Mpox", "MPOX": "Mpox", "CHICKENPOX": "Chickenpox", "MEASLES": "Measles",
              "COWPOX": "Cowpox", "HFMD": "HFMD", "HEALTHY": "Healthy", "NORMAL": "Healthy"}
# MCVSLD folder name -> manuscript class. Confirmed in Stage 1 from the six folder names.
MCVSLD_MAP = {"Monkeypox": "Mpox", "Chickenpox": "Chickenpox", "Measles": "Measles",
              "Cowpox": "Cowpox", "HFMD": "HFMD", "Healthy": "Healthy"}
NAME_RE = re.compile(r"^(MKP|HFMD|HEALTHY|CHP|CWP|MSL)_(\d+)_(\d+)(?:_(ORIGINAL|\d+))?$", re.I)
THRESH = 10                                         # pHash screening threshold (Hamming distance)


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def parse_name(stem):
    m = NAME_RE.match(stem)
    if not m:
        return None, None, "unparsed"
    pre, pnum, ino, suf = m.groups()
    photo = f"{pre.upper()}_{int(pnum):02d}_{int(ino):02d}"   # one key per source photograph
    kind = "plain" if suf is None else ("ORIGINAL" if suf.upper() == "ORIGINAL" else "variant")
    return photo, f"{pre.upper()}_{int(pnum)}", kind


def class_from_path(p, stop):
    for parent in p.parents:
        if parent == stop:
            break
        c = CLASS_DIRS.get(parent.name.strip().upper())
        if c:
            return c
    return None


DIHEDRAL = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
            Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
            Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE]


def phash64(img):
    """64-bit DCT perceptual hash (same recipe as imagehash.phash: 32x32 grey, 8x8 low band, median)."""
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    low = dct(dct(g, axis=0), axis=1)[:8, :8]
    bits = (low > np.median(low)).flatten()
    return np.packbits(bits).view(">u8")[0].astype(np.uint64)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(a, b):
    """Pairwise Hamming distances between uint64 arrays a (n,) and b (m,) -> (n, m) uint8."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return _POP[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1, dtype=np.uint8)


def hash_table(paths, root, cache_csv, dihedral=False, label_fn=None):
    """SHA-256 + pHash (and optionally the 8 rotation/flip pHashes) per file; resumable via CSV."""
    cols = ["path", "sha256", "phash"] + ([f"ph_d{i}" for i in range(1, 8)] if dihedral else [])
    done = pd.read_csv(cache_csv, dtype=str) if cache_csv.exists() else pd.DataFrame(columns=cols)
    have = set(done["path"])
    rows = []
    for p in tqdm([p for p in paths if str(p) not in have], desc=cache_csv.stem):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                r = {"path": str(p), "sha256": sha256_of(p), "phash": str(int(phash64(im)))}
                if dihedral:
                    for i, t in enumerate(DIHEDRAL[1:], 1):
                        r[f"ph_d{i}"] = str(int(phash64(im.transpose(t))))
        except Exception as e:
            r = {"path": str(p), "sha256": "", "phash": "", "error": f"{type(e).__name__}: {e}"}
        rows.append(r)
        if len(rows) >= 1000:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); done.to_csv(cache_csv, index=False); rows = []
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); done.to_csv(cache_csv, index=False)
    return done

print("Helpers ready.")


# %% CELL 2: MSLD v2.0 original images, de-duplicated to one row per photograph
orig_paths = sorted(p for p in MSLD_ORIG.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
print(f"Image files under MSLD 'Original Images': {len(orig_paths):,} (this includes the fold copies)")
ho = hash_table(orig_paths, MSLD_ORIG, OUT / "msld_original_hashes.csv", dihedral=True)
ho = ho[ho["sha256"].fillna("") != ""].copy()
ho["file"] = ho["path"].map(lambda s: Path(s).name)
parsed = ho["file"].map(lambda f: parse_name(Path(f).stem))
ho["photo"], ho["pid"], ho["kind"] = zip(*parsed)
ho["cls"] = ho["path"].map(lambda s: class_from_path(Path(s), MSLD_ORIG))
print("Unparsed MSLD names:", (ho["photo"].isna()).sum())

# One row per distinct photograph name; check that all copies of a name are byte-identical
per_name = ho.groupby("photo")["sha256"].nunique()
print(f"Distinct MSLD photograph names: {per_name.size} | names whose copies differ in bytes: {(per_name > 1).sum()}")
msld = ho.sort_values("path").drop_duplicates("photo").reset_index(drop=True)
print(f"Distinct MSLD photographs: {len(msld)} | distinct SHA-256 among them: {msld['sha256'].nunique()} | "
      f"patients: {msld['pid'].nunique()}")
print(msld.groupby("cls").agg(images=("photo", "size"), patients=("pid", "nunique")).to_string())
same_bytes = msld[msld.duplicated("sha256", keep=False)].sort_values("sha256")
print("\nDifferent MSLD photograph names with byte-identical files:")
print(same_bytes[["photo", "cls", "sha256"]].to_string(index=False) if len(same_bytes) else "none")
msld.to_csv(OUT / "msld_originals_unique.csv", index=False)


# %% CELL 3: MSLD v2.0 released augmented images (explains MCVSLD's numbered variants)
if MSLD_AUG is not None:
    aug_paths = sorted(p for p in MSLD_AUG.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
    print(f"Image files under MSLD 'Augmented Images': {len(aug_paths):,}")
    # SHA only (pHash of 37k augmented files is not needed for the decision)
    cache = OUT / "msld_augmented_sha.csv"
    done = pd.read_csv(cache, dtype=str) if cache.exists() else pd.DataFrame(columns=["path", "sha256"])
    have, rows = set(done["path"]), []
    for p in tqdm([p for p in aug_paths if str(p) not in have], desc="msld_aug_sha"):
        rows.append({"path": str(p), "sha256": sha256_of(p)})
        if len(rows) >= 2000:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); done.to_csv(cache, index=False); rows = []
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); done.to_csv(cache, index=False)
    maug = done.copy()
    maug["file"] = maug["path"].map(lambda s: Path(s).name)
    print(f"Distinct augmented file names: {maug['file'].nunique():,} | distinct SHA-256: {maug['sha256'].nunique():,}")
else:
    maug = pd.DataFrame(columns=["path", "sha256", "file"])
    print("No 'Augmented Images' folder found; skipping.")


# %% CELL 4: hash every MCVSLD image (SHA from Stage 1, pHash computed here)
if (S1 / "mcvsld_inventory.csv").exists():
    inv = pd.read_csv(S1 / "mcvsld_inventory.csv", dtype=str)
    inv = inv[inv["readable"].str.lower() == "true"].copy()
    inv["path"] = inv["rel_path"].map(lambda r: str(MCVSLD_ROOT / r))
    paths = [Path(p) for p in inv["path"]]
else:   # rebuild the minimum needed from the files themselves
    paths = sorted(p for p in MCVSLD_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
    inv = pd.DataFrame({"path": [str(p) for p in paths],
                        "rel_path": [str(p.relative_to(MCVSLD_ROOT)) for p in paths],
                        "parent_folder": [p.parent.name for p in paths], "filename": [p.name for p in paths]})
hm = hash_table(paths, MCVSLD_ROOT, OUT / "mcvsld_phash.csv")
hm = hm[hm["sha256"].fillna("") != ""]
mc = inv.drop(columns=["sha256"], errors="ignore").merge(hm[["path", "sha256", "phash"]], on="path", how="inner")
mc["split"] = mc["rel_path"].map(lambda r: Path(r).parts[1] if len(Path(r).parts) > 3 else "")
mc["cls"] = mc["parent_folder"].map(MCVSLD_MAP)
parsed = mc["filename"].map(lambda f: parse_name(Path(f).stem))
mc["photo"], mc["pid"], mc["kind"] = zip(*parsed)
print(f"MCVSLD images: {len(mc):,} | unmapped class folders: {mc['cls'].isna().sum()} | "
      f"names not in MSLD format: {(mc['kind'] == 'unparsed').sum()}")
print(pd.crosstab(mc["cls"], mc["kind"]).to_string())


# %% CELL 5: exact (byte-identical) matches against MSLD v2.0
orig_sha = msld.groupby("sha256")["photo"].apply(lambda s: ";".join(sorted(s))).to_dict()
orig_cls = msld.groupby("sha256")["cls"].first().to_dict()
aug_sha = maug.groupby("sha256")["file"].apply(lambda s: ";".join(sorted(set(s)))[:200]).to_dict()
mc["exact_msld_original"] = mc["sha256"].map(orig_sha)
mc["exact_msld_original_cls"] = mc["sha256"].map(orig_cls)
mc["exact_msld_augmented"] = mc["sha256"].map(aug_sha)
mc["exact_any"] = mc["exact_msld_original"].notna() | mc["exact_msld_augmented"].notna()

exact_tab = mc[mc["exact_any"]][["rel_path", "cls", "sha256", "exact_msld_original", "exact_msld_original_cls",
                                 "exact_msld_augmented"]]
exact_tab.to_csv(OUT / "cross_dataset_exact_matches.csv", index=False)
print("Byte-identical to an MSLD v2.0 ORIGINAL image:", mc["exact_msld_original"].notna().sum())
print("Byte-identical to an MSLD v2.0 AUGMENTED image:", mc["exact_msld_augmented"].notna().sum())
print("Byte-identical to either:", mc["exact_any"].sum(), "of", len(mc))
print(pd.crosstab(mc["kind"], mc["exact_any"], margins=True).to_string())
lab_ok = mc.dropna(subset=["exact_msld_original_cls"])
print("Exact original matches with a DIFFERENT class label:", (lab_ok["cls"] != lab_ok["exact_msld_original_cls"]).sum())


# %% CELL 6: perceptual matches (pHash, with rotations/flips of the MSLD originals)
mc_ph = mc["phash"].astype("uint64").to_numpy()
ref = np.stack([msld["phash"].astype("uint64").to_numpy()] +
               [msld[f"ph_d{i}"].astype("uint64").to_numpy() for i in range(1, 8)], axis=1)   # (755, 8)
best_d = np.full(len(mc), 255, np.uint8); best_j = np.zeros(len(mc), int); best_t = np.zeros(len(mc), int)
for s in tqdm(range(0, len(mc), 500), desc="pHash vs MSLD"):
    d = hamming(mc_ph[s:s + 500], ref.reshape(-1))                    # (chunk, 755*8)
    d = d.reshape(d.shape[0], len(msld), 8)
    dmin_t = d.min(-1); arg = dmin_t.argmin(1)
    best_d[s:s + 500] = dmin_t[np.arange(len(arg)), arg]; best_j[s:s + 500] = arg
    best_t[s:s + 500] = d[np.arange(len(arg)), arg].argmin(-1)
mc["nn_msld_photo"] = msld["photo"].to_numpy()[best_j]
mc["nn_msld_cls"] = msld["cls"].to_numpy()[best_j]
mc["nn_distance"] = best_d
mc["nn_transform"] = best_t                                           # 0 = as is, 1-7 = rotation/flip
mc["nn_is_own_name"] = mc["nn_msld_photo"] == mc["photo"]

# Distance to the MSLD original that carries the SAME photograph name
idx = {p: i for i, p in enumerate(msld["photo"])}
own = np.array([idx.get(p, -1) for p in mc["photo"]])
has = own >= 0
d_own = np.full(len(mc), 255, np.uint8)
d_own[has] = np.array([hamming(np.array([mc_ph[k]]), ref[own[k]])[0].min() for k in np.where(has)[0]])
mc["own_name_distance"] = np.where(has, d_own, np.nan)

print(f"MCVSLD images whose nearest MSLD original is within {THRESH} bits: {(mc['nn_distance'] <= THRESH).sum():,}")
print(f"  ... and that nearest original has the same photograph name: {((mc['nn_distance'] <= THRESH) & mc['nn_is_own_name']).sum():,}")
print(f"Images with NO MSLD original within {THRESH} bits: {(mc['nn_distance'] > THRESH).sum():,}")
print("\nNearest-original distance by MCVSLD file kind (median, 90th percentile, share <= 10):")
print(mc.groupby("kind")["nn_distance"].agg(median="median", p90=lambda s: np.percentile(s, 90),
                                            share_le10=lambda s: round((s <= THRESH).mean(), 4)).to_string())
print("\nDistance histogram, all MCVSLD images:")
print(pd.cut(mc["nn_distance"].astype(int), [-1, 0, 2, 5, 10, 15, 20, 64]).value_counts().sort_index().to_string())
mc.to_csv(OUT / "mcvsld_vs_msld_matches.csv", index=False)


# %% CELL 7: near-duplicates inside MCVSLD between DIFFERENT photograph names
pairs = []
for s in tqdm(range(0, len(mc), 400), desc="MCVSLD internal"):
    d = hamming(mc_ph[s:s + 400], mc_ph)
    ii, jj = np.where(d <= THRESH)
    for i, j in zip(ii + s, jj):
        if i < j and mc["photo"].iat[i] != mc["photo"].iat[j]:
            pairs.append((mc["rel_path"].iat[i], mc["rel_path"].iat[j], int(d[i - s, j]),
                          mc["cls"].iat[i], mc["cls"].iat[j], mc["sha256"].iat[i] == mc["sha256"].iat[j]))
internal = pd.DataFrame(pairs, columns=["a", "b", "distance", "cls_a", "cls_b", "byte_identical"]).sort_values("distance")
internal.to_csv(OUT / "mcvsld_internal_cross_photo_candidates.csv", index=False)
print(f"Cross-photograph pairs within {THRESH} bits: {len(internal):,} | byte-identical: {internal['byte_identical'].sum()} | "
      f"different class labels: {(internal['cls_a'] != internal['cls_b']).sum()}")
print(internal.head(20).to_string(index=False))


# %% CELL 8: contact sheets for visual verification (fixed random seed, not hand-picked)
from PIL import ImageDraw
rng = np.random.default_rng(2026)
msld_path = dict(zip(msld["photo"], msld["path"]))

def sheet(pairs, out_png, title, tile=180):
    """pairs: list of (left_path, right_path, caption). Two images side by side per row."""
    if not pairs:
        print("nothing to draw for", out_png.name); return
    cols, W = 3, tile * 2 + 12
    rows = (len(pairs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * W, rows * (tile + 30) + 30), "white")
    dr = ImageDraw.Draw(canvas); dr.text((6, 6), title, fill="black")
    for k, (a, b, cap) in enumerate(pairs):
        x, y = (k % cols) * W, 30 + (k // cols) * (tile + 30)
        for off, p in ((0, a), (tile + 4, b)):
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB"); im.thumbnail((tile, tile))
                    canvas.paste(im, (x + off, y))
            except Exception:
                dr.text((x + off + 5, y + 5), "unreadable", fill="red")
        dr.text((x + 2, y + tile + 2), cap[:60], fill="black")
    canvas.save(out_png); print("saved", out_png)

# A: MCVSLD unsuffixed files vs the MSLD original of the same name (random 24)
plain = mc[mc["kind"] == "plain"].sample(min(24, (mc["kind"] == "plain").sum()), random_state=2026)
sheet([(r.path, msld_path.get(r.photo, ""), f"{r.photo} d={r.own_name_distance:.0f}") for r in plain.itertuples()],
      OUT / "sheet_A_plain_vs_msld.png", "A. MCVSLD (left) vs MSLD original with the same name (right)")
# B: numbered variants vs their MSLD original (random 24)
var = mc[mc["kind"] == "variant"].sample(min(24, (mc["kind"] == "variant").sum()), random_state=2026)
sheet([(r.path, msld_path.get(r.photo, ""), f"{Path(r.rel_path).name} d={r.own_name_distance:.0f}") for r in var.itertuples()],
      OUT / "sheet_B_variant_vs_msld.png", "B. MCVSLD numbered variant (left) vs MSLD original (right)")
# C: every MCVSLD image farther than the threshold from all MSLD originals (random 36 of them)
far = mc[mc["nn_distance"] > THRESH]
far = far.sample(min(36, len(far)), random_state=2026)
sheet([(r.path, msld_path.get(r.photo, msld_path.get(r.nn_msld_photo, "")), f"{Path(r.rel_path).name} nn={r.nn_distance}")
       for r in far.itertuples()], OUT / "sheet_C_no_close_msld_match.png",
      f"C. MCVSLD images with no MSLD original within {THRESH} bits (left) vs MSLD photo of the same name (right)")
# D: closest cross-photograph pairs inside MCVSLD
sheet([(str(MCVSLD_ROOT / r.a), str(MCVSLD_ROOT / r.b), f"d={r.distance} {r.cls_a}/{r.cls_b}") for r in internal.head(24).itertuples()],
      OUT / "sheet_D_internal_cross_photo.png", "D. Closest pairs of different photograph names inside MCVSLD")


# %% CELL 9: audit summary (numbers only; categories needing visual review are marked as candidates)
summary = {
    "mcvsld_source_used": str(MCVSLD_ROOT),
    "published_mcvsld_images_mendeley": 9060,
    "images_present_in_this_copy": int(len(inv)),
    "unreadable_images": int(len(inv) - len(mc)),
    "distinct_photograph_names_mcvsld": int(mc["photo"].nunique()),
    "distinct_patient_ids_mcvsld": int(mc["pid"].nunique()),
    "msld_original_photographs": int(len(msld)),
    "mcvsld_names_also_in_msld": int(mc["photo"].isin(set(msld["photo"])).sum()),
    "exact_internal_duplicate_images_mcvsld": int(mc.duplicated("sha256", keep=False).sum()),
    "exact_internal_duplicate_groups_mcvsld": int(mc[mc.duplicated("sha256", keep=False)]["sha256"].nunique()),
    "cross_dataset_exact_to_msld_original": int(mc["exact_msld_original"].notna().sum()),
    "cross_dataset_exact_to_msld_augmented": int(mc["exact_msld_augmented"].notna().sum()),
    "cross_dataset_exact_either": int(mc["exact_any"].sum()),
    "phash_le10_to_some_msld_original": int((mc["nn_distance"] <= THRESH).sum()),
    "phash_le10_and_same_name": int(((mc["nn_distance"] <= THRESH) & mc["nn_is_own_name"]).sum()),
    "no_msld_original_within_10_bits": int((mc["nn_distance"] > THRESH).sum()),
    "no_exact_and_no_phash_match": int(((~mc["exact_any"]) & (mc["nn_distance"] > THRESH)).sum()),
    "internal_cross_photo_candidate_pairs": int(len(internal)),
}
(OUT / "stage2_summary.json").write_text(json.dumps(summary, indent=2))
for k, v in summary.items():
    print(f"{k:45s} {v}")
print("\nShare with Claude: the printed output of Cells 2 to 9, stage2_summary.json, and the four sheet_*.png images.")
