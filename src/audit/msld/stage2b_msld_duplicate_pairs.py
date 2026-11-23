# %% [markdown]
# # Stage 2b: same photograph under two patient IDs inside MSLD v2.0
# CPU only. Needs the MSLD v2.0 dataset attached and /kaggle/working/stage2/msld_originals_unique.csv
# (from Stage 2 in this session; if it is missing, re-run Stage 2 Cells 1 and 2 first).
# 1) draws every pair of different MSLD photographs whose perceptual hashes are within 12 bits,
#    so each pair can be judged by eye;
# 2) looks up where each pair sits in the five official datalog sheets (train / val / test).

# %% CELL 1: candidate pairs and contact sheet
import os, re, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

WORK = Path(os.environ.get("STAGE2_WORK", "/kaggle/working"))
INPUT_ROOT = Path(os.environ.get("STAGE2_INPUT_ROOT", "/kaggle/input"))
S2, OUT = WORK / "stage2", WORK / "stage2b"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = S2 / "msld_originals_unique.csv"
if not CACHE.exists():
    CACHE = OUT / "msld_originals_unique.csv"
if not CACHE.exists():
    # Stage 2 outputs are not in this session: hash the MSLD originals again (about 1 to 2 minutes)
    import hashlib
    from tqdm.auto import tqdm
    try:
        from scipy.fft import dct
    except ImportError:
        from scipy.fftpack import dct
    Image.MAX_IMAGE_PIXELS = None
    orig_dir = min((p for p in INPUT_ROOT.rglob("*") if p.is_dir() and p.name.lower() == "original images"),
                   key=lambda p: len(p.parts))
    CLASS_DIRS = {"MONKEYPOX": "Mpox", "CHICKENPOX": "Chickenpox", "MEASLES": "Measles",
                  "COWPOX": "Cowpox", "HFMD": "HFMD", "HEALTHY": "Healthy"}
    TRANS = [Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
             Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
             Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE]
    def phash64(img):
        g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
        low = dct(dct(g, axis=0), axis=1)[:8, :8]
        return int(np.packbits((low > np.median(low)).flatten()).view(">u8")[0])
    rows = []
    for p in tqdm(sorted(q for q in orig_dir.rglob("*") if q.suffix.lower() in {".jpg", ".jpeg", ".png"}), desc="MSLD originals"):
        mm = re.match(r"^([A-Za-z]+)_(\d+)_(\d+)", p.stem)
        if not mm:
            continue
        photo = f"{mm.group(1).upper()}_{int(mm.group(2)):02d}_{int(mm.group(3)):02d}"
        cls = next((CLASS_DIRS[q.name.upper()] for q in p.parents if q.name.upper() in CLASS_DIRS), None)
        with Image.open(p) as im:
            im = im.convert("RGB")
            r = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "phash": str(phash64(im)),
                 "photo": photo, "pid": f"{mm.group(1).upper()}_{int(mm.group(2))}", "cls": cls}
            for i, t in enumerate(TRANS, 1):
                r[f"ph_d{i}"] = str(phash64(im.transpose(t)))
        rows.append(r)
    pd.DataFrame(rows).sort_values("path").drop_duplicates("photo").to_csv(CACHE, index=False)
m = pd.read_csv(CACHE, dtype=str)
print(f"MSLD photographs: {len(m)} (from {CACHE})")
assert len(m) == 755, f"expected 755 MSLD photographs, found {len(m)}"

cols = ["phash"] + [f"ph_d{i}" for i in range(1, 8)]
P = np.array([[int(v) for v in m[c]] for c in cols], dtype=np.uint64).T           # (755, 8)
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
def ham(a, b):
    x = np.bitwise_xor(a[:, None], b[None, :])
    return POP[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1)
D = np.stack([ham(P[:, 0], P[:, t]) for t in range(8)]).min(0)                     # any rotation/flip of j
D = np.minimum(D, D.T)
ii, jj = np.where(np.triu(D <= 12, k=1))
pairs = pd.DataFrame({"a": m["photo"].to_numpy()[ii], "b": m["photo"].to_numpy()[jj], "distance": D[ii, jj],
                      "cls_a": m["cls"].to_numpy()[ii], "cls_b": m["cls"].to_numpy()[jj],
                      "same_patient_id": m["pid"].to_numpy()[ii] == m["pid"].to_numpy()[jj],
                      "byte_identical": m["sha256"].to_numpy()[ii] == m["sha256"].to_numpy()[jj]})
pairs = pairs.sort_values(["distance", "a"]).reset_index(drop=True)
pairs.index = pairs.index + 1
pairs.to_csv(OUT / "msld_cross_photo_pairs.csv", index_label="pair_no")
print(f"Pairs of different MSLD photographs within 12 bits: {len(pairs)}")
print(pairs.to_string())

path = dict(zip(m["photo"], m["path"]))
tile, per_row = 200, 3
W, H = tile * 2 + 16, tile + 34
canvas = Image.new("RGB", (per_row * W, ((len(pairs) + per_row - 1) // per_row) * H + 30), "white")
dr = ImageDraw.Draw(canvas)
dr.text((6, 8), "MSLD v2.0 originals: pairs of different photograph names within 12 pHash bits", fill="black")
for k, r in enumerate(pairs.itertuples()):
    x, y = (k % per_row) * W, 30 + (k // per_row) * H
    for off, ph in ((0, r.a), (tile + 4, r.b)):
        with Image.open(path[ph]) as im:
            im = im.convert("RGB"); im.thumbnail((tile, tile)); canvas.paste(im, (x + off, y))
    dr.text((x + 2, y + tile + 2), f"#{r.Index} {r.a} | {r.b} d={r.distance}", fill="black")
canvas.save(OUT / "sheet_E_msld_cross_photo_pairs.png")
print("saved", OUT / "sheet_E_msld_cross_photo_pairs.png")


# %% CELL 2: where each pair sits in the official datalog folds
xlsx = next(iter(INPUT_ROOT.rglob("datalog.xlsx")), None)
assert xlsx is not None, "datalog.xlsx not found; attach MSLD v2.0"
sheets = pd.read_excel(xlsx, sheet_name=None, engine="openpyxl")
def key(x):                                              # normalise MKP_4_1 / MKP_04_01.jpg to MKP_04_01
    m_ = re.match(r"^([A-Za-z]+)_(\d+)_(\d+)", Path(str(x)).stem.strip())
    return f"{m_.group(1).upper()}_{int(m_.group(2)):02d}_{int(m_.group(3)):02d}" if m_ else None
where = {}
for name, df in sheets.items():
    part = {}
    for split in ("train", "val", "test"):
        col = next((c for c in df.columns if split in str(c).lower()), None)
        if col is not None:
            for v in df[col].dropna():
                if key(v):
                    part[key(v)] = split
    where[name] = part
print("Sheets:", list(where), "| photographs per sheet:", [len(v) for v in where.values()])

rows = []
for r in pairs.itertuples():
    row = {"pair_no": r.Index, "a": r.a, "b": r.b, "distance": r.distance}
    flags = []
    for name, part in where.items():
        pa, pb = part.get(r.a, "-"), part.get(r.b, "-")
        row[name] = f"{pa}/{pb}"
        if pa != pb and "test" in (pa, pb) and "-" not in (pa, pb):
            flags.append(name)
    row["sheets_with_one_in_test_other_not"] = ";".join(flags)
    rows.append(row)
loc = pd.DataFrame(rows)
loc.to_csv(OUT / "msld_cross_photo_pairs_in_datalog.csv", index=False)
print(loc.to_string(index=False))
print("\nShare with Claude: the printed output of both cells and sheet_E_msld_cross_photo_pairs.png")
