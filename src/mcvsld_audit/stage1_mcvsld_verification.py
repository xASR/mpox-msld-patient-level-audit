# %% [markdown]
# # Stage 1: MCVSLD archive verification and image inventory
# CPU only (Accelerator: None). Internet not needed. Inputs are read, never modified.
# Outputs go to /kaggle/working/stage1/. Run the cells in order.

# %% CELL 1: configuration and dependency check
import os, sys, json, hashlib, zipfile, time, re, platform
from pathlib import Path
from collections import Counter, defaultdict

INPUT_ROOT = Path(os.environ.get("STAGE1_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("STAGE1_WORK", "/kaggle/working"))
OUT = WORK / "stage1"
EXTRACT_DIR = WORK / "mcvsld_extracted"          # extraction target, separate from inputs
OUT.mkdir(parents=True, exist_ok=True)

# Published Mendeley counts (MCVSLD v1, DOI 10.17632/dfztdtfsxz.1), used only for comparison
PUBLISHED = {"HFMD": 1932, "Mpox": 3408, "Measles": 660, "Healthy": 1368, "Chickenpox": 900, "Cowpox": 792}
# MSLD v2.0 original images per class (our audit), used only for the ratio column
MSLD_ORIGINALS = {"HFMD": 161, "Mpox": 284, "Measles": 55, "Healthy": 114, "Chickenpox": 75, "Cowpox": 66}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".jfif", ".heic", ".heif"}
SYSTEM_JUNK = {".ds_store", "thumbs.db", "desktop.ini"}

missing = []
for mod in ("numpy", "pandas", "PIL", "tqdm"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    raise SystemExit(f"Missing packages: {missing}. They are preinstalled on Kaggle; check the notebook image.")
import numpy as np, pandas as pd, PIL
from PIL import Image
from tqdm.auto import tqdm
Image.MAX_IMAGE_PIXELS = None   # medical photos can be large; we only read, never trust size limits blindly

print(f"Python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__} | Pillow {PIL.__version__}")
print(f"Input root: {INPUT_ROOT}  (exists: {INPUT_ROOT.exists()})")
print(f"Output dir: {OUT}")


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# %% CELL 2: discover what is mounted under /kaggle/input
def summarise_tree(root, max_depth=4):
    """Print directories (not files) with their direct and total file counts."""
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames.sort()
        if depth > max_depth:
            dirnames[:] = []
            continue
        rows.append((depth, dirpath, len(filenames)))
    totals = Counter()
    for dirpath, _, filenames in os.walk(root):
        p = Path(dirpath)
        for anc in [p, *p.parents]:
            totals[str(anc)] += len(filenames)
            if anc == root:
                break
    for depth, dirpath, n_direct in rows:
        name = Path(dirpath).name if depth else str(root)
        print(f"{'  ' * depth}{name}/  [{n_direct} files here, {totals[dirpath]} below]")

print("=== Directory tree under", INPUT_ROOT, "(depth <= 4) ===")
summarise_tree(INPUT_ROOT)

zips = sorted(p for p in INPUT_ROOT.rglob("*") if p.is_file() and p.suffix.lower() == ".zip")
print("\n=== ZIP files found ===")
for z in zips:
    print(f"{z}  ({z.stat().st_size:,} bytes)")
if not zips:
    print("None. Kaggle usually unpacks a .zip when it is uploaded as a dataset; see Cell 3.")

msld_hits = sorted(p for p in INPUT_ROOT.rglob("*") if p.is_dir() and p.name.lower() == "original images")
print("\n=== Candidate MSLD v2.0 'Original Images' folders (needed in Stage 2) ===")
for p in msld_hits:
    n = sum(1 for q in p.rglob("*") if q.is_file() and q.suffix.lower() in IMG_EXT)
    print(f"{p}  ({n} image files below)")

discovery = {"zips": [str(z) for z in zips], "msld_original_dirs": [str(p) for p in msld_hits]}
(OUT / "discovery.json").write_text(json.dumps(discovery, indent=2))
print(f"\nSaved {OUT / 'discovery.json'}")


# %% CELL 3: verify the ZIP archive (or record that Kaggle extracted it)
# >>> Set this to the MCVSLD zip path printed in Cell 2. If Kaggle already unpacked the
# >>> archive, leave MCVSLD_ZIP = None and set MCVSLD_DIR to the unpacked folder instead.
MCVSLD_ZIP = None   # e.g. Path("/kaggle/input/mcvsld-zip/Skin Lesion Dataset.zip")
MCVSLD_DIR = None   # e.g. Path("/kaggle/input/mcvsld-zip/Skin Lesion Dataset")

if MCVSLD_ZIP is None and MCVSLD_DIR is None and len(zips) == 1:
    MCVSLD_ZIP = zips[0]
    print(f"Only one ZIP is mounted; using it: {MCVSLD_ZIP}")

archive_report = {"mode": None}
if MCVSLD_ZIP is not None:
    MCVSLD_ZIP = Path(MCVSLD_ZIP)
    assert MCVSLD_ZIP.is_file(), f"Not found: {MCVSLD_ZIP}"
    archive_report.update(mode="zip", path=str(MCVSLD_ZIP), size_bytes=MCVSLD_ZIP.stat().st_size)
    print(f"File: {MCVSLD_ZIP.name}\nSize: {archive_report['size_bytes']:,} bytes "
          f"({archive_report['size_bytes'] / 2**20:.1f} MiB)")
    print("Computing SHA-256 of the archive...")
    archive_report["sha256"] = sha256_of(MCVSLD_ZIP)
    print("SHA-256:", archive_report["sha256"])
    archive_report["is_zipfile"] = zipfile.is_zipfile(MCVSLD_ZIP)
    print("Valid ZIP signature:", archive_report["is_zipfile"])
    if not archive_report["is_zipfile"]:
        raise SystemExit("Not a valid ZIP. Re-download the file from Mendeley and compare its size.")

    with zipfile.ZipFile(MCVSLD_ZIP) as zf:
        infos = zf.infolist()
        print(f"Entries in the central directory: {len(infos):,}")
        print("Testing CRC of every entry (reads the whole archive)...")
        first_bad = zf.testzip()                     # None means every CRC matched
        archive_report["testzip_first_bad_entry"] = first_bad
        print("CRC test:", "all entries OK" if first_bad is None else f"FIRST CORRUPT ENTRY: {first_bad}")
        encrypted = [i.filename for i in infos if i.flag_bits & 0x1]
        unsafe = [i.filename for i in infos if i.filename.startswith(("/", "\\")) or ".." in Path(i.filename).parts]
        archive_report.update(n_entries=len(infos), n_dirs=sum(i.is_dir() for i in infos),
                              n_files=sum(not i.is_dir() for i in infos),
                              uncompressed_bytes=sum(i.file_size for i in infos),
                              encrypted_entries=len(encrypted), unsafe_paths=unsafe[:20])
        print(f"Files: {archive_report['n_files']:,} | directories: {archive_report['n_dirs']:,} | "
              f"uncompressed size: {archive_report['uncompressed_bytes'] / 2**20:.1f} MiB | "
              f"encrypted: {len(encrypted)} | unsafe paths: {len(unsafe)}")
        if unsafe:
            raise SystemExit(f"Archive contains unsafe paths, not extracting: {unsafe[:5]}")

        # Directory structure as stored in the archive (folder -> files directly inside)
        per_dir = Counter(str(Path(i.filename).parent) for i in infos if not i.is_dir())
        print("\nFolders inside the archive and their file counts:")
        for d, n in sorted(per_dir.items()):
            print(f"  {d}: {n}")

        # Extract once into /kaggle/working; the input copy stays untouched
        marker = EXTRACT_DIR / ".extracted_from_sha256"
        if marker.exists() and marker.read_text().strip() == archive_report["sha256"]:
            print(f"\nAlready extracted to {EXTRACT_DIR} (same archive hash); skipping extraction.")
        else:
            print(f"\nExtracting to {EXTRACT_DIR} ...")
            EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
            for info in tqdm(infos, desc="extract"):
                zf.extract(info, EXTRACT_DIR)
            marker.write_text(archive_report["sha256"])
    MCVSLD_ROOT = EXTRACT_DIR
else:
    assert MCVSLD_DIR is not None, "Set MCVSLD_ZIP or MCVSLD_DIR above."
    MCVSLD_DIR = Path(MCVSLD_DIR)
    assert MCVSLD_DIR.is_dir(), f"Not a folder: {MCVSLD_DIR}"
    archive_report.update(mode="pre-extracted by Kaggle", path=str(MCVSLD_DIR),
                          note="Archive-level checks (size, SHA-256, CRC) must come from the local copy.")
    print("Using the folder Kaggle unpacked. Record the ZIP's size and SHA-256 from your own computer.")
    MCVSLD_ROOT = MCVSLD_DIR

archive_report["mcvsld_root"] = str(MCVSLD_ROOT)
(OUT / "archive_verification.json").write_text(json.dumps(archive_report, indent=2, default=str))
print(f"\nMCVSLD_ROOT = {MCVSLD_ROOT}\nSaved {OUT / 'archive_verification.json'}")


# %% CELL 4: image inventory (one row per file, resumable)
INV_CSV = OUT / "mcvsld_inventory.csv"
COLUMNS = ["rel_path", "top_folder", "parent_folder", "filename", "ext", "size_bytes", "sha256",
           "is_image_ext", "readable", "pil_format", "width", "height", "mode", "n_frames", "error"]

all_files = sorted(p for p in MCVSLD_ROOT.rglob("*") if p.is_file() and p.name != ".extracted_from_sha256")
print(f"Files under {MCVSLD_ROOT}: {len(all_files):,}")

done = set()
if INV_CSV.exists():
    done = set(pd.read_csv(INV_CSV, usecols=["rel_path"])["rel_path"])
    print(f"Resuming: {len(done):,} files already in {INV_CSV.name}")

def inspect(path):
    rel = path.relative_to(MCVSLD_ROOT)
    parts = rel.parts
    row = {"rel_path": str(rel), "top_folder": parts[0] if len(parts) > 1 else "",
           "parent_folder": parts[-2] if len(parts) > 1 else "", "filename": path.name,
           "ext": path.suffix.lower(), "size_bytes": path.stat().st_size, "sha256": sha256_of(path),
           "is_image_ext": path.suffix.lower() in IMG_EXT, "readable": False, "pil_format": "",
           "width": np.nan, "height": np.nan, "mode": "", "n_frames": np.nan, "error": ""}
    if path.name.lower() in SYSTEM_JUNK or "__MACOSX" in parts or path.name.startswith("._"):
        row["error"] = "system file"
        return row
    try:
        with Image.open(path) as im:
            im.verify()                       # structural check
        with Image.open(path) as im:          # verify() invalidates the object; reopen to decode
            im.load()                         # full decode catches truncated data
            row.update(readable=True, pil_format=im.format or "", width=im.size[0], height=im.size[1],
                       mode=im.mode, n_frames=getattr(im, "n_frames", 1))
    except Exception as e:                    # unreadable, truncated or not an image
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    return row

todo = [p for p in all_files if str(p.relative_to(MCVSLD_ROOT)) not in done]
buffer, write_header = [], not INV_CSV.exists()
for i, p in enumerate(tqdm(todo, desc="inventory")):
    buffer.append(inspect(p))
    if len(buffer) >= 500 or i == len(todo) - 1:
        pd.DataFrame(buffer, columns=COLUMNS).to_csv(INV_CSV, mode="a", header=write_header, index=False)
        write_header, buffer = False, []

inv = pd.read_csv(INV_CSV, keep_default_na=False, na_values=[""])
inv = inv.drop_duplicates("rel_path", keep="last")
print(f"Inventory rows: {len(inv):,}  ->  {INV_CSV}")


# %% CELL 5: reports from the inventory (no class mapping is assumed here)
img = inv[inv["is_image_ext"].astype(str).str.lower() == "true"]
ok = img[img["readable"].astype(str).str.lower() == "true"]

print("=== Extension distribution (all files) ===")
ext_tab = inv["ext"].replace("", "(none)").value_counts().rename_axis("ext").reset_index(name="files")
print(ext_tab.to_string(index=False))
ext_tab.to_csv(OUT / "extension_distribution.csv", index=False)

print("\n=== Files per folder (parent folder of each file) ===")
folder_tab = (inv.groupby(["top_folder", "parent_folder"])
                 .agg(files=("rel_path", "size"),
                      image_ext=("is_image_ext", lambda s: (s.astype(str).str.lower() == "true").sum()),
                      readable=("readable", lambda s: (s.astype(str).str.lower() == "true").sum()))
                 .reset_index())
print(folder_tab.to_string(index=False))
folder_tab.to_csv(OUT / "directory_structure.csv", index=False)

print("\n=== Non-image or unreadable files ===")
bad = inv[(inv["readable"].astype(str).str.lower() != "true")]
print(f"{len(bad)} files")
print(bad[["rel_path", "ext", "size_bytes", "error"]].head(50).to_string(index=False))
bad.to_csv(OUT / "unreadable_or_non_image.csv", index=False)

# Format and extension disagreements (e.g. a PNG saved as .jpg); informative, not an error
ext_to_fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".jfif": "JPEG", ".png": "PNG", ".bmp": "BMP",
              ".gif": "GIF", ".tif": "TIFF", ".tiff": "TIFF", ".webp": "WEBP"}
mism = ok[ok.apply(lambda r: ext_to_fmt.get(r["ext"], r["pil_format"]) != r["pil_format"], axis=1)]
print(f"\nReadable images whose extension does not match the decoded format: {len(mism)}")
mism.to_csv(OUT / "extension_format_mismatch.csv", index=False)

print("\n=== Image size summary (readable images) ===")
print(ok[["width", "height"]].astype(float).describe().round(1).to_string())
print("Colour modes:", dict(ok["mode"].value_counts()))

print("\n=== Duplicate file names ===")
name_counts = ok.groupby("filename")["parent_folder"].agg(["count", lambda s: sorted(set(s))])
name_counts.columns = ["copies", "folders"]
dup_names = name_counts[name_counts["copies"] > 1].sort_values("copies", ascending=False)
cross = dup_names[dup_names["folders"].apply(len) > 1]
print(f"Names used more than once: {len(dup_names)} | of those, in more than one folder: {len(cross)}")
print(dup_names.head(20).to_string())
dup_names.to_csv(OUT / "duplicate_filenames.csv")

print("\n=== Byte-identical files inside MCVSLD (preview; full audit in Stage 2) ===")
grp = ok.groupby("sha256")
dup_hash = grp.filter(lambda g: len(g) > 1)
print(f"Readable images: {len(ok):,} | distinct SHA-256: {ok['sha256'].nunique():,} | "
      f"images in a byte-identical group: {len(dup_hash):,}")
dup_hash.sort_values(["sha256", "rel_path"]).to_csv(OUT / "exact_internal_duplicates_preview.csv", index=False)

print("\n=== Folder names, to be mapped to classes by hand (not alphabetically) ===")
for name, n in ok["parent_folder"].value_counts().items():
    print(f"  {name!r}: {n} readable images")

# Provisional comparison: keyword match on folder names, for display only.
# The real mapping is fixed after you confirm the folder names above.
KEYWORDS = {"HFMD": ["hfmd", "hand", "foot"], "Mpox": ["mpox", "monkey"], "Measles": ["measle"],
            "Healthy": ["healthy", "normal"], "Chickenpox": ["chicken"], "Cowpox": ["cowpox", "cow"]}
def guess(folder):
    f = folder.lower()
    hits = [c for c, kws in KEYWORDS.items() if any(k in f for k in kws)]
    return hits[0] if len(hits) == 1 else ("AMBIGUOUS" if hits else "UNMATCHED")
counts = ok.assign(guess=ok["parent_folder"].map(guess)).groupby("guess").size()
cmp_rows = []
for c in PUBLISHED:
    actual = int(counts.get(c, 0))
    cmp_rows.append({"class (provisional)": c, "published": PUBLISHED[c], "readable_found": actual,
                     "difference": actual - PUBLISHED[c], "msld_v2_originals": MSLD_ORIGINALS[c],
                     "found / msld_originals": round(actual / MSLD_ORIGINALS[c], 3)})
cmp = pd.DataFrame(cmp_rows)
for extra in ("AMBIGUOUS", "UNMATCHED"):
    if extra in counts:
        print(f"WARNING: {counts[extra]} readable images in folders matched as {extra}")
print("\n=== Class counts against the Mendeley record (PROVISIONAL mapping) ===")
print(cmp.to_string(index=False))
print(f"Total readable images: {len(ok):,} (published total 9,060)")
cmp.to_csv(OUT / "class_counts_vs_published_PROVISIONAL.csv", index=False)

summary = {"files_total": int(len(inv)), "image_ext_files": int(len(img)), "readable_images": int(len(ok)),
           "unreadable_or_non_image": int(len(bad)), "distinct_sha256": int(ok["sha256"].nunique()),
           "duplicate_filenames": int(len(dup_names)), "ext_format_mismatch": int(len(mism))}
(OUT / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
print("\nSummary:", json.dumps(summary))
print("\nShare with Claude: the printed output of Cells 2 to 5, and the files in", OUT)
