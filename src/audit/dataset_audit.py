"""Patient and fold audit of MSLD v2.0 (paper Section 3.2, Tables 2 and 3).

Answers four questions from the files alone, without training anything:

1. How many original images and distinct patients does each class have?
   (Compared with the patient counts reported by Ali et al., 2024.)
2. Are there image files that sit outside any class folder?
3. Do the five sheets of datalog.xlsx keep patients apart inside each fold,
   and how much of the dataset ends up in a test partition across the five?
4. How large is the released augmented pool, and are the same augmented file
   names repeated across fold folders?

Patient IDs come from the file names, e.g. MKP_17_01.jpg -> MKP_17
(class prefix + patient number; the last number is the image index).
A strict pattern is tried first and a lenient one as a fallback. If both give
the same counts, the count is a property of the files, not of the regex.

Usage:
    python dataset_audit.py /path/to/msld_v2_download
    python dataset_audit.py /path/to/msld_v2_download --json audit.json

The root is the folder you get after unzipping the Kaggle download. The script
looks for "Original Images", datalog.xlsx and FOLDS_AUG anywhere below it.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Folder name -> class name used in the paper
CLASS_DIRS = {
    "MONKEYPOX": "Mpox", "MPOX": "Mpox",
    "CHICKENPOX": "Chickenpox", "CHICKEN POX": "Chickenpox",
    "MEASLES": "Measles",
    "COWPOX": "Cowpox",
    "HFMD": "HFMD",
    "HEALTHY": "Healthy", "NORMAL": "Healthy",
}
# Patients per class as reported in the dataset paper (Ali et al., 2024)
REPORTED_PATIENTS = {"Mpox": 143, "Chickenpox": 62, "Measles": 46,
                     "Cowpox": 41, "HFMD": 144, "Healthy": 105}
CLASS_ORDER = ["Mpox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]

STRICT_RE = re.compile(r"^(MKP|CHP|MSL|CWP|HFMD|HEALTHY)_(\d+)_\d+", re.I)
LENIENT_RE = re.compile(
    r"(MKP|MPOX|CHP|CHICKEN|MSL|MEASLE|CWP|COWPOX|HFMD|HEALTHY|NORMAL)[-_\s]*\(?(\d+)", re.I)
ALIAS = {"MPOX": "MKP", "CHICKEN": "CHP", "MEASLE": "MSL", "COWPOX": "CWP", "NORMAL": "HEALTHY"}


def patient_id(name, lenient=False):
    stem = Path(str(name)).stem.strip()
    m = (LENIENT_RE.search(stem) if lenient else STRICT_RE.match(stem))
    if not m:
        return None
    prefix = m.group(1).upper()
    prefix = ALIAS.get(prefix, prefix)
    return f"{prefix}_{int(m.group(2))}"


def class_of(path, stop):
    """Class from the nearest parent folder whose name is a class name."""
    for parent in path.parents:
        if parent == stop:
            break
        key = parent.name.strip().upper()
        if key in CLASS_DIRS:
            return CLASS_DIRS[key]
    return None


def find_dir(root, name):
    hits = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == name.lower()]
    return min(hits, key=lambda p: len(p.parts)) if hits else None


def audit_originals(orig_dir):
    rows, outside = [], []
    for p in sorted(orig_dir.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in IMG_EXT):
            continue
        cls = class_of(p, orig_dir.parent)
        if cls is None:
            outside.append(str(p.relative_to(orig_dir.parent)))
            continue
        rows.append({"stem": p.stem.upper(), "cls": cls,
                     "pid_strict": patient_id(p.name),
                     "pid_lenient": patient_id(p.name, lenient=True)})
    # Each photograph appears once per class folder; drop exact duplicates
    seen, unique = set(), []
    for r in rows:
        key = (r["cls"], r["stem"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique, outside


def per_class_table(rows, key):
    imgs, pats = Counter(), defaultdict(set)
    for r in rows:
        imgs[r["cls"]] += 1
        if r[key]:
            pats[r["cls"]].add(r[key])
    return imgs, {c: len(s) for c, s in pats.items()}


def audit_datalog(xlsx, stem_to_pid):
    import pandas as pd
    sheets = pd.read_excel(xlsx, sheet_name=None, engine="openpyxl")
    out, test_imgs, test_pats = [], set(), set()
    for name, df in sheets.items():
        cols = {k: next((c for c in df.columns if k in str(c).lower()), None)
                for k in ("train", "val", "test")}
        if any(v is None for v in cols.values()):
            out.append({"sheet": name, "error": f"columns not found in {list(df.columns)}"})
            continue
        parts = {k: {Path(str(x)).stem.strip().upper() for x in df[c].dropna()}
                 for k, c in cols.items()}
        pids = {k: {stem_to_pid[s] for s in v if s in stem_to_pid} for k, v in parts.items()}
        unmatched = sum(len(v - set(stem_to_pid)) for v in parts.values())
        test_imgs |= parts["test"]
        test_pats |= pids["test"]
        out.append({
            "sheet": name,
            "train": len(parts["train"]), "val": len(parts["val"]), "test": len(parts["test"]),
            "total": sum(len(v) for v in parts.values()),
            "unmatched_names": unmatched,
            "patients_train_and_val": len(pids["train"] & pids["val"]),
            "patients_train_and_test": len(pids["train"] & pids["test"]),
            "patients_val_and_test": len(pids["val"] & pids["test"]),
        })
    return out, len(test_imgs & set(stem_to_pid)), len(test_pats)


def audit_augmented(aug_dir):
    fold_dirs = sorted(p for p in aug_dir.iterdir() if p.is_dir())
    per_fold, name_folds, n_files = {}, defaultdict(set), 0
    for fd in fold_dirs:
        names = [p.name.lower() for p in fd.rglob("*")
                 if p.is_file() and p.suffix.lower() in IMG_EXT]
        per_fold[fd.name] = len(names)
        n_files += len(names)
        for n in names:
            name_folds[n].add(fd.name)
    multi = sum(1 for s in name_folds.values() if len(s) > 1)
    return {"fold_folders": per_fold, "files": n_files,
            "unique_names": len(name_folds), "names_in_more_than_one_fold": multi}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path, help="unzipped MSLD v2.0 folder")
    ap.add_argument("--json", type=Path, help="also write the results to this file")
    a = ap.parse_args()
    root = a.root.resolve()
    report = {}

    orig_dir = find_dir(root, "Original Images")
    if orig_dir is None:
        sys.exit(f"No 'Original Images' folder found under {root}")
    # Some downloads nest "Original Images/Original Images"; use the innermost
    inner = find_dir(orig_dir, "Original Images")
    orig_dir = inner or orig_dir
    rows, outside = audit_originals(orig_dir)
    print(f"Original images folder: {orig_dir}")
    print(f"Original images in class folders: {len(rows)}")
    if outside:
        print(f"Image files outside any class folder ({len(outside)}): {outside[:10]}")

    print("\nClass         Images  Patients(strict)  Patients(lenient)  Reported")
    tables = {}
    for key in ("pid_strict", "pid_lenient"):
        tables[key] = per_class_table(rows, key)
    imgs = tables["pid_strict"][0]
    for c in CLASS_ORDER:
        print(f"{c:<12}{imgs[c]:>8}{tables['pid_strict'][1].get(c, 0):>18}"
              f"{tables['pid_lenient'][1].get(c, 0):>19}{REPORTED_PATIENTS[c]:>10}")
    tot_s = sum(tables["pid_strict"][1].values())
    tot_l = sum(tables["pid_lenient"][1].values())
    print(f"{'Total':<12}{sum(imgs.values()):>8}{tot_s:>18}{tot_l:>19}"
          f"{sum(REPORTED_PATIENTS.values()):>10}")
    no_pid = [r["stem"] for r in rows if r["pid_strict"] is None]
    if no_pid:
        print(f"File names the strict pattern could not parse: {no_pid[:10]}")
    print("Strict and lenient patterns agree." if tot_s == tot_l else
          "Strict and lenient patterns DISAGREE; inspect the unparsed names above.")
    report["originals"] = {"images": sum(imgs.values()), "outside_class_folders": outside,
                           "per_class_images": dict(imgs),
                           "per_class_patients": tables["pid_strict"][1],
                           "patients_total": tot_s, "patients_total_lenient": tot_l}

    stem_to_pid = {r["stem"]: r["pid_strict"] or r["pid_lenient"] or r["stem"] for r in rows}
    xlsx = next(iter(root.rglob("datalog.xlsx")), None)
    if xlsx is None:
        print("\ndatalog.xlsx not found; skipping the fold audit.")
    else:
        sheets, n_test_imgs, n_test_pats = audit_datalog(xlsx, stem_to_pid)
        print(f"\nOfficial folds ({xlsx.name})")
        print("Sheet      Train  Val  Test  Total  Unmatched  Shared patients (tr-va, tr-te, va-te)")
        for s in sheets:
            if "error" in s:
                print(f"{s['sheet']:<10} {s['error']}")
                continue
            print(f"{s['sheet']:<10}{s['train']:>6}{s['val']:>5}{s['test']:>6}{s['total']:>7}"
                  f"{s['unmatched_names']:>11}      {s['patients_train_and_val']}, "
                  f"{s['patients_train_and_test']}, {s['patients_val_and_test']}")
        n = len(rows)
        print(f"Images that are in at least one test partition: {n_test_imgs} of {n} "
              f"({100 * n_test_imgs / max(n, 1):.1f}%), covering {n_test_pats} patients.")
        report["datalog"] = {"sheets": sheets, "test_union_images": n_test_imgs,
                             "test_union_patients": n_test_pats}

    aug_dir = find_dir(root, "FOLDS_AUG")
    if aug_dir is None:
        print("\nFOLDS_AUG not found; skipping the augmented-pool audit.")
    else:
        aug = audit_augmented(aug_dir)
        print(f"\nAugmented pool ({aug_dir})")
        for k, v in aug["fold_folders"].items():
            print(f"  {k}: {v} files")
        print(f"  Files: {aug['files']}, unique file names: {aug['unique_names']} "
              f"({aug['unique_names'] / max(len(rows), 1):.1f} per original image)")
        print(f"  File names present in more than one fold folder: "
              f"{aug['names_in_more_than_one_fold']}")
        report["augmented"] = aug

    if a.json:
        a.json.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {a.json}")


if __name__ == "__main__":
    main()
