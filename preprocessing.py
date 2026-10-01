#!/usr/bin/env python3
"""
preprocessing.py
- Builds breast-level CC+MLO paired CSV
- Drops any study with < 4 images (incomplete exams)
- Applies CLAHE + breast crop + resize 224x224
- Saves preprocessed PNGs to data/preprocessed/<study_id>/<image_id>.png
- Saves train.csv and test.csv to analysis/
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path("/workspace/BIRAD")
DATA       = ROOT / "data" / "images_png"
PREP       = ROOT / "data" / "preprocessed"
BREAST_CSV = ROOT / "breast-level_annotations.csv"
OUT        = ROOT / "analysis"
PREP.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

IMG_SIZE = 224

# ── Load CSV ───────────────────────────────────────────────────────────────────
print("[1/5] Loading annotations...")
df = pd.read_csv(BREAST_CSV)
print(f"  Total rows: {len(df)}")

# ── Drop studies with < 4 images ───────────────────────────────────────────────
print("[2/5] Dropping incomplete studies (< 4 images)...")
study_counts = df.groupby("study_id")["image_id"].count()
valid_studies = study_counts[study_counts >= 4].index
dropped = len(study_counts) - len(valid_studies)
df = df[df["study_id"].isin(valid_studies)].reset_index(drop=True)
print(f"  Dropped {dropped} studies, keeping {df['study_id'].nunique()} studies, {len(df)} rows")

# ── Build breast-level paired CSV ─────────────────────────────────────────────
print("[3/5] Building breast-level paired CSV...")

records = []
for (study_id, laterality), grp in df.groupby(["study_id", "laterality"]):
    views = grp.set_index("view_position")
    if "CC" not in views.index or "MLO" not in views.index:
        continue  # skip if either view missing

    cc_row  = views.loc["CC"]
    mlo_row = views.loc["MLO"]

    # handle duplicate view entries — take first
    if isinstance(cc_row, pd.DataFrame):
        cc_row = cc_row.iloc[0]
    if isinstance(mlo_row, pd.DataFrame):
        mlo_row = mlo_row.iloc[0]

    records.append({
        "study_id"      : study_id,
        "laterality"    : laterality,
        "cc_image_id"   : cc_row["image_id"],
        "mlo_image_id"  : mlo_row["image_id"],
        "breast_birads" : cc_row["breast_birads"],
        "breast_density": cc_row["breast_density"],
        "split"         : cc_row["split"],
    })

paired_df = pd.DataFrame(records)
print(f"  Total breast pairs: {len(paired_df)}")
print(f"  Train: {(paired_df['split']=='training').sum()}, Test: {(paired_df['split']=='test').sum()}")

# map BI-RADS to integer label 0-4
birads_map = {"BI-RADS 1": 0, "BI-RADS 2": 1, "BI-RADS 3": 2, "BI-RADS 4": 3, "BI-RADS 5": 4}
paired_df["label"] = paired_df["breast_birads"].map(birads_map)

# ── Preprocessing functions ────────────────────────────────────────────────────
def crop_breast(img):
    """Crop to bounding box of non-zero region (remove black borders)."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

def apply_clahe(img):
    """Apply CLAHE to grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocess_image(src_path, dst_path):
    """Full pipeline: load → crop → CLAHE → resize → save."""
    if dst_path.exists():
        return True  # already cached
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    img = crop_breast(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), img)
    return True

# ── Process all images ─────────────────────────────────────────────────────────
print("[4/5] Preprocessing images (cached — skips already done)...")

all_image_ids = pd.concat([
    df[["study_id", "image_id"]]
]).drop_duplicates()

failed = []
for _, row in tqdm(all_image_ids.iterrows(), total=len(all_image_ids), desc="  Processing"):
    src = DATA / str(row["study_id"]) / f"{row['image_id']}.png"
    dst = PREP / str(row["study_id"]) / f"{row['image_id']}.png"
    ok  = preprocess_image(src, dst)
    if not ok:
        failed.append(str(src))

print(f"  Done. Failed: {len(failed)}")
if failed:
    with open(OUT / "preprocessing_failed.json", "w") as f:
        json.dump(failed, f, indent=2)

# ── Save CSVs ──────────────────────────────────────────────────────────────────
print("[5/5] Saving train/test CSVs...")

train_df = paired_df[paired_df["split"] == "training"].reset_index(drop=True)
test_df  = paired_df[paired_df["split"] == "test"].reset_index(drop=True)

train_df.to_csv(OUT / "train.csv", index=False)
test_df.to_csv(OUT  / "test.csv",  index=False)

stats = {
    "total_pairs" : int(len(paired_df)),
    "train_pairs" : int(len(train_df)),
    "test_pairs"  : int(len(test_df)),
    "dropped_studies" : int(dropped),
    "failed_images"   : len(failed),
    "label_distribution_train": train_df["breast_birads"].value_counts().to_dict(),
    "label_distribution_test" : test_df["breast_birads"].value_counts().to_dict(),
}
with open(OUT / "preprocessing_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n" + "="*60)
print("PREPROCESSING COMPLETE")
print(f"  Train pairs : {len(train_df)}")
print(f"  Test pairs  : {len(test_df)}")
print(f"  Preprocessed images in: {PREP}")
print("="*60)