#!/usr/bin/env python3
"""
EDA script for VinDr-Mammo dataset.
Saves all results to /workspace/BIRAD/analysis/
Run: python eda.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path("/workspace/BIRAD")
DATA        = ROOT / "data" / "images_png"
BREAST_CSV  = ROOT / "breast-level_annotations.csv"
FINDING_CSV = ROOT / "finding_annotations.csv"
META_CSV    = ROOT / "metadata.csv"
OUT         = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

print(f"[INFO] Saving all results to {OUT}")

# ── Load CSVs ──────────────────────────────────────────────────────────────────
print("\n[1/8] Loading CSVs...")
breast_df  = pd.read_csv(BREAST_CSV)
finding_df = pd.read_csv(FINDING_CSV)
meta_df    = pd.read_csv(META_CSV)

print(f"  breast_df  : {breast_df.shape}")
print(f"  finding_df : {finding_df.shape}")
print(f"  meta_df    : {meta_df.shape}")

breast_df.to_csv(OUT / "breast_df_head.csv", index=False)
print(f"  Columns breast_df  : {list(breast_df.columns)}")
print(f"  Columns finding_df : {list(finding_df.columns)}")
print(f"  Columns meta_df    : {list(meta_df.columns)}")

# ── 1. Label Distribution ──────────────────────────────────────────────────────
print("\n[2/8] Label distribution...")

birads_overall = breast_df["breast_birads"].value_counts().sort_index()
birads_train   = breast_df[breast_df["split"] == "training"]["breast_birads"].value_counts().sort_index()
birads_test    = breast_df[breast_df["split"] == "test"]["breast_birads"].value_counts().sort_index()

label_stats = {
    "overall" : birads_overall.to_dict(),
    "train"   : birads_train.to_dict(),
    "test"    : birads_test.to_dict(),
    "overall_pct" : (birads_overall / birads_overall.sum() * 100).round(2).to_dict(),
    "train_pct"   : (birads_train   / birads_train.sum()   * 100).round(2).to_dict(),
    "test_pct"    : (birads_test    / birads_test.sum()    * 100).round(2).to_dict(),
}
with open(OUT / "label_distribution.json", "w") as f:
    json.dump(label_stats, f, indent=2)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (data, title) in zip(axes, [
    (birads_overall, "Overall"),
    (birads_train,   "Train"),
    (birads_test,    "Test"),
]):
    bars = ax.bar(data.index.astype(str), data.values, color="steelblue", edgecolor="black")
    for bar, val in zip(bars, data.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"BI-RADS Distribution — {title}")
    ax.set_xlabel("BI-RADS Category")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUT / "label_distribution.png", dpi=150)
plt.close()
print(f"  Saved label_distribution.json + .png")

# ── 2. Split Verification ──────────────────────────────────────────────────────
print("\n[3/8] Split verification...")

split_counts = breast_df.groupby("split")["study_id"].nunique()
study_stats  = {
    "unique_studies_total" : int(breast_df["study_id"].nunique()),
    "per_split_studies"    : split_counts.to_dict(),
    "unique_images_total"  : int(breast_df["image_id"].nunique()),
    "rows_total"           : int(len(breast_df)),
}

# patient leakage check — no study_id should appear in both splits
train_studies = set(breast_df[breast_df["split"] == "training"]["study_id"])
test_studies  = set(breast_df[breast_df["split"] == "test"]["study_id"])
overlap       = train_studies & test_studies
study_stats["train_test_study_overlap"] = len(overlap)
study_stats["leakage_free"]             = len(overlap) == 0

with open(OUT / "split_verification.json", "w") as f:
    json.dump(study_stats, f, indent=2)
print(f"  Studies train={len(train_studies)}, test={len(test_studies)}, overlap={len(overlap)}")
print(f"  Leakage free: {len(overlap) == 0}")

# ── 3. Image Path Sanity Check ─────────────────────────────────────────────────
print("\n[4/8] Image path sanity check (this may take a moment)...")

missing, found = [], []
for _, row in tqdm(breast_df.iterrows(), total=len(breast_df), desc="  Checking paths"):
    p = DATA / str(row["study_id"]) / f"{row['image_id']}.png"
    (found if p.exists() else missing).append(str(p))

path_stats = {
    "total_rows"    : len(breast_df),
    "found"         : len(found),
    "missing"       : len(missing),
    "missing_paths" : missing[:50],   # first 50 only
}
with open(OUT / "path_sanity.json", "w") as f:
    json.dump(path_stats, f, indent=2)
print(f"  Found: {len(found)}, Missing: {len(missing)}")

# ── 4. CC+MLO Pairing per Breast ──────────────────────────────────────────────
print("\n[5/8] CC+MLO pairing check...")

# each breast = (study_id, laterality)
pairing = breast_df.groupby(["study_id", "laterality"])["view_position"].apply(set).reset_index()
pairing["has_CC"]      = pairing["view_position"].apply(lambda x: "CC"  in x)
pairing["has_MLO"]     = pairing["view_position"].apply(lambda x: "MLO" in x)
pairing["both_views"]  = pairing["has_CC"] & pairing["has_MLO"]
pairing["views_count"] = pairing["view_position"].apply(len)

pair_stats = {
    "total_breasts"          : int(len(pairing)),
    "both_CC_and_MLO"        : int(pairing["both_views"].sum()),
    "missing_CC"             : int((~pairing["has_CC"]).sum()),
    "missing_MLO"            : int((~pairing["has_MLO"]).sum()),
    "view_count_distribution": pairing["views_count"].value_counts().to_dict(),
}
with open(OUT / "cc_mlo_pairing.json", "w") as f:
    json.dump(pair_stats, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 4))
pairing["views_count"].value_counts().sort_index().plot(kind="bar", ax=ax, color="teal", edgecolor="black")
ax.set_title("Views per Breast (CC+MLO)")
ax.set_xlabel("Number of Views")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUT / "cc_mlo_pairing.png", dpi=150)
plt.close()
print(f"  Total breasts: {len(pairing)}, Both views: {pairing['both_views'].sum()}")

# ── 5. Finding Annotations ────────────────────────────────────────────────────
print("\n[6/8] Finding annotations analysis...")

find_stats = {
    "total_findings"         : int(len(finding_df)),
    "unique_studies"         : int(finding_df["study_id"].nunique()),
    "columns"                : list(finding_df.columns),
}

# findings per BI-RADS if column exists
if "finding_birads" in finding_df.columns:
    find_per_birads = finding_df["finding_birads"].value_counts().sort_index().to_dict()
    find_stats["findings_per_birads"] = {str(k): int(v) for k, v in find_per_birads.items()}

# bounding box coverage
bbox_cols = ["xmin", "ymin", "xmax", "ymax"]
has_bbox  = all(c in finding_df.columns for c in bbox_cols)
find_stats["has_bbox_columns"] = has_bbox
if has_bbox:
    valid_bbox = finding_df.dropna(subset=bbox_cols)
    find_stats["findings_with_valid_bbox"] = int(len(valid_bbox))
    find_stats["findings_without_bbox"]    = int(len(finding_df) - len(valid_bbox))

    # bbox size distribution
    valid_bbox = valid_bbox.copy()
    valid_bbox["bbox_w"] = valid_bbox["xmax"] - valid_bbox["xmin"]
    valid_bbox["bbox_h"] = valid_bbox["ymax"] - valid_bbox["ymin"]
    valid_bbox["bbox_area"] = valid_bbox["bbox_w"] * valid_bbox["bbox_h"]

    find_stats["bbox_width_stats"]  = valid_bbox["bbox_w"].describe().round(2).to_dict()
    find_stats["bbox_height_stats"] = valid_bbox["bbox_h"].describe().round(2).to_dict()
    find_stats["bbox_area_stats"]   = valid_bbox["bbox_area"].describe().round(2).to_dict()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(axes,
        ["bbox_w", "bbox_h", "bbox_area"],
        ["BBox Width", "BBox Height", "BBox Area"]):
        ax.hist(valid_bbox[col].clip(upper=valid_bbox[col].quantile(0.99)), bins=50, color="coral", edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("Pixels")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT / "bbox_size_distribution.png", dpi=150)
    plt.close()

with open(OUT / "finding_annotations.json", "w") as f:
    json.dump(find_stats, f, indent=2)
print(f"  Total findings: {len(finding_df)}, Has bbox: {has_bbox}")

# ── 6. Image Statistics Sample ────────────────────────────────────────────────
print("\n[7/8] Image statistics (sampling 500 images)...")

sample_df   = breast_df.sample(min(500, len(breast_df)), random_state=42)
heights, widths, means, stds = [], [], [], []

for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="  Reading images"):
    p = DATA / str(row["study_id"]) / f"{row['image_id']}.png"
    if not p.exists():
        continue
    try:
        img = np.array(Image.open(p).convert("L"), dtype=np.float32)
        heights.append(img.shape[0])
        widths.append(img.shape[1])
        means.append(img.mean())
        stds.append(img.std())
    except Exception:
        pass

img_stats = {
    "sample_size" : len(heights),
    "height" : {"mean": float(np.mean(heights)), "min": int(np.min(heights)), "max": int(np.max(heights)), "std": float(np.std(heights))},
    "width"  : {"mean": float(np.mean(widths)),  "min": int(np.min(widths)),  "max": int(np.max(widths)),  "std": float(np.std(widths))},
    "pixel_mean" : {"mean": float(np.mean(means)), "std": float(np.std(means))},
    "pixel_std"  : {"mean": float(np.mean(stds)),  "std": float(np.std(stds))},
}
with open(OUT / "image_statistics.json", "w") as f:
    json.dump(img_stats, f, indent=2)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, data, title in zip(axes.flat,
    [heights, widths, means, stds],
    ["Image Heights", "Image Widths", "Pixel Means (grayscale)", "Pixel Stds (grayscale)"]):
    ax.hist(data, bins=40, color="mediumseagreen", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUT / "image_statistics.png", dpi=150)
plt.close()
print(f"  Sampled {len(heights)} images")
print(f"  H: {np.mean(heights):.0f}±{np.std(heights):.0f}, W: {np.mean(widths):.0f}±{np.std(widths):.0f}")

# ── 7. Density Distribution ───────────────────────────────────────────────────
print("\n[8/8] Density distribution...")

if "breast_density" in breast_df.columns:
    density_dist = breast_df["breast_density"].value_counts().sort_index()
    density_by_birads = breast_df.groupby(["breast_birads", "breast_density"]).size().unstack(fill_value=0)

    density_stats = {
        "density_distribution" : density_dist.to_dict(),
        "density_by_birads"    : density_by_birads.to_dict(),
    }
    with open(OUT / "density_distribution.json", "w") as f:
        json.dump(density_stats, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    density_dist.plot(kind="bar", ax=axes[0], color="orchid", edgecolor="black")
    axes[0].set_title("Breast Density Distribution")
    axes[0].set_xlabel("Density Category")
    axes[0].set_ylabel("Count")

    density_by_birads.plot(kind="bar", ax=axes[1], colormap="tab10", edgecolor="black")
    axes[1].set_title("Density by BI-RADS Category")
    axes[1].set_xlabel("BI-RADS Category")
    axes[1].set_ylabel("Count")
    axes[1].legend(title="Density", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(OUT / "density_distribution.png", dpi=150)
    plt.close()
    print(f"  Density categories: {list(density_dist.index)}")
else:
    print("  No breast_density column found.")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = {
    "label_distribution" : label_stats,
    "split_verification" : study_stats,
    "path_sanity"        : path_stats,
    "cc_mlo_pairing"     : pair_stats,
    "finding_annotations": find_stats,
    "image_statistics"   : img_stats,
}
with open(OUT / "eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("EDA COMPLETE. Files saved to /workspace/BIRAD/analysis/")
print("="*60)
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}")