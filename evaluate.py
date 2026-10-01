#!/usr/bin/env python3
"""
evaluate.py
Full evaluation for a trained model.
Usage:
    python evaluate.py --model convnext_t
"""

import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)
from torch.cuda.amp import autocast
from tqdm import tqdm

from dataset import get_dataloaders
from loss    import CombinedLoss, compute_class_weights

ROOT      = Path("/workspace/BIRAD")
OUT       = ROOT / "analysis"
CKPT_DIR  = ROOT / "checkpoints"
NUM_CLASSES = 5
BIRADS_NAMES = ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5"]


def bootstrap_ci(y_true, y_pred, y_prob, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    f1s, aucs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
        try:
            aucs.append(roc_auc_score(
                np.eye(NUM_CLASSES)[y_true[idx]], y_prob[idx],
                multi_class="ovr", average="macro"
            ))
        except Exception:
            pass
    return {
        "f1_mean": float(np.mean(f1s)), "f1_ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "auc_mean": float(np.mean(aucs)), "auc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res_dir = OUT / "results" / args.model
    res_dir.mkdir(parents=True, exist_ok=True)

    # load model
    if args.model == "resnet50":
        from models.resnet50 import get_model
    elif args.model == "efficientnet_b2":
        from models.efficientnet_b2 import get_model
    elif args.model == "efficientnet_b4":
        from models.efficientnet_b4 import get_model
    elif args.model == "swin_t":
        from models.swin_t import get_model
    elif args.model == "convnext_t":
        from models.convnext_t import get_model
    elif args.model == "convnext_swin":
        from models.convnext_swin import get_model
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = get_model(NUM_CLASSES).to(device)
    ckpt  = torch.load(CKPT_DIR / f"{args.model}_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[INFO] Loaded {args.model} — best epoch {ckpt['epoch']}, val_f1={ckpt['val_f1']:.4f}")

    # dataloader
    train_csv = OUT / "train.csv"
    test_csv  = OUT / "test.csv"
    weights   = compute_class_weights(train_csv, NUM_CLASSES, device)
    criterion = CombinedLoss(weight=weights)
    _, test_dl = get_dataloaders(train_csv, test_csv, batch_size=32, num_workers=4)

    # inference
    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for cc, mlo, labels in tqdm(test_dl, desc="Evaluating"):
            cc, mlo, labels = cc.to(device), mlo.to(device), labels.to(device)
            with autocast():
                logits = model(cc, mlo)
                loss   = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            total_loss  += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # metrics
    macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    macro_auc = roc_auc_score(np.eye(NUM_CLASSES)[y_true], y_prob,
                               multi_class="ovr", average="macro")
    per_class_auc = roc_auc_score(np.eye(NUM_CLASSES)[y_true], y_prob,
                                   multi_class="ovr", average=None)
    cr = classification_report(y_true, y_pred, target_names=BIRADS_NAMES,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    ci = bootstrap_ci(y_true, y_pred, y_prob)

    # 3 vs 4 confusion
    mask_34 = np.isin(y_true, [2, 3])
    cm_34   = confusion_matrix(y_true[mask_34], y_pred[mask_34], labels=[2, 3]).tolist()

    results = {
        "model"          : args.model,
        "macro_f1"       : float(macro_f1),
        "macro_auc"      : float(macro_auc),
        "per_class_f1"   : {BIRADS_NAMES[i]: float(v) for i, v in enumerate(per_class_f1)},
        "per_class_auc"  : {BIRADS_NAMES[i]: float(v) for i, v in enumerate(per_class_auc)},
        "classification_report": cr,
        "confusion_matrix"     : cm.tolist(),
        "birads_3v4_confusion" : cm_34,
        "bootstrap_ci"         : ci,
        "test_loss"            : float(total_loss / len(test_dl)),
    }
    with open(res_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # confusion matrix plot
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=BIRADS_NAMES, yticklabels=BIRADS_NAMES, ax=ax)
    ax.set_title(f"Confusion Matrix — {args.model}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(res_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(BIRADS_NAMES):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={per_class_auc[i]:.3f})")
    ax.plot([0,1],[0,1], "k--")
    ax.set_title(f"ROC Curves — {args.model}")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    plt.tight_layout()
    plt.savefig(res_dir / "roc_curves.png", dpi=150)
    plt.close()

    # BI-RADS 3 vs 4 heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(np.array(cm_34), annot=True, fmt="d", cmap="Oranges",
                xticklabels=["BI-RADS 3","BI-RADS 4"],
                yticklabels=["BI-RADS 3","BI-RADS 4"], ax=ax)
    ax.set_title(f"BI-RADS 3 vs 4 — {args.model}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(res_dir / "birads_3v4.png", dpi=150)
    plt.close()

    print(f"\n[RESULTS] {args.model}")
    print(f"  Macro F1  : {macro_f1:.4f}  CI: {ci['f1_ci']}")
    print(f"  Macro AUC : {macro_auc:.4f} CI: {ci['auc_ci']}")
    print(f"  Saved to  : {res_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args)