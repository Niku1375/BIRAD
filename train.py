#!/usr/bin/env python3
"""
train.py
Usage:
    python train.py --model convnext_t --loss focal_ls
    python train.py --model resnet50   --loss ce
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    classification_report
)
from torch.cuda.amp import GradScaler, autocast

from dataset import get_dataloaders, mixup_batch
from loss    import get_loss, compute_class_weights

# ── Hardcoded config — change here if needed ───────────────────────────────────
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 1e-4
NUM_WORKERS = 4
VAL_FRAC    = 0.1
USE_MIXUP   = True
NUM_CLASSES = 5
WEIGHT_DECAY= 1e-4
CHECKPOINT_START_FRAC = 0.6   # start checkpointing after 60% of epochs

ROOT     = Path("/workspace/BIRAD")
OUT      = ROOT / "analysis"
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

BIRADS_NAMES = ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5"]


def get_model(name):
    if name == "resnet50":
        from models.resnet50 import get_model
    elif name == "efficientnet_b2":
        from models.efficientnet_b2 import get_model
    elif name == "efficientnet_b4":
        from models.efficientnet_b4 import get_model
    elif name == "swin_t":
        from models.swin_t import get_model
    elif name == "convnext_t":
        from models.convnext_t import get_model
    elif name == "convnext_swin":
        from models.convnext_swin import get_model
    else:
        raise ValueError(f"Unknown model: {name}")
    return get_model(NUM_CLASSES)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for cc, mlo, labels in tqdm(loader, desc="  Train", leave=False):
        cc, mlo, labels = cc.to(device), mlo.to(device), labels.to(device)

        if USE_MIXUP:
            cc, mlo, soft_labels = mixup_batch(cc, mlo, labels, NUM_CLASSES)
            soft_labels = soft_labels.to(device)
            target = soft_labels
        else:
            target = labels

        optimizer.zero_grad()
        with autocast():
            logits = model(cc, mlo)
            loss   = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []

    for cc, mlo, labels in tqdm(loader, desc="  Val", leave=False):
        cc, mlo, labels = cc.to(device), mlo.to(device), labels.to(device)
        with autocast():
            logits = model(cc, mlo)
            loss   = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        total_loss  += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    y_true   = np.array(all_labels)
    y_pred   = np.array(all_preds)
    y_prob   = np.array(all_probs)

    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_f1   = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    try:
        macro_auc = roc_auc_score(
            np.eye(NUM_CLASSES)[y_true], y_prob,
            multi_class="ovr", average="macro"
        )
        per_auc = roc_auc_score(
            np.eye(NUM_CLASSES)[y_true], y_prob,
            multi_class="ovr", average=None
        ).tolist()
    except Exception:
        macro_auc = 0.0
        per_auc   = [0.0] * NUM_CLASSES

    # rare class F1 (BI-RADS 3,4,5 → indices 2,3,4)
    rare_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)[[2, 3, 4]].mean()

    return {
        "loss"      : float(avg_loss),
        "accuracy"  : float(acc),
        "macro_f1"  : float(macro_f1),
        "rare_f1"   : float(rare_f1),
        "macro_auc" : float(macro_auc),
        "per_class_f1"  : {BIRADS_NAMES[i]: float(v) for i, v in enumerate(per_f1)},
        "per_class_auc" : {BIRADS_NAMES[i]: float(v) for i, v in enumerate(per_auc)},
    }


def main(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{args.model}__{args.loss}"
    print(f"\n[INFO] Run: {run_name} | Device: {device}")

    # dirs
    log_dir  = OUT / "logs" / run_name
    ckpt_run = CKPT_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_run.mkdir(parents=True, exist_ok=True)

    # data
    train_csv = OUT / "train.csv"
    test_csv  = OUT / "test.csv"
    train_dl, val_dl, _ = get_dataloaders(
        train_csv, test_csv,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_frac=VAL_FRAC
    )

    # model + loss + optimizer
    model     = get_model(args.model).to(device)
    weights   = compute_class_weights(train_csv, NUM_CLASSES, device)
    criterion = get_loss(args.loss, weight=weights)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler()

    checkpoint_start = int(EPOCHS * CHECKPOINT_START_FRAC)
    best_f1  = 0.0
    best_acc = 0.0
    history  = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}  [{run_name}]")

        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_dl, optimizer, criterion, scaler, device
        )
        val_metrics = evaluate_loader(model, val_dl, criterion, device)
        scheduler.step()

        print(f"  Train — Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f} | F1: {tr_f1:.4f}")
        print(f"  Val   — Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['macro_f1']:.4f} | "
              f"AUC: {val_metrics['macro_auc']:.4f} | "
              f"RareF1: {val_metrics['rare_f1']:.4f}")

        row = {
            "epoch"   : epoch,
            "tr_loss" : float(tr_loss),
            "tr_acc"  : float(tr_acc),
            "tr_f1"   : float(tr_f1),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        # checkpointing after 60%
        if epoch >= checkpoint_start:
            val_f1  = val_metrics["macro_f1"]
            val_acc = val_metrics["accuracy"]

            save = False
            tag  = []

            if val_f1 > best_f1:
                best_f1 = val_f1
                save = True
                tag.append("best_f1")

            if val_acc > best_acc:
                best_acc = val_acc
                save = True
                tag.append("best_acc")

            if save:
                tag_str = "_".join(tag)
                ckpt_path = ckpt_run / f"epoch{epoch:04d}_{tag_str}_f1{val_f1:.4f}_acc{val_acc:.4f}.pth"
                torch.save({
                    "epoch"       : epoch,
                    "model_state" : model.state_dict(),
                    "val_f1"      : val_f1,
                    "val_acc"     : val_acc,
                    "val_auc"     : val_metrics["macro_auc"],
                    "val_rare_f1" : val_metrics["rare_f1"],
                    "run_name"    : run_name,
                }, ckpt_path)
                print(f"  [✓] Checkpoint saved [{tag_str}] → {ckpt_path.name}")

        # save history every epoch
        with open(log_dir / "history.json", "w") as f:
            json.dump({
                "run"        : run_name,
                "model"      : args.model,
                "loss"       : args.loss,
                "best_val_f1" : best_f1,
                "best_val_acc": best_acc,
                "history"    : history,
            }, f, indent=2)

    print(f"\n[DONE] {run_name} | Best F1: {best_f1:.4f} | Best Acc: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet50","efficientnet_b2","efficientnet_b4",
                                 "swin_t","convnext_t","convnext_swin"])
    parser.add_argument("--loss",  type=str, required=True,
                        choices=["ce","ce_weighted","focal","focal_ls"])
    args = parser.parse_args()
    main(args)