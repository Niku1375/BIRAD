#!/usr/bin/env python3
"""
dataset.py
- Reads from preprocessed cache — never reprocesses
- Paired CC+MLO breast-level dataset
- Train/val split at patient (study) level
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

ROOT = Path("/workspace/BIRAD")
PREP = ROOT / "data" / "preprocessed"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(split="train"):
    if split == "train":
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])


def make_train_val_split(train_csv, val_frac=0.1, seed=42):
    """
    Patient-level split of the training CSV into train and val.
    Returns two DataFrames.
    """
    df         = pd.read_csv(train_csv)
    studies    = df["study_id"].unique()
    rng        = np.random.default_rng(seed)
    rng.shuffle(studies)
    n_val      = int(len(studies) * val_frac)
    val_studies   = set(studies[:n_val])
    train_studies = set(studies[n_val:])
    return (
        df[df["study_id"].isin(train_studies)].reset_index(drop=True),
        df[df["study_id"].isin(val_studies)].reset_index(drop=True),
    )


class VinDrDataset(Dataset):
    def __init__(self, df, split="train", transform=None):
        self.df        = df.reset_index(drop=True)
        self.split     = split
        self.transform = transform or get_transforms(split)
        self.prep      = PREP

    def __len__(self):
        return len(self.df)

    def _load(self, study_id, image_id):
        p   = self.prep / str(study_id) / f"{image_id}.png"
        img = Image.open(p).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        cc_img  = self._load(row["study_id"], row["cc_image_id"])
        mlo_img = self._load(row["study_id"], row["mlo_image_id"])
        label   = int(row["label"])
        return cc_img, mlo_img, label


def mixup_batch(cc, mlo, labels, num_classes=5, alpha=0.4):
    lam  = np.random.beta(alpha, alpha)
    idx  = torch.randperm(cc.size(0))
    cc2  = lam * cc  + (1 - lam) * cc[idx]
    mlo2 = lam * mlo + (1 - lam) * mlo[idx]
    y    = torch.zeros(cc.size(0), num_classes)
    y.scatter_(1, labels.unsqueeze(1), 1.0)
    y2   = lam * y + (1 - lam) * y[idx]
    return cc2, mlo2, y2


def get_dataloaders(train_csv, test_csv, batch_size=32, num_workers=4, val_frac=0.1):
    train_df, val_df = make_train_val_split(train_csv, val_frac=val_frac)
    test_df          = pd.read_csv(test_csv)

    train_ds = VinDrDataset(train_df, split="train")
    val_ds   = VinDrDataset(val_df,   split="val")
    test_ds  = VinDrDataset(test_df,  split="val")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_dl, val_dl, test_dl