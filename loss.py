#!/usr/bin/env python3
"""
loss.py
Named loss functions — pass by name to train.py
Options: ce | ce_weighted | focal | focal_ls
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


def compute_class_weights(train_csv, num_classes=5, device="cuda"):
    df      = pd.read_csv(train_csv)
    counts  = df["label"].value_counts().sort_index()
    n       = counts.sum()
    weights = np.array(
        [n / (num_classes * counts.get(c, 1)) for c in range(num_classes)],
        dtype=np.float32
    )
    return torch.tensor(weights).to(device)


# ── Plain Cross Entropy ────────────────────────────────────────────────────────
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        if targets.ndim == 2:
            targets = targets.argmax(1)
        return self.ce(logits, targets)


# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim=1)
        prob     = log_prob.exp()

        if targets.ndim == 1:
            targets_oh = F.one_hot(targets, num_classes=logits.size(1)).float()
        else:
            targets_oh = targets

        focal_w = (1 - prob) ** self.gamma
        loss    = -(focal_w * log_prob * targets_oh)

        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)

        return loss.sum(1).mean()


# ── Label Smoothing CE ─────────────────────────────────────────────────────────
class LabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon=0.1, weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.weight  = weight

    def forward(self, logits, targets):
        C        = logits.size(1)
        log_prob = F.log_softmax(logits, dim=1)

        if targets.ndim == 1:
            targets_oh = F.one_hot(targets, num_classes=C).float()
        else:
            targets_oh = targets

        smooth = (1 - self.epsilon) * targets_oh + self.epsilon / C
        loss   = -(smooth * log_prob)

        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)

        return loss.sum(1).mean()


# ── Combined Focal + Label Smoothing ──────────────────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, epsilon=0.1, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.fl    = FocalLoss(gamma=gamma, weight=weight)
        self.ls    = LabelSmoothingLoss(epsilon=epsilon, weight=weight)

    def forward(self, logits, targets):
        return self.alpha * self.fl(logits, targets) + self.beta * self.ls(logits, targets)


# ── Loss selector ─────────────────────────────────────────────────────────────
def get_loss(name, weight=None):
    """
    name: ce | ce_weighted | focal | focal_ls
    """
    if name == "ce":
        return CrossEntropyLoss(weight=None)
    elif name == "ce_weighted":
        return CrossEntropyLoss(weight=weight)
    elif name == "focal":
        return FocalLoss(gamma=2.0, weight=weight)
    elif name == "focal_ls":
        return CombinedLoss(weight=weight, gamma=2.0, epsilon=0.1, alpha=0.5, beta=0.5)
    else:
        raise ValueError(f"Unknown loss: {name}. Choose from: ce | ce_weighted | focal | focal_ls")