#!/usr/bin/env python3
"""
Direct PGD evaluation using the ORIGINAL pgd.py attack code (app.ml.attacks.pgd),
not an independent re-implementation. This gives the exact numbers the original
codebase would produce.

Tests all three attack configurations:
  1. Original PGD (alpha=0.01 for both cont and cat — the repo default)
  2. Fixed PGD (alpha_cont=0.01, alpha_cat=1.0 — properly scaled)
  3. Unconstrained PGD (no DACM snapping)

Against both models, on the full test set.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import the ORIGINAL attack code
from app.ml.attacks.pgd import pgd_attack
from app.ml.data.loader import CATEGORICAL_GROUPS, CONTINUOUS_COLS, FEATURE_DIM

BASELINE_WEIGHTS = "app/ml/model.pth"
HARDENED_WEIGHTS = "app/ml/model_adv.pth"
TEST_CSV = "./data/nsl-kdd-test.csv"

ALL_CAT_IDX = [i for g in CATEGORICAL_GROUPS for i in g]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(18, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_model(path):
    m = MLP().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True), strict=True)
    m.eval()
    return m


def pgd_attack_fixed_cat(model, images, labels, epsilon=0.15, alpha_cont=0.01, alpha_cat=1.0, steps=40):
    """
    PGD with separate alpha for categorical features.
    Same logic as pgd.py but with properly-scaled categorical step size.
    """
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    ori_images = images.clone().detach()

    if CONTINUOUS_COLS:
        random_noise = torch.empty_like(ori_images[:, CONTINUOUS_COLS]).uniform_(-epsilon, epsilon)
        images[:, CONTINUOUS_COLS] = torch.clamp(ori_images[:, CONTINUOUS_COLS] + random_noise, 0.0, 1.0)

    for i in range(steps):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        cost = loss_fn(outputs, labels)
        cost.backward()
        grad = images.grad

        if CONTINUOUS_COLS:
            adv_cont = images[:, CONTINUOUS_COLS] + alpha_cont * grad[:, CONTINUOUS_COLS].sign()
            eta = torch.clamp(adv_cont - ori_images[:, CONTINUOUS_COLS], min=-epsilon, max=epsilon)
            adv_cont_snapped = torch.clamp(ori_images[:, CONTINUOUS_COLS] + eta, min=0.0, max=1.0)
            images.data[:, CONTINUOUS_COLS] = adv_cont_snapped

        for cat_group in CATEGORICAL_GROUPS:
            adv_cat = images[:, cat_group] + alpha_cat * grad[:, cat_group].sign()
            nearest_idx = torch.argmax(adv_cat, dim=1)
            snapped_tensor = F.one_hot(nearest_idx, num_classes=len(cat_group)).float()
            images.data[:, cat_group] = snapped_tensor

        images = images.detach()

    return images


def pgd_unconstrained(model, images, labels, epsilon=0.15, alpha=0.01, steps=40):
    """PGD with no DACM snapping — all features treated as continuous."""
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    ori_images = images.clone().detach()

    random_noise = torch.empty_like(ori_images).uniform_(-epsilon, epsilon)
    images = torch.clamp(ori_images + random_noise, 0.0, 1.0)

    for i in range(steps):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        cost = loss_fn(outputs, labels)
        cost.backward()
        grad = images.grad

        adv = images + alpha * grad.sign()
        eta = torch.clamp(adv - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, min=0.0, max=1.0).detach()

    return images


def evaluate(model, loader, attack_fn, attack_name, **kwargs):
    """Run evaluation with given attack function."""
    correct_clean = 0
    correct_adv = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

        with torch.no_grad():
            clean_pred = model(batch_x).argmax(dim=1)
        correct_clean += (clean_pred == batch_y).sum().item()

        adv = attack_fn(model, batch_x, batch_y, **kwargs)

        with torch.no_grad():
            adv_pred = model(adv).argmax(dim=1)
        correct_adv += (adv_pred == batch_y).sum().item()
        total += batch_y.size(0)

    clean_acc = correct_clean / total * 100
    robust_acc = correct_adv / total * 100
    print(f"  [{attack_name}] Clean: {clean_acc:.2f}%, Robust: {robust_acc:.2f}% ({correct_adv}/{total})")
    return clean_acc, robust_acc


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("PGD EVALUATION — Using Original pgd.py Attack Code")
    print(f"Config: epsilon=0.15, alpha=0.01, steps=40, device={DEVICE}")
    print("=" * 70)

    # Load data
    data = np.loadtxt(TEST_CSV, delimiter=',')
    X = torch.tensor(data[:, :18], dtype=torch.float32)
    y = torch.tensor(data[:, 18], dtype=torch.long)
    print(f"Test set: {X.shape[0]} samples")

    loader = DataLoader(TensorDataset(X, y), batch_size=500, shuffle=False)

    # Load models
    baseline = load_model(BASELINE_WEIGHTS)
    hardened = load_model(HARDENED_WEIGHTS)

    print(f"\n{'='*70}")
    print("MODEL: Baseline (model.pth)")
    print(f"{'='*70}")

    # 1. Original PGD from pgd.py (alpha=0.01 for everything)
    torch.manual_seed(42)
    evaluate(baseline, loader, pgd_attack,
             "Original PGD (α=0.01, from pgd.py)",
             epsilon=0.15, alpha=0.01, steps=40)

    # 2. Fixed PGD (alpha_cat=1.0)
    torch.manual_seed(42)
    evaluate(baseline, loader, pgd_attack_fixed_cat,
             "Fixed PGD (α_cont=0.01, α_cat=1.0)",
             epsilon=0.15, alpha_cont=0.01, alpha_cat=1.0, steps=40)

    # 3. Unconstrained PGD
    torch.manual_seed(42)
    evaluate(baseline, loader, pgd_unconstrained,
             "Unconstrained PGD (no DACM snap)",
             epsilon=0.15, alpha=0.01, steps=40)

    print(f"\n{'='*70}")
    print("MODEL: FGSM-Hardened (model_adv.pth)")
    print(f"{'='*70}")

    # 1. Original PGD from pgd.py
    torch.manual_seed(42)
    evaluate(hardened, loader, pgd_attack,
             "Original PGD (α=0.01, from pgd.py)",
             epsilon=0.15, alpha=0.01, steps=40)

    # 2. Fixed PGD (alpha_cat=1.0)
    torch.manual_seed(42)
    evaluate(hardened, loader, pgd_attack_fixed_cat,
             "Fixed PGD (α_cont=0.01, α_cat=1.0)",
             epsilon=0.15, alpha_cont=0.01, alpha_cat=1.0, steps=40)

    # 3. Unconstrained PGD
    torch.manual_seed(42)
    evaluate(hardened, loader, pgd_unconstrained,
             "Unconstrained PGD (no DACM snap)",
             epsilon=0.15, alpha=0.01, steps=40)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY — All PGD, Full Test Set")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
