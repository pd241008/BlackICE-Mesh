#!/usr/bin/env python3
"""Section III Faithful Diagnostic — ported original pgd_attack with correct defaults.

Runs the original AdvGuard categorical-handling logic (argmax -> nearest one-hot)
with the CORRECT continuous_cols and categorical_groups defaults against the
original baseline and hardened NSL-KDD checkpoints.

Evaluation protocol:
  - Denominator: correct/total (standard, not clean-correct-conditioned)
  - Epsilon schedule: 0.10, 0.12, 0.15 (matching original paper Table II)
  - Alpha: 0.01, Steps: 40, Random start: YES (uniform ε-ball init)
  - Sample scales: n=100 (original quick check) and n=22543 (full test set)

The original pgd_attack function (git commit 566735a) used:
  - continuous_cols = list(range(18))  -- ALL features treated as continuous (BUG)
  - categorical_groups = []            -- DACM snapping disabled (BUG)
These bugs produced the 29.10% artifact. This script corrects both defaults
while keeping the snap logic itself verbatim (argmax -> one_hot).

Usage:
    python scripts/section3_faithful_diagnostic.py
    python scripts/section3_faithful_diagnostic.py --device cuda
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config — NSL-KDD, correct defaults
# ---------------------------------------------------------------------------
CONTINUOUS_COLS = [0, 1, 2, 3]
CATEGORICAL_GROUPS = [
    [4, 5, 6],        # protocol_type, |G|=3
    [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # service, |G|=11
]
FEATURE_DIM = 18
MIN_VAL, MAX_VAL = 0.0, 1.0
N_TOTAL = 22543

TEST_CSV = "./data/nsl-kdd-test.csv"
BASELINE_WEIGHTS = "models/model.pth"
HARDENED_WEIGHTS = "models/model_adv.pth"


# ---------------------------------------------------------------------------
# Model — ControlMLP (matches original architecture)
# ---------------------------------------------------------------------------
class ControlMLP(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden1=64, hidden2=32,
                 num_classes=2, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


def load_weights(path, device):
    model = ControlMLP().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Original AdvGuard pgd_attack — verbatim snap logic, CORRECT defaults
# ---------------------------------------------------------------------------
def pgd_attack_faithful(model, images, labels, epsilon, alpha=0.01, steps=40):
    """
    Original AdvGuard PGD attack with DACM categorical snapping.

    Categorical handling is VERBATIM from the original (commit 566735a):
      adv_cat = images[:, cat_group] + alpha * grad[:, cat_group].sign()
      nearest_idx = torch.argmax(adv_cat, dim=1)
      snapped_tensor = F.one_hot(nearest_idx, num_classes=len(cat_group)).float()
      images.data[:, cat_group] = snapped_tensor

    Differences from the original BUGGY defaults:
      - continuous_cols = [0,1,2,3] (not all 18)
      - categorical_groups = [[4,5,6],[7..17]] (not [])

    This is NOT eval_deepfool_k1.py (exhaustive enumeration).
    This is NOT eval_unified.py (gradient-snapped once at end).
    This applies the argmax+one_hot snap at EVERY PGD step (40 iterations).
    """
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    ori_images = images.clone().detach()

    # Random start (uniform ε-ball) — matches original PGD protocol
    random_noise = torch.empty_like(ori_images).uniform_(-epsilon, epsilon)
    images = torch.clamp(ori_images + random_noise, MIN_VAL, MAX_VAL).detach()

    for _ in range(steps):
        images.requires_grad_(True)
        outputs = model(images)
        model.zero_grad()
        cost = loss_fn(outputs, labels)
        cost.backward()
        grad = images.grad.clone()

        with torch.no_grad():
            # Continuous: gradient step -> L_inf ball -> Min-Max clamp
            if CONTINUOUS_COLS:
                adv_cont = images[:, CONTINUOUS_COLS] + alpha * grad[:, CONTINUOUS_COLS].sign()
                eta = torch.clamp(adv_cont - ori_images[:, CONTINUOUS_COLS], min=-epsilon, max=epsilon)
                images.data[:, CONTINUOUS_COLS] = torch.clamp(
                    ori_images[:, CONTINUOUS_COLS] + eta, min=MIN_VAL, max=MAX_VAL
                )

            # Categorical: VERBATIM original AdvGuard snap (argmax -> nearest one-hot)
            for cat_group in CATEGORICAL_GROUPS:
                adv_cat = images[:, cat_group] + alpha * grad[:, cat_group].sign()
                nearest_idx = torch.argmax(adv_cat, dim=1)
                snapped_tensor = F.one_hot(nearest_idx, num_classes=len(cat_group)).float()
                images.data[:, cat_group] = snapped_tensor

        images = images.detach()

    return images


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, epsilon, alpha=0.01, steps=40, device="cpu"):
    """Standard correct/total denominator (not clean-correct-conditioned)."""
    correct = total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        with torch.enable_grad():
            adv = pgd_attack_faithful(model, data, target, epsilon, alpha, steps)
        with torch.no_grad():
            pred = model(adv).argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct, total


@torch.no_grad()
def evaluate_clean(model, loader, device="cpu"):
    correct = total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        pred = model(data).argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct, total


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_nsl_kdd_test():
    """Load NSL-KDD test set from CSV."""
    import pandas as pd
    df = pd.read_csv(TEST_CSV, header=None)
    # Last column is label (0/1), all others are features
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Section III faithful diagnostic")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit test samples (default: full 22543)")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load data
    X, y = load_nsl_kdd_test()
    n_available = X.size(0)
    n_eval = min(n_available, args.max_samples) if args.max_samples else n_available

    if n_eval < n_available:
        X = X[:n_eval]
        y = y[:n_eval]

    X, y = X.to(device), y.to(device)
    loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=False)

    # Load models
    baseline_model = load_weights(BASELINE_WEIGHTS, device)
    hardened_model = load_weights(HARDENED_WEIGHTS, device)

    # Checkpoint hashes
    import hashlib
    def file_hash(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    baseline_sha = file_hash(BASELINE_WEIGHTS)
    hardened_sha = file_hash(HARDENED_WEIGHTS)

    # Clean accuracy
    baseline_clean, _ = evaluate_clean(baseline_model, loader, device)
    hardened_clean, _ = evaluate_clean(hardened_model, loader, device)

    print("=" * 72)
    print("SECTION III FAITHFUL DIAGNOSTIC")
    print("Original AdvGuard snap logic (argmax -> nearest one-hot)")
    print("CORRECT defaults: continuous_cols=[0,1,2,3], cat_groups=[[4,5,6],[7..17]]")
    print("=" * 72)
    print(f"Device          : {device}")
    print(f"Test samples    : {n_eval} / {n_available}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Attack config   : PGD-40, alpha=0.01, random start")
    print(f"Denominator     : correct / total (standard)")
    print(f"Baseline model  : {BASELINE_WEIGHTS}")
    print(f"  SHA-256       : {baseline_sha}")
    print(f"  Clean accuracy: {baseline_clean}/{n_eval} = {100*baseline_clean/n_eval:.2f}%")
    print(f"Hardened model  : {HARDENED_WEIGHTS}")
    print(f"  SHA-256       : {hardened_sha}")
    print(f"  Clean accuracy: {hardened_clean}/{n_eval} = {100*hardened_clean/n_eval:.2f}%")
    print("-" * 72)

    # Epsilon schedule: 0.10, 0.12, 0.15
    epsilons = [0.10, 0.12, 0.15]
    results = {}

    for eps in epsilons:
        print(f"\n>>> epsilon = {eps}")

        t0 = time.time()
        bl_correct, bl_total = evaluate(baseline_model, loader, eps, device=device)
        bl_time = time.time() - t0
        bl_acc = 100 * bl_correct / bl_total
        print(f"  Baseline : {bl_correct}/{bl_total} = {bl_acc:.2f}%  ({bl_time:.1f}s)")

        t0 = time.time()
        hd_correct, hd_total = evaluate(hardened_model, loader, eps, device=device)
        hd_time = time.time() - t0
        hd_acc = 100 * hd_correct / hd_total
        print(f"  Hardened : {hd_correct}/{hd_total} = {hd_acc:.2f}%  ({hd_time:.1f}s)")

        print(f"  ASR gap  : {bl_acc - hd_acc:.2f}pp (hardened is {'better' if hd_acc > bl_acc else 'worse'} than baseline)")

        results[str(eps)] = {
            "baseline": {"correct": bl_correct, "total": bl_total, "acc_pct": round(bl_acc, 2)},
            "hardened": {"correct": hd_correct, "total": hd_total, "acc_pct": round(hd_acc, 2)},
            "wall_time_s": round(bl_time + hd_time, 1),
        }

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Epsilon':>8}  {'Baseline':>10}  {'Hardened':>10}  {'Gap':>8}")
    print("-" * 42)
    for eps in epsilons:
        r = results[str(eps)]
        bl = r["baseline"]["acc_pct"]
        hd = r["hardened"]["acc_pct"]
        print(f"{eps:>8.2f}  {bl:>9.2f}%  {hd:>9.2f}%  {bl-hd:>+7.2f}pp")

    print()
    print("Original AdvGuard paper reported (Table II):")
    print("  Baseline @0.15: ~12.30%  |  Hardened @0.15: 29.10%")
    print()

    # Save results
    os.makedirs("results/section3", exist_ok=True)
    out = {
        "description": "Section III faithful diagnostic: original AdvGuard snap (argmax -> nearest one-hot) with correct defaults",
        "protocol": {
            "attack": "pgd_attack_faithful (original AdvGuard snap, verbatim)",
            "snap_logic": "torch.argmax -> F.one_hot (applied every PGD step)",
            "continuous_cols": CONTINUOUS_COLS,
            "categorical_groups": CATEGORICAL_GROUPS,
            "epsilon_schedule": epsilons,
            "alpha": 0.01,
            "steps": 40,
            "random_start": True,
            "denominator": "correct / total (standard)",
        },
        "models": {
            "baseline": {"path": BASELINE_WEIGHTS, "sha256": baseline_sha,
                         "clean_pct": round(100 * baseline_clean / n_eval, 2)},
            "hardened": {"path": HARDENED_WEIGHTS, "sha256": hardened_sha,
                         "clean_pct": round(100 * hardened_clean / n_eval, 2)},
        },
        "n_eval": n_eval,
        "n_total": n_available,
        "results": results,
    }
    out_path = "results/section3/faithful_diagnostic_nsl_kdd.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
