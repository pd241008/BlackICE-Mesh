#!/usr/bin/env python3
"""
INDEPENDENT VERIFICATION SCRIPT
================================
This script trusts NOTHING from prior work. It:
1. Loads data directly from CSV (verifies structure)
2. Loads model weights directly (verifies architecture compatibility)
3. Re-implements PGD attack from scratch (no imports from app.ml.attacks)
4. Tests EACH claimed bug independently with controlled experiments
5. Reports all numbers with full diagnostics

Claims to verify:
  BUG 1: Categorical alpha=0.01 is too small to ever flip a one-hot component
  BUG 2: Unconstrained PGD produces invalid one-hot sums ("phantom packets")
  CLAIM: Clean accuracies for baseline (~80.5%) and hardened (~78.4%)
  CLAIM: Robust accuracy collapses to ~0% when alpha_cat is properly scaled

Run from ml-optimizer/:
    python independent_verification.py
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ============================================================================
# SECTION 0: CONSTANTS (independently defined, NOT imported)
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_CSV = "./data/nsl-kdd-test.csv"
BASELINE_WEIGHTS = "app/ml/model.pth"
HARDENED_WEIGHTS = "app/ml/model_adv.pth"

# Feature layout: 4 continuous + 3 protocol one-hot + 11 service one-hot = 18
CONTINUOUS_COLS = [0, 1, 2, 3]
PROTOCOL_GROUP = [4, 5, 6]        # 3-way one-hot
SERVICE_GROUP = list(range(7, 18)) # 11-way one-hot
CATEGORICAL_GROUPS = [PROTOCOL_GROUP, SERVICE_GROUP]
ALL_CAT_IDX = PROTOCOL_GROUP + SERVICE_GROUP
FEATURE_DIM = 18


# ============================================================================
# SECTION 1: DATA LOADING (independent, from raw CSV)
# ============================================================================
def load_test_data():
    """Load directly from CSV, verify structure independently."""
    data = np.loadtxt(TEST_CSV, delimiter=',')
    print(f"[DATA] Raw shape: {data.shape}")
    assert data.shape[1] == 19, f"Expected 19 columns (18 features + 1 label), got {data.shape[1]}"

    X = torch.tensor(data[:, :18], dtype=torch.float32)
    y = torch.tensor(data[:, 18], dtype=torch.long)

    # Verify one-hot structure of loaded data
    n_samples = X.shape[0]
    protocol_sums = X[:, PROTOCOL_GROUP].sum(dim=1)
    service_sums = X[:, SERVICE_GROUP].sum(dim=1)

    proto_valid = (protocol_sums - 1.0).abs().max().item()
    svc_valid = (service_sums - 1.0).abs().max().item()
    print(f"[DATA] Protocol one-hot sum deviation from 1.0: max={proto_valid:.6f}")
    print(f"[DATA] Service one-hot sum deviation from 1.0: max={svc_valid:.6f}")
    assert proto_valid < 1e-5, "Protocol features are not valid one-hot!"
    assert svc_valid < 1e-5, "Service features are not valid one-hot!"

    # Verify continuous features are in [0,1]
    cont_min = X[:, CONTINUOUS_COLS].min().item()
    cont_max = X[:, CONTINUOUS_COLS].max().item()
    print(f"[DATA] Continuous features range: [{cont_min:.4f}, {cont_max:.4f}]")

    # Label distribution
    label_counts = torch.bincount(y)
    print(f"[DATA] Labels: {label_counts.tolist()} (class 0={label_counts[0]}, class 1={label_counts[1]})")
    print(f"[DATA] Majority class baseline: {label_counts.max().item()/n_samples*100:.2f}%")

    return X, y, n_samples


# ============================================================================
# SECTION 2: MODEL (re-implemented independently)
# ============================================================================
class IndependentMLP(nn.Module):
    """
    Re-implementation matching the checkpoint layout.
    Keys in checkpoint: fc1.weight, fc1.bias, bn1.weight, bn1.bias,
                        bn1.running_mean, bn1.running_var, bn1.num_batches_tracked,
                        fc2.weight, fc2.bias, fc3.weight, fc3.bias
    """
    def __init__(self, input_dim=18, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_model(path):
    """Load model, verify weight shapes, set eval mode."""
    model = IndependentMLP().to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)

    # Print checkpoint keys for verification
    print(f"[MODEL] Checkpoint keys: {list(state.keys())}")
    print(f"[MODEL] fc1.weight shape: {state['fc1.weight'].shape}")
    print(f"[MODEL] fc3.weight shape: {state['fc3.weight'].shape}")

    model.load_state_dict(state, strict=True)
    model.eval()

    # Verify eval mode
    assert not model.training, "Model must be in eval mode!"
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            assert not m.training, "BatchNorm must be in eval mode!"
            assert m.track_running_stats, "BatchNorm must use running stats!"

    return model


# ============================================================================
# SECTION 3: ATTACK IMPLEMENTATIONS (from scratch, not imported)
# ============================================================================
def pgd_attack_constrained(model, X, y, epsilon, alpha_cont, alpha_cat, steps):
    """
    DACM-constrained PGD attack, re-implemented from scratch.
    - Continuous features: L_inf signed gradient step + projection + [0,1] clamp
    - Categorical features: gradient step with alpha_cat + argmax snap to one-hot

    Returns: adversarial examples, per-step diagnostic info
    """
    orig = X.clone().detach()
    adv = X.clone().detach()
    loss_fn = nn.CrossEntropyLoss()

    diag = {"cat_flips_per_step": [], "loss_per_step": []}

    for step in range(steps):
        adv.requires_grad_(True)
        out = model(adv)
        loss = loss_fn(out, y)
        model.zero_grad()
        loss.backward()
        grad = adv.grad.detach()

        diag["loss_per_step"].append(loss.item())

        with torch.no_grad():
            # Track categorical state BEFORE update
            old_cat_argmax = {}
            for group in CATEGORICAL_GROUPS:
                old_cat_argmax[tuple(group)] = adv[:, group].argmax(dim=1).clone()

            # 1. Continuous perturbation
            adv_cont = adv[:, CONTINUOUS_COLS] + alpha_cont * grad[:, CONTINUOUS_COLS].sign()
            eta = (adv_cont - orig[:, CONTINUOUS_COLS]).clamp(-epsilon, epsilon)
            adv.data[:, CONTINUOUS_COLS] = (orig[:, CONTINUOUS_COLS] + eta).clamp(0.0, 1.0)

            # 2. Categorical perturbation + argmax snap
            total_flips = 0
            total_elements = 0
            for group in CATEGORICAL_GROUPS:
                adv_cat = adv[:, group] + alpha_cat * grad[:, group].sign()
                nearest_idx = torch.argmax(adv_cat, dim=1)
                snapped = F.one_hot(nearest_idx, num_classes=len(group)).float()
                adv.data[:, group] = snapped

                # Count flips
                new_argmax = adv[:, group].argmax(dim=1)
                flips = (new_argmax != old_cat_argmax[tuple(group)]).sum().item()
                total_flips += flips
                total_elements += X.shape[0]

            diag["cat_flips_per_step"].append(total_flips / total_elements * 100)

        adv = adv.detach()

    return adv, diag


def pgd_attack_unconstrained(model, X, y, epsilon, alpha, steps):
    """
    Unconstrained PGD: no DACM snapping. All features treated as continuous.
    """
    orig = X.clone().detach()
    adv = X.clone().detach()
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        adv.requires_grad_(True)
        out = model(adv)
        loss = loss_fn(out, y)
        model.zero_grad()
        loss.backward()
        grad = adv.grad.detach()

        with torch.no_grad():
            adv_new = adv + alpha * grad.sign()
            eta = (adv_new - orig).clamp(-epsilon, epsilon)
            adv = (orig + eta).clamp(0.0, 1.0)

        adv = adv.detach()

    return adv


# ============================================================================
# SECTION 4: VERIFICATION FUNCTIONS
# ============================================================================
def verify_clean_accuracy(model, X, y, name):
    """Compute clean accuracy on full test set."""
    model.eval()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            preds = model(batch_x).argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    acc = correct / total * 100
    print(f"[CLEAN] {name}: {acc:.2f}% ({correct}/{total})")
    return acc


def verify_bug1_categorical_stepsize(model, X, y):
    """
    BUG 1 VERIFICATION: Does alpha_cat=0.01 actually flip any categories?

    Method: Run constrained PGD with alpha_cat=0.01 vs alpha_cat=1.0,
    count categorical flips at each step.
    """
    print("\n" + "=" * 70)
    print("BUG 1 VERIFICATION: Categorical Step Size")
    print("=" * 70)

    # Use a subset for speed
    subset = min(1000, X.shape[0])
    X_sub = X[:subset].to(DEVICE)
    y_sub = y[:subset].to(DEVICE)

    # Test with alpha_cat = 0.01 (original)
    print("\n--- alpha_cat = 0.01 (original code's value) ---")
    _, diag_small = pgd_attack_constrained(
        model, X_sub, y_sub,
        epsilon=0.15, alpha_cont=0.01, alpha_cat=0.01, steps=40
    )
    total_flip_rate_small = np.mean(diag_small["cat_flips_per_step"])
    print(f"  Mean categorical flip rate per step: {total_flip_rate_small:.4f}%")
    print(f"  Flip rates first 5 steps: {[f'{x:.4f}%' for x in diag_small['cat_flips_per_step'][:5]]}")

    # Test with alpha_cat = 1.0 (properly scaled)
    print("\n--- alpha_cat = 1.0 (properly scaled) ---")
    _, diag_large = pgd_attack_constrained(
        model, X_sub, y_sub,
        epsilon=0.15, alpha_cont=0.01, alpha_cat=1.0, steps=40
    )
    total_flip_rate_large = np.mean(diag_large["cat_flips_per_step"])
    print(f"  Mean categorical flip rate per step: {total_flip_rate_large:.4f}%")
    print(f"  Flip rates first 5 steps: {[f'{x:.4f}%' for x in diag_large['cat_flips_per_step'][:5]]}")

    # Test with alpha_cat = 0.5 (intermediate)
    print("\n--- alpha_cat = 0.5 (intermediate) ---")
    _, diag_mid = pgd_attack_constrained(
        model, X_sub, y_sub,
        epsilon=0.15, alpha_cont=0.01, alpha_cat=0.5, steps=40
    )
    total_flip_rate_mid = np.mean(diag_mid["cat_flips_per_step"])
    print(f"  Mean categorical flip rate per step: {total_flip_rate_mid:.4f}%")

    # Now measure robust accuracy for each
    print("\n--- Robust accuracy comparison ---")
    for alpha_cat_val in [0.01, 0.1, 0.5, 1.0]:
        adv, _ = pgd_attack_constrained(
            model, X_sub, y_sub,
            epsilon=0.15, alpha_cont=0.01, alpha_cat=alpha_cat_val, steps=40
        )
        with torch.no_grad():
            preds = model(adv).argmax(dim=1)
        acc = (preds == y_sub).float().mean().item() * 100
        print(f"  alpha_cat={alpha_cat_val:.2f}: robust_acc={acc:.2f}%")

    # Verdict
    print(f"\n[BUG 1 VERDICT]:")
    if total_flip_rate_small < 1.0:
        print(f"  CONFIRMED: alpha_cat=0.01 produces <1% categorical flips ({total_flip_rate_small:.4f}%)")
        print(f"  The categorical attack surface is effectively disabled with the original step size.")
    else:
        print(f"  NOT CONFIRMED: alpha_cat=0.01 produces {total_flip_rate_small:.4f}% flips")

    return total_flip_rate_small, total_flip_rate_large


def verify_bug2_phantom_packets(model, X, y):
    """
    BUG 2 VERIFICATION: Does unconstrained PGD produce invalid one-hot sums?

    Method: Run unconstrained PGD and check if categorical feature groups
    still sum to exactly 1.0.
    """
    print("\n" + "=" * 70)
    print("BUG 2 VERIFICATION: Phantom Packets (Invalid One-Hot Sums)")
    print("=" * 70)

    subset = min(1000, X.shape[0])
    X_sub = X[:subset].to(DEVICE)
    y_sub = y[:subset].to(DEVICE)

    # Verify clean data has valid one-hot
    for i, group in enumerate(CATEGORICAL_GROUPS):
        sums = X_sub[:, group].sum(dim=1)
        print(f"  Clean data group {i} one-hot sums: min={sums.min():.4f}, max={sums.max():.4f}, mean={sums.mean():.4f}")

    # Run unconstrained PGD
    adv = pgd_attack_unconstrained(model, X_sub, y_sub, epsilon=0.15, alpha=0.01, steps=40)

    # Check one-hot validity after attack
    print("\n  After unconstrained PGD (epsilon=0.15, alpha=0.01, steps=40):")
    any_invalid = False
    for i, group in enumerate(CATEGORICAL_GROUPS):
        sums = adv[:, group].sum(dim=1)
        deviations = (sums - 1.0).abs()
        invalid_count = (deviations > 0.01).sum().item()
        print(f"  Group {i} one-hot sums: min={sums.min():.4f}, max={sums.max():.4f}, "
              f"mean={sums.mean():.4f}, invalid(>0.01 from 1.0)={invalid_count}/{subset}")

        # Check if values are still binary (0 or 1)
        vals = adv[:, group]
        non_binary = ((vals > 0.01) & (vals < 0.99)).sum().item()
        print(f"  Group {i} non-binary values (not 0 or 1): {non_binary}")

        if invalid_count > 0 or non_binary > 0:
            any_invalid = True

    # Show some example adversarial categorical values
    print(f"\n  Example adversarial protocol features (first 5 samples):")
    for j in range(min(5, subset)):
        vals = adv[j, PROTOCOL_GROUP].cpu().numpy()
        print(f"    Sample {j}: {vals} (sum={sum(vals):.4f})")

    print(f"\n  Example adversarial service features (first 3 samples):")
    for j in range(min(3, subset)):
        vals = adv[j, SERVICE_GROUP].cpu().numpy()
        print(f"    Sample {j}: sum={sum(vals):.4f}")

    print(f"\n[BUG 2 VERDICT]:")
    if any_invalid:
        print(f"  CONFIRMED: Unconstrained PGD produces invalid one-hot encodings.")
        print(f"  These represent physically impossible network packets.")
    else:
        print(f"  NOT CONFIRMED: One-hot sums remain valid after unconstrained PGD.")

    return any_invalid


def verify_robust_accuracy_full(model, X, y, name, alpha_cat=0.01):
    """
    Full test set robust accuracy evaluation.
    """
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=500, shuffle=False)

    correct = 0
    total = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        adv, _ = pgd_attack_constrained(
            model, batch_x, batch_y,
            epsilon=0.15, alpha_cont=0.01, alpha_cat=alpha_cat, steps=40
        )
        with torch.no_grad():
            preds = model(adv).argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    acc = correct / total * 100
    print(f"[ROBUST] {name} (alpha_cat={alpha_cat}): {acc:.2f}% ({correct}/{total})")
    return acc


def verify_robust_accuracy_unconstrained_full(model, X, y, name):
    """Full test set unconstrained robust accuracy."""
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=500, shuffle=False)

    correct = 0
    total = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        adv = pgd_attack_unconstrained(
            model, batch_x, batch_y,
            epsilon=0.15, alpha=0.01, steps=40
        )
        with torch.no_grad():
            preds = model(adv).argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    acc = correct / total * 100
    print(f"[ROBUST-UNCONSTRAINED] {name}: {acc:.2f}% ({correct}/{total})")
    return acc


def verify_original_pgd_matches(model, X, y):
    """
    Verify whether the ORIGINAL pgd.py code (from app.ml.attacks)
    matches our independent implementation with alpha_cat=alpha_cont.
    """
    print("\n" + "=" * 70)
    print("CROSS-CHECK: Original pgd.py vs Independent Implementation")
    print("=" * 70)

    from app.ml.attacks.pgd import pgd_attack as original_pgd

    subset = min(500, X.shape[0])
    X_sub = X[:subset].to(DEVICE)
    y_sub = y[:subset].to(DEVICE)

    # Set deterministic seed
    torch.manual_seed(42)
    adv_orig = original_pgd(model, X_sub, y_sub, epsilon=0.15, alpha=0.01, steps=40)

    torch.manual_seed(42)
    adv_indep, _ = pgd_attack_constrained(
        model, X_sub, y_sub,
        epsilon=0.15, alpha_cont=0.01, alpha_cat=0.01, steps=40
    )

    # Compare
    max_diff = (adv_orig - adv_indep).abs().max().item()
    mean_diff = (adv_orig - adv_indep).abs().mean().item()
    print(f"  Max absolute difference: {max_diff:.8f}")
    print(f"  Mean absolute difference: {mean_diff:.8f}")

    with torch.no_grad():
        pred_orig = model(adv_orig).argmax(dim=1)
        pred_indep = model(adv_indep).argmax(dim=1)
    pred_match = (pred_orig == pred_indep).float().mean().item() * 100
    print(f"  Prediction agreement: {pred_match:.2f}%")

    if max_diff < 1e-5:
        print("  MATCH: Independent implementation produces identical results.")
    else:
        print(f"  MISMATCH: Implementations differ (max diff={max_diff:.8f})")
        # Investigate where they differ
        diff_mask = (adv_orig - adv_indep).abs() > 1e-5
        if diff_mask.any():
            diff_cols = diff_mask.any(dim=0).nonzero().flatten().tolist()
            print(f"  Differing feature columns: {diff_cols}")


def verify_model_predictions_detailed(model, X, y, name):
    """Detailed prediction analysis to check for degenerate classifiers."""
    print(f"\n--- Detailed prediction analysis: {name} ---")

    X_dev = X.to(DEVICE)
    y_dev = y.to(DEVICE)

    with torch.no_grad():
        logits = model(X_dev)
        preds = logits.argmax(dim=1)

    # Prediction distribution
    pred_counts = torch.bincount(preds, minlength=2)
    print(f"  Prediction distribution: class 0={pred_counts[0].item()}, class 1={pred_counts[1].item()}")
    print(f"  Prediction ratio: {pred_counts[1].item()/preds.shape[0]*100:.1f}% predict class 1")

    # Confusion matrix
    tp = ((preds == 1) & (y_dev == 1)).sum().item()
    tn = ((preds == 0) & (y_dev == 0)).sum().item()
    fp = ((preds == 1) & (y_dev == 0)).sum().item()
    fn = ((preds == 0) & (y_dev == 1)).sum().item()
    print(f"  Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  True positive rate (sensitivity): {tp/(tp+fn)*100:.1f}%" if (tp+fn) > 0 else "  N/A")
    print(f"  True negative rate (specificity): {tn/(tn+fp)*100:.1f}%" if (tn+fp) > 0 else "  N/A")

    # Check for constant classifier
    if pred_counts[0].item() == 0 or pred_counts[1].item() == 0:
        print(f"  WARNING: Degenerate classifier — always predicts class {preds[0].item()}")
        return True

    return False


# ============================================================================
# SECTION 5: MATHEMATICAL ANALYSIS OF THE STEP-SIZE BUG
# ============================================================================
def analyze_stepsize_math():
    """
    Mathematical proof that alpha=0.01 cannot flip a one-hot component.

    For a one-hot vector like [0, 0, 1], the gradient sign for each
    component is +1 or -1. After one step:
      component_i = original_i + alpha * sign(grad_i)

    For the "hot" component (value=1.0):
      worst case: 1.0 + 0.01 * (-1) = 0.99

    For a "cold" component (value=0.0):
      best case: 0.0 + 0.01 * (+1) = 0.01

    argmax([0.01, 0.01, 0.99]) = 2 (no flip)

    After N steps of alpha=0.01:
      hot component worst case: 1.0 - N*0.01
      cold component best case: 0.0 + N*0.01

    Flip requires: cold > hot => N*0.01 > 1.0 - N*0.01 => N > 50

    But with L_inf projection (epsilon=0.15), max perturbation is 0.15:
      hot component worst case: 1.0 - 0.15 = 0.85
      cold component best case: 0.0 + 0.15 = 0.15
      argmax still picks the hot component.

    HOWEVER: the code in pgd.py does NOT apply L_inf projection to
    categorical features. Let's check what actually happens.
    """
    print("\n" + "=" * 70)
    print("MATHEMATICAL ANALYSIS: Step-Size Threshold for Category Flips")
    print("=" * 70)

    # Simulate what happens to a one-hot [1, 0, 0] with alpha=0.01
    # and gradient sign pointing away from hot component
    print("\n  Simulating one-hot [1, 0, 0] with worst-case gradient:")
    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
        # Worst case: grad pushes hot down, cold up
        hot_val = 1.0 + alpha * (-1)  # gradient wants to decrease hot
        cold_val = 0.0 + alpha * (1)   # gradient wants to increase cold

        after = [cold_val, cold_val, hot_val]
        argmax = np.argmax(after)
        flipped = argmax != 2
        print(f"    alpha={alpha:.2f}: after_step=[{cold_val:.2f}, {cold_val:.2f}, {hot_val:.2f}] "
              f"argmax={argmax} {'FLIPPED!' if flipped else 'no flip'}")

    # Now check what the ACTUAL code does
    print("\n  Checking actual code path in pgd.py (lines 47-58):")
    print("    The code does: adv_cat = images[:, cat_group] + alpha * grad[:, cat_group].sign()")
    print("    Then: nearest_idx = torch.argmax(adv_cat, dim=1)")
    print("    Note: NO L_inf projection is applied to categorical features in pgd.py")
    print("    Note: The code re-reads from images (which was snapped last step) not orig")

    # Key: in pgd.py, the categorical update reads from `images` (line 49),
    # which was snapped to one-hot at the end of the previous step.
    # So each step starts from a valid one-hot and adds alpha*sign(grad).
    # For alpha=0.01, the perturbation is too small to cross the argmax boundary.
    print("\n  Since each step starts from a snapped one-hot [1, 0, ..., 0]:")
    print("    - alpha=0.01: hot_comp=0.99 or 1.01, cold_comp=-0.01 or 0.01")
    print("    - argmax ALWAYS returns the hot component (0.99 > 0.01)")
    print("    - Therefore: ZERO flips possible with alpha=0.01")
    print("    - Minimum alpha for possible flip: 0.5 (for 3-way one-hot)")
    print("    - For 11-way one-hot: any alpha > 0 could theoretically flip")
    print("      if gradient is aligned, but 0.01 is still too small")


# ============================================================================
# SECTION 6: MAIN VERIFICATION PIPELINE
# ============================================================================
def main():
    print("=" * 70)
    print("INDEPENDENT VERIFICATION — BlackICE-Mesh DACM Results")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Step 1: Load and verify data
    print("\n" + "=" * 70)
    print("STEP 1: DATA VERIFICATION")
    print("=" * 70)
    X, y, n_samples = load_test_data()

    # Step 2: Load and verify models
    print("\n" + "=" * 70)
    print("STEP 2: MODEL LOADING AND VERIFICATION")
    print("=" * 70)
    print("\n--- Loading baseline model ---")
    baseline = load_model(BASELINE_WEIGHTS)
    print("\n--- Loading hardened model ---")
    hardened = load_model(HARDENED_WEIGHTS)

    # Step 3: Clean accuracy
    print("\n" + "=" * 70)
    print("STEP 3: CLEAN ACCURACY VERIFICATION")
    print("=" * 70)
    baseline_clean = verify_clean_accuracy(baseline, X, y, "Baseline")
    hardened_clean = verify_clean_accuracy(hardened, X, y, "FGSM-Hardened")

    # Step 3b: Detailed prediction analysis
    verify_model_predictions_detailed(baseline, X, y, "Baseline")
    verify_model_predictions_detailed(hardened, X, y, "FGSM-Hardened")

    # Step 4: Mathematical analysis
    analyze_stepsize_math()

    # Step 5: Bug 1 — Categorical step size
    verify_bug1_categorical_stepsize(hardened, X, y)

    # Step 6: Bug 2 — Phantom packets
    verify_bug2_phantom_packets(hardened, X, y)

    # Step 7: Cross-check with original code
    verify_original_pgd_matches(hardened, X, y)

    # Step 8: Full robust accuracy — both models, both alpha_cat values
    print("\n" + "=" * 70)
    print("STEP 8: FULL TEST SET ROBUST ACCURACY")
    print("=" * 70)

    print("\n--- With alpha_cat=0.01 (original, buggy) ---")
    baseline_robust_buggy = verify_robust_accuracy_full(baseline, X, y, "Baseline", alpha_cat=0.01)
    hardened_robust_buggy = verify_robust_accuracy_full(hardened, X, y, "FGSM-Hardened", alpha_cat=0.01)

    print("\n--- With alpha_cat=1.0 (fixed) ---")
    baseline_robust_fixed = verify_robust_accuracy_full(baseline, X, y, "Baseline", alpha_cat=1.0)
    hardened_robust_fixed = verify_robust_accuracy_full(hardened, X, y, "FGSM-Hardened", alpha_cat=1.0)

    print("\n--- Unconstrained PGD (no DACM snapping) ---")
    baseline_robust_uncon = verify_robust_accuracy_unconstrained_full(baseline, X, y, "Baseline")
    hardened_robust_uncon = verify_robust_accuracy_unconstrained_full(hardened, X, y, "FGSM-Hardened")

    # Step 9: Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Clean':>8} {'Robust(α_c=0.01)':>18} {'Robust(α_c=1.0)':>17} {'Unconstrained':>15}")
    print("-" * 85)
    print(f"{'Baseline':<25} {baseline_clean:>7.2f}% {baseline_robust_buggy:>17.2f}% {baseline_robust_fixed:>16.2f}% {baseline_robust_uncon:>14.2f}%")
    print(f"{'FGSM-Hardened':<25} {hardened_clean:>7.2f}% {hardened_robust_buggy:>17.2f}% {hardened_robust_fixed:>16.2f}% {hardened_robust_uncon:>14.2f}%")
    print("-" * 85)

    print("\n[VERIFICATION COMPLETE]")


if __name__ == "__main__":
    main()
