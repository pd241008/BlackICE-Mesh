#!/usr/bin/env python3
"""
GRADIENT MASKING & SHATTERING DIAGNOSTIC BATTERY
=================================================
Following Athalye et al. "Obfuscated Gradients Give a False Sense of Security"
(ICML 2018) and Carlini et al. "On Evaluating Adversarial Robustness" (2019).

Tests performed:
  1. Gradient magnitude analysis — are gradients vanishing or exploding?
  2. Gradient alignment test — does loss actually increase along the gradient direction?
  3. Loss landscape smoothness — is the loss landscape shattered/non-smooth?
  4. Random search comparison — does random perturbation work as well as gradient-based?
  5. Step-count sensitivity — does increasing PGD steps improve the attack?
  6. Transfer attack test — do adversarial examples from a substitute model transfer?
  7. DACM snap gradient destruction — does the argmax snap destroy gradient signal?
  8. Feature importance via gradient — which features actually matter?
  9. Iterative loss tracking — does loss monotonically increase during PGD?

Run from ml-optimizer/:
    python gradient_masking_audit.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_CSV = "./data/nsl-kdd-test.csv"
TRAIN_CSV = "./data/nsl-kdd-train.csv"
BASELINE_WEIGHTS = "app/ml/model.pth"
HARDENED_WEIGHTS = "app/ml/model_adv.pth"

CONTINUOUS_COLS = [0, 1, 2, 3]
PROTOCOL_GROUP = [4, 5, 6]
SERVICE_GROUP = list(range(7, 18))
CATEGORICAL_GROUPS = [PROTOCOL_GROUP, SERVICE_GROUP]
ALL_CAT_IDX = PROTOCOL_GROUP + SERVICE_GROUP
FEATURE_DIM = 18


class IndependentMLP(nn.Module):
    def __init__(self, input_dim=18, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_model(path):
    model = IndependentMLP().to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_data(csv_path, max_samples=None):
    data = np.loadtxt(csv_path, delimiter=',')
    if max_samples:
        data = data[:max_samples]
    X = torch.tensor(data[:, :18], dtype=torch.float32)
    y = torch.tensor(data[:, 18], dtype=torch.long)
    return X, y


# ============================================================================
# TEST 1: Gradient Magnitude Analysis
# ============================================================================
def test_gradient_magnitudes(model, X, y, name):
    """
    Check if gradients are vanishing, exploding, or healthy.
    Gradient masking symptom: gradients near zero on all/most features.
    """
    print(f"\n{'='*70}")
    print(f"TEST 1: GRADIENT MAGNITUDE ANALYSIS — {name}")
    print(f"{'='*70}")

    X_dev = X[:1000].to(DEVICE).requires_grad_(True)
    y_dev = y[:1000].to(DEVICE)

    out = model(X_dev)
    loss = F.cross_entropy(out, y_dev)
    model.zero_grad()
    loss.backward()

    grad = X_dev.grad.detach()

    # Overall stats
    print(f"  Overall gradient stats:")
    print(f"    Mean |grad|:  {grad.abs().mean():.6f}")
    print(f"    Max |grad|:   {grad.abs().max():.6f}")
    print(f"    Min |grad|:   {grad.abs().min():.6f}")
    print(f"    Std |grad|:   {grad.abs().std():.6f}")
    print(f"    % exactly 0:  {(grad == 0).float().mean()*100:.2f}%")

    # Per-feature group
    cont_grad = grad[:, CONTINUOUS_COLS]
    cat_grad = grad[:, ALL_CAT_IDX]
    proto_grad = grad[:, PROTOCOL_GROUP]
    svc_grad = grad[:, SERVICE_GROUP]

    print(f"\n  Per-feature-group gradient magnitudes:")
    print(f"    Continuous [0-3]:  mean={cont_grad.abs().mean():.6f}, max={cont_grad.abs().max():.6f}")
    print(f"    Protocol [4-6]:    mean={proto_grad.abs().mean():.6f}, max={proto_grad.abs().max():.6f}")
    print(f"    Service [7-17]:    mean={svc_grad.abs().mean():.6f}, max={svc_grad.abs().max():.6f}")

    # Per-feature breakdown
    print(f"\n  Per-feature mean |grad|:")
    for i in range(FEATURE_DIM):
        feat_grad = grad[:, i].abs().mean().item()
        zero_pct = (grad[:, i] == 0).float().mean().item() * 100
        label = "CONT" if i in CONTINUOUS_COLS else ("PROTO" if i in PROTOCOL_GROUP else "SVC")
        flag = " ← DEAD" if zero_pct > 99 else (" ← WEAK" if feat_grad < 1e-5 else "")
        print(f"    Feature {i:2d} ({label:5s}): {feat_grad:.8f} (zero: {zero_pct:.1f}%){flag}")

    # Gradient sign consistency
    # If gradients are noisy/shattered, the sign should be inconsistent across samples
    sign_consistency = []
    for i in range(FEATURE_DIM):
        signs = grad[:, i].sign()
        # Fraction of samples that agree with the majority sign
        pos_frac = (signs > 0).float().mean().item()
        neg_frac = (signs < 0).float().mean().item()
        consistency = max(pos_frac, neg_frac)
        sign_consistency.append(consistency)

    print(f"\n  Gradient sign consistency (fraction agreeing with majority):")
    print(f"    Continuous: {np.mean([sign_consistency[i] for i in CONTINUOUS_COLS]):.4f}")
    print(f"    Protocol:   {np.mean([sign_consistency[i] for i in PROTOCOL_GROUP]):.4f}")
    print(f"    Service:    {np.mean([sign_consistency[i] for i in SERVICE_GROUP]):.4f}")

    if cont_grad.abs().mean() < 1e-6:
        print(f"\n  ⚠️ WARNING: Continuous gradients near zero — possible gradient masking!")
    if cat_grad.abs().mean() < 1e-6:
        print(f"\n  ⚠️ WARNING: Categorical gradients near zero — possible gradient masking!")

    return grad


# ============================================================================
# TEST 2: Gradient Alignment (Does loss increase along gradient direction?)
# ============================================================================
def test_gradient_alignment(model, X, y, name):
    """
    If gradients are informative, moving along the gradient direction should
    increase loss. If loss decreases or doesn't change, gradients are misleading.
    This is the core test for gradient masking.
    """
    print(f"\n{'='*70}")
    print(f"TEST 2: GRADIENT ALIGNMENT — {name}")
    print(f"{'='*70}")

    X_dev = X[:500].to(DEVICE)
    y_dev = y[:500].to(DEVICE)

    # Get gradient
    X_dev.requires_grad_(True)
    out = model(X_dev)
    loss_orig = F.cross_entropy(out, y_dev)
    model.zero_grad()
    loss_orig.backward()
    grad = X_dev.grad.detach()

    loss_orig_val = loss_orig.item()

    # Move along gradient direction with various step sizes
    print(f"  Original loss: {loss_orig_val:.6f}")
    print(f"  Testing loss change along gradient direction:")

    step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    for step in step_sizes:
        with torch.no_grad():
            X_perturbed = X_dev.detach() + step * grad.sign()
            X_perturbed = X_perturbed.clamp(0, 1)
            out_pert = model(X_perturbed)
            loss_pert = F.cross_entropy(out_pert, y_dev).item()
            delta = loss_pert - loss_orig_val
            direction = "↑ GOOD" if delta > 0 else ("→ FLAT" if abs(delta) < 1e-4 else "↓ BAD")
            print(f"    step={step:.3f}: loss={loss_pert:.6f} (Δ={delta:+.6f}) {direction}")

    # Test on continuous features only vs all features
    print(f"\n  Gradient alignment — continuous features only:")
    for step in [0.01, 0.05, 0.1]:
        with torch.no_grad():
            X_pert2 = X_dev.detach().clone()
            X_pert2[:, CONTINUOUS_COLS] += step * grad[:, CONTINUOUS_COLS].sign()
            X_pert2 = X_pert2.clamp(0, 1)
            out_pert2 = model(X_pert2)
            loss_pert2 = F.cross_entropy(out_pert2, y_dev).item()
            delta2 = loss_pert2 - loss_orig_val
            direction2 = "↑ GOOD" if delta2 > 0 else ("→ FLAT" if abs(delta2) < 1e-4 else "↓ BAD")
            print(f"    step={step:.3f}: loss={loss_pert2:.6f} (Δ={delta2:+.6f}) {direction2}")

    # Test on categorical features only (with proper one-hot snapping)
    print(f"\n  Gradient alignment — categorical features with argmax snap:")
    for alpha_cat in [0.01, 0.1, 0.5, 1.0]:
        with torch.no_grad():
            X_pert3 = X_dev.detach().clone()
            for group in CATEGORICAL_GROUPS:
                adv_cat = X_pert3[:, group] + alpha_cat * grad[:, group].sign()
                nearest = torch.argmax(adv_cat, dim=1)
                snapped = F.one_hot(nearest, num_classes=len(group)).float()
                X_pert3[:, group] = snapped
            out_pert3 = model(X_pert3)
            loss_pert3 = F.cross_entropy(out_pert3, y_dev).item()
            delta3 = loss_pert3 - loss_orig_val
            direction3 = "↑ GOOD" if delta3 > 0 else ("→ FLAT" if abs(delta3) < 1e-4 else "↓ BAD")
            print(f"    alpha_cat={alpha_cat:.2f}: loss={loss_pert3:.6f} (Δ={delta3:+.6f}) {direction3}")


# ============================================================================
# TEST 3: Loss Landscape Smoothness (Shattering Detection)
# ============================================================================
def test_loss_smoothness(model, X, y, name):
    """
    Gradient shattering means the loss landscape is highly non-smooth.
    Test: compute loss at many points along a random direction and check
    if it's smooth or jagged.
    """
    print(f"\n{'='*70}")
    print(f"TEST 3: LOSS LANDSCAPE SMOOTHNESS — {name}")
    print(f"{'='*70}")

    X_dev = X[:200].to(DEVICE)
    y_dev = y[:200].to(DEVICE)

    # Pick a random direction
    torch.manual_seed(42)
    direction = torch.randn_like(X_dev)
    direction = direction / direction.norm()

    # Sample loss along this direction
    alphas = np.linspace(-0.3, 0.3, 61)
    losses = []
    for a in alphas:
        with torch.no_grad():
            X_shifted = (X_dev + a * direction).clamp(0, 1)
            out = model(X_shifted)
            loss = F.cross_entropy(out, y_dev).item()
            losses.append(loss)

    losses = np.array(losses)

    # Compute smoothness metrics
    first_diffs = np.diff(losses)
    second_diffs = np.diff(first_diffs)

    print(f"  Loss range along random direction: [{losses.min():.4f}, {losses.max():.4f}]")
    print(f"  Loss std: {losses.std():.6f}")
    print(f"  Mean |first diff|:  {np.abs(first_diffs).mean():.6f}")
    print(f"  Mean |second diff|: {np.abs(second_diffs).mean():.6f}")
    print(f"  Max |second diff|:  {np.abs(second_diffs).max():.6f}")

    # Smoothness ratio: |second_diff| / |first_diff|
    # For smooth functions, this should be small
    ratio = np.abs(second_diffs).mean() / (np.abs(first_diffs).mean() + 1e-10)
    print(f"  Smoothness ratio (|2nd|/|1st|): {ratio:.4f}")
    if ratio > 2.0:
        print(f"  ⚠️ WARNING: High smoothness ratio suggests loss landscape shattering!")
    else:
        print(f"  ✓ Loss landscape appears smooth (ratio < 2.0)")

    # Also check along the gradient direction specifically
    X_dev2 = X_dev.clone().requires_grad_(True)
    out2 = model(X_dev2)
    loss2 = F.cross_entropy(out2, y_dev)
    loss2.backward()
    grad_dir = X_dev2.grad.detach()
    grad_dir = grad_dir / (grad_dir.norm() + 1e-10)

    alphas_grad = np.linspace(0, 0.3, 31)
    losses_grad = []
    for a in alphas_grad:
        with torch.no_grad():
            X_shifted = (X_dev.detach() + a * grad_dir).clamp(0, 1)
            out = model(X_shifted)
            loss = F.cross_entropy(out, y_dev).item()
            losses_grad.append(loss)

    losses_grad = np.array(losses_grad)
    grad_first_diffs = np.diff(losses_grad)

    print(f"\n  Loss along gradient direction:")
    print(f"    Loss at 0:    {losses_grad[0]:.6f}")
    print(f"    Loss at 0.1:  {losses_grad[10]:.6f}")
    print(f"    Loss at 0.2:  {losses_grad[20]:.6f}")
    print(f"    Loss at 0.3:  {losses_grad[30]:.6f}")
    print(f"    Monotonically increasing: {all(d >= -1e-4 for d in grad_first_diffs)}")

    sign_changes = sum(1 for i in range(len(grad_first_diffs)-1)
                      if grad_first_diffs[i] * grad_first_diffs[i+1] < 0)
    print(f"    Sign changes in gradient direction: {sign_changes}/29")
    if sign_changes > 10:
        print(f"  ⚠️ WARNING: Many sign changes along gradient = shattered landscape!")
    else:
        print(f"  ✓ Few sign changes = smooth gradient direction")


# ============================================================================
# TEST 4: Random Search Comparison
# ============================================================================
def test_random_vs_gradient(model, X, y, name):
    """
    Classic gradient masking test (Athalye et al.):
    If random perturbations are as effective as gradient-based attacks,
    gradients are not providing useful information.
    """
    print(f"\n{'='*70}")
    print(f"TEST 4: RANDOM SEARCH vs GRADIENT-BASED ATTACK — {name}")
    print(f"{'='*70}")

    X_dev = X[:1000].to(DEVICE)
    y_dev = y[:1000].to(DEVICE)

    with torch.no_grad():
        clean_pred = model(X_dev).argmax(dim=1)
        clean_acc = (clean_pred == y_dev).float().mean().item() * 100

    print(f"  Clean accuracy: {clean_acc:.2f}%")

    for epsilon in [0.05, 0.10, 0.15, 0.20, 0.30]:
        # Gradient-based (single FGSM step)
        X_dev.requires_grad_(True)
        out = model(X_dev)
        loss = F.cross_entropy(out, y_dev)
        model.zero_grad()
        loss.backward()
        grad = X_dev.grad.detach()

        with torch.no_grad():
            # FGSM on all features
            adv_fgsm = (X_dev.detach() + epsilon * grad.sign()).clamp(0, 1)
            fgsm_acc = (model(adv_fgsm).argmax(dim=1) == y_dev).float().mean().item() * 100

            # Random perturbation (same budget, 10 random tries, take best)
            best_random_acc = clean_acc
            for _ in range(10):
                noise = torch.empty_like(X_dev).uniform_(-epsilon, epsilon)
                adv_rand = (X_dev.detach() + noise).clamp(0, 1)
                rand_acc = (model(adv_rand).argmax(dim=1) == y_dev).float().mean().item() * 100
                best_random_acc = min(best_random_acc, rand_acc)

        gap = fgsm_acc - best_random_acc
        flag = "⚠️ MASKING" if gap > -2 else "✓ OK"
        print(f"  ε={epsilon:.2f}: FGSM={fgsm_acc:.1f}%, Random(best of 10)={best_random_acc:.1f}%, "
              f"gap={gap:+.1f}pp {flag}")


# ============================================================================
# TEST 5: PGD Step-Count Sensitivity
# ============================================================================
def test_step_sensitivity(model, X, y, name):
    """
    If the attack is working properly, more PGD steps should not increase
    robust accuracy (should decrease or plateau). If it increases, there's
    likely gradient masking causing the attack to oscillate.
    """
    print(f"\n{'='*70}")
    print(f"TEST 5: PGD STEP-COUNT SENSITIVITY — {name}")
    print(f"{'='*70}")

    X_dev = X[:500].to(DEVICE)
    y_dev = y[:500].to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    for steps in [1, 5, 10, 20, 40, 80, 160]:
        # Run PGD with this many steps (continuous only, since that's what works)
        orig = X_dev.clone().detach()
        adv = X_dev.clone().detach()

        for step in range(steps):
            adv.requires_grad_(True)
            out = model(adv)
            loss = loss_fn(out, y_dev)
            model.zero_grad()
            loss.backward()
            grad = adv.grad.detach()

            with torch.no_grad():
                adv_full = adv.detach().clone()
                adv_cont = adv_full[:, CONTINUOUS_COLS] + 0.01 * grad[:, CONTINUOUS_COLS].sign()
                eta = (adv_cont - orig[:, CONTINUOUS_COLS]).clamp(-0.15, 0.15)
                adv_full[:, CONTINUOUS_COLS] = (orig[:, CONTINUOUS_COLS] + eta).clamp(0, 1)
                adv = adv_full

        with torch.no_grad():
            acc = (model(adv).argmax(dim=1) == y_dev).float().mean().item() * 100
            final_loss = F.cross_entropy(model(adv), y_dev).item()

        print(f"  steps={steps:3d}: robust_acc={acc:.2f}%, loss={final_loss:.4f}")


# ============================================================================
# TEST 6: Transfer Attack Test
# ============================================================================
def test_transfer_attack(model_target, X, y, name):
    """
    Classic gradient masking diagnostic: train a simple substitute model,
    generate adversarial examples from it, and test transferability.

    If gradient masking exists on the target model, transfer attacks from
    a substitute should be MORE effective than direct attacks on the target.
    """
    print(f"\n{'='*70}")
    print(f"TEST 6: TRANSFER ATTACK — {name}")
    print(f"{'='*70}")

    # Train a simple substitute model on the same data
    print("  Training substitute model...")
    X_train, y_train = load_data(TRAIN_CSV, max_samples=5000)

    substitute = IndependentMLP().to(DEVICE)
    optimizer = torch.optim.Adam(substitute.parameters(), lr=1e-3)
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    substitute.train()
    for epoch in range(20):
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(substitute(bx), by)
            loss.backward()
            optimizer.step()

    substitute.eval()

    X_dev = X[:1000].to(DEVICE)
    y_dev = y[:1000].to(DEVICE)

    # Clean accuracy of both models
    with torch.no_grad():
        target_clean = (model_target(X_dev).argmax(dim=1) == y_dev).float().mean().item() * 100
        sub_clean = (substitute(X_dev).argmax(dim=1) == y_dev).float().mean().item() * 100
    print(f"  Target clean acc:     {target_clean:.2f}%")
    print(f"  Substitute clean acc: {sub_clean:.2f}%")

    for epsilon in [0.10, 0.15, 0.20]:
        # Direct attack on target
        X_dev.requires_grad_(True)
        out_t = model_target(X_dev)
        loss_t = F.cross_entropy(out_t, y_dev)
        model_target.zero_grad()
        loss_t.backward()
        grad_t = X_dev.grad.detach()

        with torch.no_grad():
            adv_direct = (X_dev.detach() + epsilon * grad_t.sign()).clamp(0, 1)
            direct_acc = (model_target(adv_direct).argmax(dim=1) == y_dev).float().mean().item() * 100

        # Transfer attack: generate from substitute, test on target
        X_dev2 = X_dev.detach().clone().requires_grad_(True)
        out_s = substitute(X_dev2)
        loss_s = F.cross_entropy(out_s, y_dev)
        substitute.zero_grad()
        loss_s.backward()
        grad_s = X_dev2.grad.detach()

        with torch.no_grad():
            adv_transfer = (X_dev.detach() + epsilon * grad_s.sign()).clamp(0, 1)
            transfer_acc = (model_target(adv_transfer).argmax(dim=1) == y_dev).float().mean().item() * 100

        gap = transfer_acc - direct_acc
        flag = "⚠️ MASKING SIGNAL" if gap < -5 else "✓ OK"
        print(f"  ε={epsilon:.2f}: direct={direct_acc:.1f}%, transfer={transfer_acc:.1f}%, "
              f"gap={gap:+.1f}pp {flag}")


# ============================================================================
# TEST 7: DACM Snap Gradient Destruction Analysis
# ============================================================================
def test_dacm_gradient_destruction(model, X, y, name):
    """
    The DACM argmax snap is a non-differentiable operation applied INSIDE
    the PGD loop. Even though gradients are computed before the snap,
    the snap modifies the tensor in-place via .data, which means:
    1. The gradient at the current point reflects the pre-snap input
    2. After snapping, we're at a different point
    3. The gradient we used is stale w.r.t. the post-snap position

    This test measures how much the snap moves the input and whether
    the gradient at the snapped point agrees with the pre-snap gradient.
    """
    print(f"\n{'='*70}")
    print(f"TEST 7: DACM SNAP GRADIENT DESTRUCTION — {name}")
    print(f"{'='*70}")

    X_dev = X[:500].to(DEVICE)
    y_dev = y[:500].to(DEVICE)

    # Compute gradient at clean input
    X_dev.requires_grad_(True)
    out1 = model(X_dev)
    loss1 = F.cross_entropy(out1, y_dev)
    model.zero_grad()
    loss1.backward()
    grad_before_snap = X_dev.grad.detach().clone()

    # Apply one PGD step + snap, then compute gradient again
    with torch.no_grad():
        X_stepped = X_dev.detach().clone()
        for group in CATEGORICAL_GROUPS:
            adv_cat = X_stepped[:, group] + 1.0 * grad_before_snap[:, group].sign()
            nearest = torch.argmax(adv_cat, dim=1)
            snapped = F.one_hot(nearest, num_classes=len(group)).float()
            X_stepped[:, group] = snapped

    X_stepped.requires_grad_(True)
    out2 = model(X_stepped)
    loss2 = F.cross_entropy(out2, y_dev)
    model.zero_grad()
    loss2.backward()
    grad_after_snap = X_stepped.grad.detach()

    # Compare gradients
    # Cosine similarity between pre-snap and post-snap gradients
    cos_sim_all = F.cosine_similarity(
        grad_before_snap.view(X_dev.shape[0], -1),
        grad_after_snap.view(X_dev.shape[0], -1),
        dim=1
    )

    # Per-group cosine similarity
    cos_sim_cont = F.cosine_similarity(
        grad_before_snap[:, CONTINUOUS_COLS],
        grad_after_snap[:, CONTINUOUS_COLS],
        dim=1
    )

    print(f"  Cosine similarity between pre-snap and post-snap gradients:")
    print(f"    All features:  mean={cos_sim_all.mean():.4f}, std={cos_sim_all.std():.4f}")
    print(f"    Continuous:    mean={cos_sim_cont.mean():.4f}, std={cos_sim_cont.std():.4f}")

    # How much did the snap move the input?
    snap_movement = (X_stepped.detach() - X_dev.detach()).abs()
    print(f"\n  Input movement from snap:")
    print(f"    Categorical features L1: {snap_movement[:, ALL_CAT_IDX].sum(dim=1).mean():.4f}")
    print(f"    Continuous features L1:  {snap_movement[:, CONTINUOUS_COLS].sum(dim=1).mean():.4f}")

    # Sign agreement
    sign_agree_all = (grad_before_snap.sign() == grad_after_snap.sign()).float().mean()
    sign_agree_cont = (grad_before_snap[:, CONTINUOUS_COLS].sign() ==
                       grad_after_snap[:, CONTINUOUS_COLS].sign()).float().mean()
    sign_agree_cat = (grad_before_snap[:, ALL_CAT_IDX].sign() ==
                      grad_after_snap[:, ALL_CAT_IDX].sign()).float().mean()

    print(f"\n  Gradient sign agreement after snap:")
    print(f"    All features:  {sign_agree_all:.4f}")
    print(f"    Continuous:    {sign_agree_cont:.4f}")
    print(f"    Categorical:   {sign_agree_cat:.4f}")

    if cos_sim_all.mean() < 0.3:
        print(f"\n  ⚠️ WARNING: Low gradient cosine similarity after snap!")
        print(f"  The DACM snap is significantly disrupting the gradient landscape.")
    if sign_agree_cont < 0.7:
        print(f"\n  ⚠️ WARNING: Even continuous feature gradients change sign after snap!")
        print(f"  The categorical snap is destabilizing gradients for ALL features.")


# ============================================================================
# TEST 8: Feature Importance via Integrated Gradients
# ============================================================================
def test_feature_importance(model, X, y, name):
    """
    Which features actually drive the model's decisions?
    If the model relies mostly on categorical features but the attack
    can't perturb them, the attack is fundamentally limited.
    """
    print(f"\n{'='*70}")
    print(f"TEST 8: FEATURE IMPORTANCE — {name}")
    print(f"{'='*70}")

    X_dev = X[:1000].to(DEVICE)
    y_dev = y[:1000].to(DEVICE)

    # Simple gradient-based importance (mean |grad| per feature)
    X_dev.requires_grad_(True)
    out = model(X_dev)
    loss = F.cross_entropy(out, y_dev)
    model.zero_grad()
    loss.backward()
    grad = X_dev.grad.detach()

    importance = grad.abs().mean(dim=0).cpu().numpy()
    total_importance = importance.sum()

    print(f"  Feature importance (fraction of total |grad|):")
    cont_imp = importance[CONTINUOUS_COLS].sum() / total_importance * 100
    proto_imp = importance[PROTOCOL_GROUP].sum() / total_importance * 100
    svc_imp = importance[SERVICE_GROUP].sum() / total_importance * 100

    print(f"    Continuous [0-3]:  {cont_imp:.1f}%")
    print(f"    Protocol [4-6]:   {proto_imp:.1f}%")
    print(f"    Service [7-17]:   {svc_imp:.1f}%")

    # Weight-based importance (first layer weights)
    with torch.no_grad():
        w1 = model.fc1.weight.abs().mean(dim=0).cpu().numpy()  # [18]
        w1_total = w1.sum()

    print(f"\n  First-layer weight importance (fraction of total |w|):")
    w_cont = w1[CONTINUOUS_COLS].sum() / w1_total * 100
    w_proto = w1[PROTOCOL_GROUP].sum() / w1_total * 100
    w_svc = w1[SERVICE_GROUP].sum() / w1_total * 100
    print(f"    Continuous [0-3]:  {w_cont:.1f}%")
    print(f"    Protocol [4-6]:   {w_proto:.1f}%")
    print(f"    Service [7-17]:   {w_svc:.1f}%")

    # Feature ablation test
    print(f"\n  Feature ablation (set to zero, measure accuracy drop):")
    with torch.no_grad():
        clean_acc = (model(X_dev).argmax(dim=1) == y_dev).float().mean().item() * 100

        # Zero out continuous
        X_no_cont = X_dev.detach().clone()
        X_no_cont[:, CONTINUOUS_COLS] = 0
        acc_no_cont = (model(X_no_cont).argmax(dim=1) == y_dev).float().mean().item() * 100

        # Zero out categorical
        X_no_cat = X_dev.detach().clone()
        X_no_cat[:, ALL_CAT_IDX] = 0
        acc_no_cat = (model(X_no_cat).argmax(dim=1) == y_dev).float().mean().item() * 100

        # Randomize continuous
        X_rand_cont = X_dev.detach().clone()
        X_rand_cont[:, CONTINUOUS_COLS] = torch.rand_like(X_rand_cont[:, CONTINUOUS_COLS])
        acc_rand_cont = (model(X_rand_cont).argmax(dim=1) == y_dev).float().mean().item() * 100

        # Randomize categorical (valid one-hot)
        X_rand_cat = X_dev.detach().clone()
        for group in CATEGORICAL_GROUPS:
            rand_idx = torch.randint(0, len(group), (X_dev.shape[0],), device=DEVICE)
            X_rand_cat[:, group] = F.one_hot(rand_idx, num_classes=len(group)).float()
        acc_rand_cat = (model(X_rand_cat).argmax(dim=1) == y_dev).float().mean().item() * 100

    print(f"    Clean:               {clean_acc:.2f}%")
    print(f"    Zero continuous:     {acc_no_cont:.2f}% (Δ={acc_no_cont-clean_acc:+.2f}pp)")
    print(f"    Zero categorical:    {acc_no_cat:.2f}% (Δ={acc_no_cat-clean_acc:+.2f}pp)")
    print(f"    Random continuous:   {acc_rand_cont:.2f}% (Δ={acc_rand_cont-clean_acc:+.2f}pp)")
    print(f"    Random categorical:  {acc_rand_cat:.2f}% (Δ={acc_rand_cat-clean_acc:+.2f}pp)")

    if abs(acc_rand_cat - clean_acc) > abs(acc_rand_cont - clean_acc):
        print(f"\n  ⚠️ Categorical features are MORE important than continuous!")
        print(f"  An attack that can't perturb categoricals is fundamentally weakened.")


# ============================================================================
# TEST 9: Iterative Loss Tracking During PGD
# ============================================================================
def test_iterative_loss_tracking(model, X, y, name):
    """
    Track loss at every PGD step. In a well-functioning attack:
    - Loss should generally increase (maximize loss = fool classifier)
    - Loss should plateau, not oscillate wildly

    If loss oscillates or decreases, gradient masking or shattering is present.
    """
    print(f"\n{'='*70}")
    print(f"TEST 9: ITERATIVE LOSS TRACKING — {name}")
    print(f"{'='*70}")

    X_dev = X[:500].to(DEVICE)
    y_dev = y[:500].to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    # Full PGD with loss tracking
    orig = X_dev.clone().detach()
    adv = X_dev.clone().detach()
    losses = []
    accs = []

    for step in range(80):
        # Record metrics before step
        with torch.no_grad():
            out_eval = model(adv)
            loss_eval = loss_fn(out_eval, y_dev).item()
            acc_eval = (out_eval.argmax(dim=1) == y_dev).float().mean().item() * 100
            losses.append(loss_eval)
            accs.append(acc_eval)

        # PGD step (all features)
        adv.requires_grad_(True)
        out = model(adv)
        loss = loss_fn(out, y_dev)
        model.zero_grad()
        loss.backward()
        grad = adv.grad.detach()

        with torch.no_grad():
            adv_new = adv.detach() + 0.01 * grad.sign()
            eta = (adv_new - orig).clamp(-0.15, 0.15)
            adv = (orig + eta).clamp(0, 1)

    # Report
    print(f"  Loss trajectory (every 10 steps):")
    for i in range(0, 80, 10):
        delta = losses[i] - losses[0] if i > 0 else 0
        print(f"    Step {i:3d}: loss={losses[i]:.4f} (Δ={delta:+.4f}), acc={accs[i]:.1f}%")

    # Oscillation check
    loss_diffs = np.diff(losses)
    sign_changes = sum(1 for i in range(len(loss_diffs)-1)
                      if loss_diffs[i] * loss_diffs[i+1] < 0)
    increases = (np.array(loss_diffs) > 0).sum()
    decreases = (np.array(loss_diffs) < 0).sum()

    print(f"\n  Loss dynamics:")
    print(f"    Total increases: {increases}/79")
    print(f"    Total decreases: {decreases}/79")
    print(f"    Sign changes:    {sign_changes}/78")
    print(f"    Overall delta:   {losses[-1]-losses[0]:+.4f}")

    if sign_changes > 50:
        print(f"  ⚠️ WARNING: Highly oscillatory loss — shattering or masking likely!")
    elif increases < 20:
        print(f"  ⚠️ WARNING: Loss rarely increases — gradient may not be useful!")
    else:
        print(f"  ✓ Loss generally increases along gradient direction")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("GRADIENT MASKING & SHATTERING DIAGNOSTIC BATTERY")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Following Athalye et al. 2018 / Carlini et al. 2019 methodology")

    torch.manual_seed(42)
    np.random.seed(42)

    X, y = load_data(TEST_CSV, max_samples=2000)

    models = {
        "Baseline": load_model(BASELINE_WEIGHTS),
        "FGSM-Hardened": load_model(HARDENED_WEIGHTS),
    }

    for name, model in models.items():
        print(f"\n\n{'#'*70}")
        print(f"# MODEL: {name}")
        print(f"{'#'*70}")

        test_gradient_magnitudes(model, X, y, name)
        test_gradient_alignment(model, X, y, name)
        test_loss_smoothness(model, X, y, name)
        test_random_vs_gradient(model, X, y, name)
        test_step_sensitivity(model, X, y, name)
        test_dacm_gradient_destruction(model, X, y, name)
        test_feature_importance(model, X, y, name)
        test_iterative_loss_tracking(model, X, y, name)

    # Transfer test uses both models
    print(f"\n\n{'#'*70}")
    print(f"# TRANSFER ATTACKS")
    print(f"{'#'*70}")
    test_transfer_attack(models["FGSM-Hardened"], X, y, "FGSM-Hardened (target)")

    print(f"\n\n{'='*70}")
    print("ALL DIAGNOSTIC TESTS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
