#!/usr/bin/env python3
"""
DACM Replication Test — preliminary baseline verification before the Rust/Go
microservices migration.

Hooks directly into the retained NSL-KDD loaders (app.ml.data.loader) and the
persisted checkpoint layout. No placeholder data is generated: the full
nsl-kdd-test.csv set is evaluated through the original weight snapshots.

Run from ml-optimizer/:
    python dacm_replication_test.py
    python dacm_replication_test.py --samples 2000 --batch-size 500
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from app.ml.data.loader import (
    CATEGORICAL_GROUPS,
    CONTINUOUS_COLS,
    FEATURE_DIM,
    load_tabular_data,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_CSV = "./data/nsl-kdd-test.csv"
BASELINE_WEIGHTS = "models/model.pth"
HARDENED_WEIGHTS = "models/model_adv.pth"

MIN_VAL, MAX_VAL = 0.0, 1.0


# =============================================================================
# Step 1 — Data Preprocessing & Bounds Extraction
# =============================================================================
def extract_categorical_bounds():
    """
    Pull the categorical feature indices and their structurally valid
    mathematical states post-One-Hot encoding directly from the retained
    loader's CATEGORICAL_GROUPS topology (NSL-KDD: 3 protocol + 11 service).
    Each valid state is a simplex vertex (unit vector) of the one-hot space.
    """
    cat_idx = [i for group in CATEGORICAL_GROUPS for i in group]
    bounds = [torch.eye(len(group), dtype=torch.float32) for group in CATEGORICAL_GROUPS]
    return cat_idx, bounds


def build_test_loader(batch_size, max_samples=None):
    """Full NSL-KDD test set through the retained ingestion module."""
    dataset = load_tabular_data(TEST_CSV)
    if max_samples is not None and max_samples < len(dataset):
        x, y = dataset.tensors
        dataset = TensorDataset(x[:max_samples], y[:max_samples])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), len(dataset)


# =============================================================================
# Step 2 — Model Instantiation
# =============================================================================
class ControlMLP(nn.Module):
    """
    Original control architecture: 4-layer MLP (input + 3 linear stages) with
    ReLU activations and a 0.3 dropout threshold.

    Layer names mirror the persisted TabularMLP checkpoint layout
    (fc1 -> bn1 -> fc2 -> fc3) so the retained weights load with strict=True.
    Dropout is a no-op in eval mode and therefore does not skew evaluation.
    """

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


def load_weights(path):
    """Loading hook: instantiate the control MLP and inject the checkpoint."""
    model = ControlMLP().to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# =============================================================================
# Step 3 — The DACM + PGD Attack Loop
# =============================================================================
@torch.no_grad()
def dacm_snap_categorical(adv_batch, group_idx, valid_states):
    """
    DACM snapping: isolate the categorical features by index and force the
    continuous adversarial gradient onto the L2-nearest structurally valid
    state. On a one-hot basis this reduces to argmax, but the full L2 distance
    computation is kept for a faithful replication of the paper's definition.
    """
    sub = adv_batch[:, group_idx]                 # (B, G)
    states = valid_states.to(sub.device)          # (V, G)
    diff = sub[:, None, :] - states[None, :, :]   # (B, V, G)
    dist = diff.square().sum(dim=-1)              # (B, V)
    nearest = dist.argmin(dim=-1)                 # (B,)
    return states[nearest]


class SnapBPDA(torch.autograd.Function):
    """
    Backward Pass Differentiable Approximation (BPDA) for DACM snapping.
    Forward: applies the hard categorical snap (non-differentiable argmin).
    Backward: straight-through estimator — identity function passing
    continuous gradients straight through the non-differentiable snap.
    """
    @staticmethod
    def forward(ctx, sub, states):
        states = states.to(sub.device)
        with torch.no_grad():
            diff = sub[:, None, :] - states[None, :, :]
            dist = diff.square().sum(dim=-1)
            nearest = dist.argmin(dim=-1)
            snapped = states[nearest]
        # Force gradient tracking so autograd invokes backward()
        snapped = snapped + 0 * sub
        ctx.save_for_backward(sub)
        return snapped

    @staticmethod
    def backward(ctx, grad_output):
        sub, = ctx.saved_tensors
        # Straight-through: pass gradient directly to input
        return grad_output, None


def pgd_dacm_attack(model, images, labels, epsilon, alpha, steps, bounds,
                    snap_times):
    """
    PGD multi-step attack (steps=40, alpha=0.01). Continuous columns receive a
    signed gradient step, L_inf projection and Min-Max clamp. Categorical
    columns are re-snapped to valid states by DACM at every iteration.
    """
    orig = images.clone().detach()
    adv = images.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    cat_idx = [i for group in CATEGORICAL_GROUPS for i in group]

    for step in range(steps):
        adv.requires_grad_(True)
        out = model(adv)
        loss = loss_fn(out, labels)
        model.zero_grad()
        loss.backward()
        grad = adv.grad

        if step < 5:
            mean_abs_grad = grad.abs().mean().item()
            cat_grad_mean = grad[:, cat_idx].abs().mean().item() if cat_idx else 0.0
            print(f"    [DEBUG] Step {step+1}: mean |grad| = {mean_abs_grad:.6f}, "
                  f"cat |grad| = {cat_grad_mean:.6f}")
            if cat_grad_mean == 0.0:
                print(f"    [WARNING] Step {step+1}: categorical gradient is exactly 0.0 — "
                      f"possible gradient masking from hard snap!")

        with torch.no_grad():
            # Continuous: signed gradient step -> L_inf ball -> Min-Max clamp
            adv_cont = adv[:, CONTINUOUS_COLS] + alpha * grad[:, CONTINUOUS_COLS].sign()
            eta = (adv_cont - orig[:, CONTINUOUS_COLS]).clamp(-epsilon, epsilon)
            adv.data[:, CONTINUOUS_COLS] = (orig[:, CONTINUOUS_COLS] + eta).clamp(MIN_VAL, MAX_VAL)

            # Categorical: gradient step -> L_inf ball -> DACM snap
            for group_idx, valid_states in zip(CATEGORICAL_GROUPS, bounds):
                adv_cat = adv[:, group_idx] + alpha * grad[:, group_idx].sign()
                eta_cat = (adv_cat - orig[:, group_idx]).clamp(-epsilon, epsilon)
                adv_cat_proj = (orig[:, group_idx] + eta_cat).clamp(MIN_VAL, MAX_VAL)
                t0 = time.perf_counter()
                snapped = dacm_snap_categorical(adv, group_idx, valid_states)
                snap_times["total"] += time.perf_counter() - t0
                snap_times["calls"] += 1
                adv.data[:, group_idx] = snapped

        adv = adv.detach()

    return adv


def unconstrained_pgd_attack(model, images, labels, epsilon, alpha, steps):
    """
    Unconstrained PGD: identical to standard PGD but ABSOLUTELY NO DACM
    snapping is applied, either inside the loop or at the end. Categorical
    features are left at their continuous adversarial values, producing
    physically impossible network packets.
    """
    orig = images.clone().detach()
    adv = images.clone().detach()
    loss_fn = nn.CrossEntropyLoss()

    all_cols = CONTINUOUS_COLS + [i for group in CATEGORICAL_GROUPS for i in group]
    if all_cols:
        random_noise = torch.empty_like(orig[:, all_cols]).uniform_(-epsilon, epsilon)
        adv[:, all_cols] = torch.clamp(orig[:, all_cols] + random_noise, MIN_VAL, MAX_VAL)

    for _ in range(steps):
        adv.requires_grad_(True)
        out = model(adv)
        loss = loss_fn(out, labels)
        model.zero_grad()
        loss.backward()
        grad = adv.grad

        with torch.no_grad():
            # Continuous: signed gradient step -> L_inf ball -> Min-Max clamp
            adv_cont = adv[:, CONTINUOUS_COLS] + alpha * grad[:, CONTINUOUS_COLS].sign()
            eta = (adv_cont - orig[:, CONTINUOUS_COLS]).clamp(-epsilon, epsilon)
            adv.data[:, CONTINUOUS_COLS] = (orig[:, CONTINUOUS_COLS] + eta).clamp(MIN_VAL, MAX_VAL)

            # Categorical: signed gradient step -> L_inf ball, NO snapping
            for group_idx in CATEGORICAL_GROUPS:
                adv_cat = adv[:, group_idx] + alpha * grad[:, group_idx].sign()
                eta_cat = (adv_cat - orig[:, group_idx]).clamp(-epsilon, epsilon)
                adv.data[:, group_idx] = (orig[:, group_idx] + eta_cat).clamp(MIN_VAL, MAX_VAL)

        adv = adv.detach()

    return adv


def pgd_bpda_attack(model, images, labels, epsilon, alpha, steps, bounds,
                    snap_times):
    """
    PGD multi-step attack with BPDA (Straight-Through Estimator) for DACM
    snapping. The hard categorical snap is applied in the forward path
    (model receives snapped inputs), and gradients flow through via the
    identity backward pass.
    """
    orig = images.clone().detach()
    adv = images.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    cat_idx = [i for group in CATEGORICAL_GROUPS for i in group]

    for step in range(steps):
        adv.requires_grad_(True)

        # Apply BPDA snap to categorical features BEFORE model forward
        adv_snapped = adv.clone()
        with torch.no_grad():
            for group_idx, valid_states in zip(CATEGORICAL_GROUPS, bounds):
                t0 = time.perf_counter()
                snapped = SnapBPDA.apply(adv[:, group_idx], valid_states)
                snap_times["total"] += time.perf_counter() - t0
                snap_times["calls"] += 1
                adv_snapped[:, group_idx] = snapped

        out = model(adv_snapped)
        loss = loss_fn(out, labels)
        model.zero_grad()
        loss.backward()
        grad = adv.grad

        if step < 5:
            mean_abs_grad = grad.abs().mean().item()
            cat_grad_mean = grad[:, cat_idx].abs().mean().item() if cat_idx else 0.0
            print(f"    [DEBUG-BPDA] Step {step+1}: mean |grad| = {mean_abs_grad:.6f}, "
                  f"cat |grad| = {cat_grad_mean:.6f}")
            if cat_grad_mean == 0.0:
                print(f"    [WARNING-BPDA] Step {step+1}: categorical gradient is exactly 0.0 — "
                      f"possible gradient masking from hard snap!")

        with torch.no_grad():
            # Continuous: signed gradient step -> L_inf ball -> Min-Max clamp
            adv_cont = adv[:, CONTINUOUS_COLS] + alpha * grad[:, CONTINUOUS_COLS].sign()
            eta = (adv_cont - orig[:, CONTINUOUS_COLS]).clamp(-epsilon, epsilon)
            adv.data[:, CONTINUOUS_COLS] = (orig[:, CONTINUOUS_COLS] + eta).clamp(MIN_VAL, MAX_VAL)
            # Categorical: BPDA handles snapping in forward pass above

        adv = adv.detach()

    return adv


def evaluate_clean(model, loader):
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total


def evaluate_robust(model, loader, epsilon, alpha, steps, bounds, snap_times):
    correct = total = 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        adv = pgd_dacm_attack(model, data, target, epsilon, alpha, steps, bounds,
                              snap_times)
        with torch.no_grad():
            pred = model(adv).argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="DACM baseline replication test")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="run on cuda, cpu, or auto (cuda if available)")
    parser.add_argument("--samples", type=int, default=None,
                        help="limit test samples (default: full NSL-KDD test set)")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.15,
                        help="perturbation budget (max)")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    global DEVICE
    args = parse_args()
    if args.device == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(args.device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) if DEVICE.type == "cuda" else None

    cat_idx, bounds = extract_categorical_bounds()
    loader, n_samples = build_test_loader(args.batch_size, args.samples)
    n_samples = min(n_samples, args.samples) if args.samples else n_samples

    baseline_model = load_weights(BASELINE_WEIGHTS)
    hardened_model = load_weights(HARDENED_WEIGHTS)

    print("=" * 64)
    print("DACM REPLICATION TEST — BlackICE-Mesh baseline verification")
    print("=" * 64)
    print(f"device          : {DEVICE}")
    print(f"test set        : {n_samples} samples (batch={args.batch_size})")
    print(f"attack config   : PGD steps={args.steps} alpha={args.alpha} eps={args.epsilon}")
    print(f"categorical idx : {cat_idx}")
    print(f"valid states    : {[b.shape[0] for b in bounds]} per group "
          f"(groups={[len(g) for g in CATEGORICAL_GROUPS]})")
    print("-" * 64)

    clean_acc = evaluate_clean(baseline_model, loader)
    print(f"[1] CLEAN ACCURACY (baseline)             : {clean_acc * 100:.2f}%  (target ~81.10%)")

    # --- Step A: Unconstrained PGD (no snapping) ---
    print("\n>>> RUNNING UNCONSTRAINED PGD (no DACM snapping)")
    unconstrained_start = time.perf_counter()
    # Use a small batch for speed in diagnostics
    small_loader, _ = build_test_loader(64, 64)
    batch_data, batch_target = next(iter(small_loader))
    batch_data, batch_target = batch_data.to(DEVICE), batch_target.to(DEVICE)
    unconstrained_adv = unconstrained_pgd_attack(hardened_model, batch_data, batch_target,
                                                 args.epsilon, args.alpha, args.steps)
    with torch.no_grad():
        unconstrained_pred = hardened_model(unconstrained_adv).argmax(dim=1)
    unconstrained_acc = (unconstrained_pred == batch_target).float().mean().item()
    unconstrained_wall = time.perf_counter() - unconstrained_start
    print(f"[A] HARDENED ROBUST ACCURACY (UNCONSTRAINED) : {unconstrained_acc * 100:.2f}%  "
          f"(target ~29.10%)")
    print(f"    wall time (64 samples): {unconstrained_wall:.4f}s")

    # --- Step B: BPDA-constrained PGD (gradient diagnostics) ---
    print("\n>>> RUNNING BPDA-CONSTRAINED PGD (gradient diagnostics)")
    bpda_snap_times = {"total": 0.0, "calls": 0}
    bpda_adv = pgd_bpda_attack(hardened_model, batch_data, batch_target,
                               args.epsilon, args.alpha, args.steps, bounds,
                               bpda_snap_times)
    with torch.no_grad():
        bpda_pred = hardened_model(bpda_adv).argmax(dim=1)
    bpda_acc = (bpda_pred == batch_target).float().mean().item()
    print(f"[B] HARDENED ROBUST ACCURACY (BPDA)        : {bpda_acc * 100:.2f}%  "
          f"(target ~29.10% if gradient masking exists)")

    # --- Step C: Original constrained PGD for reference ---
    print("\n>>> RUNNING ORIGINAL CONSTRAINED PGD (reference)")
    snap_times = {"total": 0.0, "calls": 0}
    attack_start = time.perf_counter()

    baseline_robust = evaluate_robust(baseline_model, loader, args.epsilon,
                                      args.alpha, args.steps, bounds, snap_times)
    print(f"[C] BASELINE ROBUST ACCURACY (PGD)        : {baseline_robust * 100:.2f}%  (target ~12.30%)")

    hardened_robust = evaluate_robust(hardened_model, loader, args.epsilon,
                                      args.alpha, args.steps, bounds, snap_times)
    print(f"[D] HARDENED ROBUST ACCURACY (PGD)        : {hardened_robust * 100:.2f}%  (target ~29.10%)")

    attack_wall = time.perf_counter() - attack_start
    avg_snap_ms_per_payload = (snap_times["total"] / n_samples) * 1000.0
    avg_snap_ms_per_call = (snap_times["total"] / snap_times["calls"]) * 1000.0

    print(f"[4] AVG DACM SNAP LATENCY                 : {avg_snap_ms_per_payload:.2f} ms/payload "
          f"({avg_snap_ms_per_call:.3f} ms/call, target ~15-25ms)")
    print(f"    DACM calls issued: {snap_times['calls']} | attack wall time: {attack_wall:.2f}s")
    print("-" * 64)

    clean_correct = clean_acc * n_samples
    print(f"ASR baseline  = {100 * (clean_acc - baseline_robust):.2f}%")
    print(f"ASR hardened  = {100 * (clean_acc - hardened_robust):.2f}%")
    print(f"robust gap    = {100 * (hardened_robust - baseline_robust):.2f}pp "
          f"({100 * clean_correct / n_samples:.0f} clean-correct samples)")
    print("=" * 64)


if __name__ == "__main__":
    main()
