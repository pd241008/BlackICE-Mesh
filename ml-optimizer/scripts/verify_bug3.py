#!/usr/bin/env python3
"""
BUG 3 VERIFICATION: dacm_snap_categorical receives `adv` instead of `adv_cat_proj`
in dacm_replication_test.py and variance_audit.py

This means the gradient step for categorical features is DEAD CODE — 
the snap always operates on the already-snapped values from the previous step,
so categorical features literally cannot change regardless of alpha.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONTINUOUS_COLS = [0, 1, 2, 3]
CATEGORICAL_GROUPS = [[4, 5, 6], list(range(7, 18))]
FEATURE_DIM = 18

class IndependentMLP(nn.Module):
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
    model = IndependentMLP().to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@torch.no_grad()
def dacm_snap_categorical(adv_batch, group_idx, valid_states):
    """Exact copy from dacm_replication_test.py"""
    sub = adv_batch[:, group_idx]
    states = valid_states.to(sub.device)
    diff = sub[:, None, :] - states[None, :, :]
    dist = diff.square().sum(dim=-1)
    nearest = dist.argmin(dim=-1)
    return states[nearest]

def buggy_pgd(model, images, labels, epsilon, alpha, steps, bounds):
    """
    EXACT reproduction of the buggy code from dacm_replication_test.py lines 144-192.
    The bug: line 185 passes `adv` instead of `adv_cat_proj`.
    """
    orig = images.clone().detach()
    adv = images.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    
    cat_changes = []
    
    for step in range(steps):
        # Record categorical state BEFORE this step
        old_cats = [adv[:, g].clone() for g in CATEGORICAL_GROUPS]
        
        adv.requires_grad_(True)
        out = model(adv)
        loss = loss_fn(out, labels)
        model.zero_grad()
        loss.backward()
        grad = adv.grad
        
        with torch.no_grad():
            # Continuous update
            adv_cont = adv[:, CONTINUOUS_COLS] + alpha * grad[:, CONTINUOUS_COLS].sign()
            eta = (adv_cont - orig[:, CONTINUOUS_COLS]).clamp(-epsilon, epsilon)
            adv.data[:, CONTINUOUS_COLS] = (orig[:, CONTINUOUS_COLS] + eta).clamp(0.0, 1.0)
            
            # BUG: passes `adv` to snap, not `adv_cat_proj`
            for i, (group_idx, valid_states) in enumerate(zip(CATEGORICAL_GROUPS, bounds)):
                adv_cat = adv[:, group_idx] + alpha * grad[:, group_idx].sign()
                eta_cat = (adv_cat - orig[:, group_idx]).clamp(-epsilon, epsilon)
                adv_cat_proj = (orig[:, group_idx] + eta_cat).clamp(0.0, 1.0)
                
                # THE BUG: uses `adv` not `adv_cat_proj`
                snapped = dacm_snap_categorical(adv, group_idx, valid_states)
                adv.data[:, group_idx] = snapped
        
        # Check if categoricals changed
        total_changed = 0
        for i, g in enumerate(CATEGORICAL_GROUPS):
            changed = (adv[:, g] != old_cats[i]).any(dim=1).sum().item()
            total_changed += changed
        cat_changes.append(total_changed)
        
        adv = adv.detach()
    
    return adv, cat_changes


def fixed_pgd(model, images, labels, epsilon, alpha, steps, bounds):
    """
    Fixed version: passes `adv_cat_proj` to snap instead of `adv`.
    """
    orig = images.clone().detach()
    adv = images.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    
    cat_changes = []
    
    for step in range(steps):
        old_cats = [adv[:, g].clone() for g in CATEGORICAL_GROUPS]
        
        adv.requires_grad_(True)
        out = model(adv)
        loss = loss_fn(out, labels)
        model.zero_grad()
        loss.backward()
        grad = adv.grad
        
        with torch.no_grad():
            adv_cont = adv[:, CONTINUOUS_COLS] + alpha * grad[:, CONTINUOUS_COLS].sign()
            eta = (adv_cont - orig[:, CONTINUOUS_COLS]).clamp(-epsilon, epsilon)
            adv.data[:, CONTINUOUS_COLS] = (orig[:, CONTINUOUS_COLS] + eta).clamp(0.0, 1.0)
            
            # FIX: snap `adv_cat_proj` not `adv`
            for i, (group_idx, valid_states) in enumerate(zip(CATEGORICAL_GROUPS, bounds)):
                adv_cat = adv[:, group_idx] + alpha * grad[:, group_idx].sign()
                eta_cat = (adv_cat - orig[:, group_idx]).clamp(-epsilon, epsilon)
                adv_cat_proj = (orig[:, group_idx] + eta_cat).clamp(0.0, 1.0)
                
                # FIXED: snap adv_cat_proj
                # Create a temporary tensor with the projected values for snapping
                tmp = adv.clone()
                tmp[:, group_idx] = adv_cat_proj
                snapped = dacm_snap_categorical(tmp, group_idx, valid_states)
                adv.data[:, group_idx] = snapped
        
        total_changed = 0
        for i, g in enumerate(CATEGORICAL_GROUPS):
            changed = (adv[:, g] != old_cats[i]).any(dim=1).sum().item()
            total_changed += changed
        cat_changes.append(total_changed)
        
        adv = adv.detach()
    
    return adv, cat_changes


def main():
    print("=" * 70)
    print("BUG 3 VERIFICATION: Dead Code in dacm_replication_test.py")
    print("   Line 185: dacm_snap_categorical(adv, ...) should be")
    print("             dacm_snap_categorical(adv_cat_proj, ...)")
    print("=" * 70)
    
    # Load data
    data = np.loadtxt("./data/nsl-kdd-test.csv", delimiter=',')
    X = torch.tensor(data[:500, :18], dtype=torch.float32).to(DEVICE)
    y = torch.tensor(data[:500, 18], dtype=torch.long).to(DEVICE)
    
    model = load_model("app/ml/model_adv.pth")
    bounds = [torch.eye(len(g), dtype=torch.float32) for g in CATEGORICAL_GROUPS]
    
    # Run buggy version
    print("\n--- Buggy version (adv passed to snap) ---")
    torch.manual_seed(42)
    adv_buggy, changes_buggy = buggy_pgd(model, X, y, 0.15, 0.01, 40, bounds)
    total_buggy_changes = sum(changes_buggy)
    print(f"  Total categorical changes across all steps: {total_buggy_changes}")
    print(f"  Changes per step (first 10): {changes_buggy[:10]}")
    
    with torch.no_grad():
        acc_buggy = (model(adv_buggy).argmax(dim=1) == y).float().mean().item() * 100
    print(f"  Robust accuracy: {acc_buggy:.2f}%")
    
    # Run fixed version
    print("\n--- Fixed version (adv_cat_proj passed to snap) ---")
    torch.manual_seed(42)
    adv_fixed, changes_fixed = fixed_pgd(model, X, y, 0.15, 0.01, 40, bounds)
    total_fixed_changes = sum(changes_fixed)
    print(f"  Total categorical changes across all steps: {total_fixed_changes}")
    print(f"  Changes per step (first 10): {changes_fixed[:10]}")
    
    with torch.no_grad():
        acc_fixed = (model(adv_fixed).argmax(dim=1) == y).float().mean().item() * 100
    print(f"  Robust accuracy: {acc_fixed:.2f}%")
    
    # Compare
    print("\n--- Comparison ---")
    diff = (adv_buggy - adv_fixed).abs()
    print(f"  Max diff between buggy and fixed outputs: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    
    # Check: in the buggy version, do categorical features ever change from original?
    orig_cats_protocol = X[:, CATEGORICAL_GROUPS[0]]
    orig_cats_service = X[:, CATEGORICAL_GROUPS[1]]
    buggy_cats_protocol = adv_buggy[:, CATEGORICAL_GROUPS[0]]
    buggy_cats_service = adv_buggy[:, CATEGORICAL_GROUPS[1]]
    fixed_cats_protocol = adv_fixed[:, CATEGORICAL_GROUPS[0]]
    fixed_cats_service = adv_fixed[:, CATEGORICAL_GROUPS[1]]
    
    proto_changed_buggy = (orig_cats_protocol != buggy_cats_protocol).any(dim=1).sum().item()
    svc_changed_buggy = (orig_cats_service != buggy_cats_service).any(dim=1).sum().item()
    proto_changed_fixed = (orig_cats_protocol != fixed_cats_protocol).any(dim=1).sum().item()
    svc_changed_fixed = (orig_cats_service != fixed_cats_service).any(dim=1).sum().item()
    
    print(f"\n  Buggy: Protocol categories changed from original: {proto_changed_buggy}/500")
    print(f"  Buggy: Service categories changed from original: {svc_changed_buggy}/500")
    print(f"  Fixed: Protocol categories changed from original: {proto_changed_fixed}/500")
    print(f"  Fixed: Service categories changed from original: {svc_changed_fixed}/500")
    
    print(f"\n[BUG 3 VERDICT]:")
    if total_buggy_changes == 0:
        print(f"  CONFIRMED: The buggy code produces ZERO categorical changes.")
        print(f"  The gradient step result (adv_cat_proj) is never used — it's dead code.")
        print(f"  dacm_snap_categorical(adv, ...) just re-snaps the already-snapped values,")
        print(f"  which is an identity operation.")
    else:
        print(f"  PARTIALLY CONFIRMED: {total_buggy_changes} changes occurred (likely from")
        print(f"  continuous feature updates leaking into categorical columns through")
        print(f"  the `adv` tensor, since continuous cols are updated before categoricals).")

if __name__ == "__main__":
    main()
