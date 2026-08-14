import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import CONTINUOUS_COLS, CATEGORICAL_GROUPS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mixed_norm_attack(model, images, labels, epsilon=0.15, alpha=0.01, steps=40, K=0):
    orig = images.clone().detach()
    adv = images.clone().detach()
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    for _ in range(steps):
        adv.requires_grad_(True)
        out = model(adv)
        loss = loss_fn(out, labels)
        model.zero_grad()
        loss.backward()
        grad = adv.grad
        
        with torch.no_grad():
            # Continuous update (L_inf)
            if CONTINUOUS_COLS:
                adv_cont = adv[:, CONTINUOUS_COLS] + alpha * grad[:, CONTINUOUS_COLS].sign()
                eta = (adv_cont - orig[:, CONTINUOUS_COLS]).clamp(-epsilon, epsilon)
                adv.data[:, CONTINUOUS_COLS] = (orig[:, CONTINUOUS_COLS] + eta).clamp(0.0, 1.0)
            
            # Categorical update (L_0 <= K)
            if K > 0:
                batch_size = adv.shape[0]
                num_groups = len(CATEGORICAL_GROUPS)
                scores = torch.zeros(batch_size, num_groups, device=DEVICE)
                targets = torch.zeros(batch_size, num_groups, dtype=torch.long, device=DEVICE)
                
                for i, group_idx in enumerate(CATEGORICAL_GROUPS):
                    g = grad[:, group_idx] # shape (batch_size, num_classes)
                    # Normalize gradient magnitude per group
                    g_norm = g / (g.norm(dim=1, keepdim=True) + 1e-8)
                    
                    # Find the current class to mask it out
                    curr_idx = orig[:, group_idx].argmax(dim=1)
                    
                    g_norm_masked = g_norm.clone()
                    g_norm_masked.scatter_(1, curr_idx.unsqueeze(1), -float('inf'))
                    
                    max_val, max_idx = g_norm_masked.max(dim=1)
                    scores[:, i] = max_val
                    targets[:, i] = max_idx
                    
                # Pick the top K groups
                actual_K = min(K, num_groups)
                topk_scores, topk_group_indices = scores.topk(actual_K, dim=1)
                
                # Reset all categoricals to ORIG (enforce strict L0)
                for group_idx in CATEGORICAL_GROUPS:
                    adv.data[:, group_idx] = orig[:, group_idx]
                
                # Apply the top K flips
                for b in range(batch_size):
                    for k in range(actual_K):
                        g_idx = topk_group_indices[b, k].item()
                        group_cols = CATEGORICAL_GROUPS[g_idx]
                        target_class = targets[b, g_idx].item()
                        
                        # Apply flip
                        adv.data[b, group_cols] = 0.0
                        adv.data[b, group_cols[target_class]] = 1.0
            else:
                # K=0: strict L0 (no flips allowed)
                for group_idx in CATEGORICAL_GROUPS:
                    adv.data[:, group_idx] = orig[:, group_idx]
                
        adv = adv.detach()
    return adv

def evaluate_model(model_name, weight_path, loader, epsilon=0.15, alpha=0.01, steps=40):
    model = TabularMLP().to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    print("=" * 60)
    print(f"EVALUATING MODEL: {model_name}")
    print("=" * 60)
    
    # Evaluate Clean
    clean_correct = 0
    total = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        with torch.no_grad():
            pred = model(batch_x).argmax(dim=1)
        clean_correct += (pred == batch_y).sum().item()
        total += batch_y.size(0)
    print(f"Clean Accuracy: {clean_correct/total*100:.2f}%")
    
    # Evaluate K=0, 1, 2
    for K in [0, 1, 2]:
        robust_correct = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            # Deterministic, so no random start
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            
            adv = mixed_norm_attack(model, batch_x, batch_y, epsilon, alpha, steps, K=K)
            
            with torch.no_grad():
                pred = model(adv).argmax(dim=1)
            robust_correct += (pred == batch_y).sum().item()
            
        acc = robust_correct / total * 100
        print(f"Mixed-Norm (K={K}) Robust Accuracy: {acc:.2f}%")
    print()

def main():
    print("Loading test dataset...")
    data = np.loadtxt("./data/nsl-kdd-test.csv", delimiter=',')
    X = torch.tensor(data[:, :18], dtype=torch.float32)
    y = torch.tensor(data[:, 18], dtype=torch.long)
    
    loader = DataLoader(TensorDataset(X, y), batch_size=1000, shuffle=False)
    
    models = [
        ("Baseline (Clean Trained)", "models/model.pth"),
        ("Legacy FGSM-Hardened", "models/model_adv.pth"),
        ("New Curriculum PGD-Hardened", "models/model_adv_pgd_curriculum.pth")
    ]
    
    for name, path in models:
        evaluate_model(name, path, loader, epsilon=0.15, alpha=0.01, steps=40)

if __name__ == "__main__":
    main()
