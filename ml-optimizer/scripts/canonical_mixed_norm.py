import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
from torch.utils.data import TensorDataset, DataLoader

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import CATEGORICAL_GROUPS, CONTINUOUS_COLS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_all_states():
    # Group 0 (Protocol) has 3 classes, Group 1 (Flag) has 11 classes
    states = []
    for p in range(3):
        for f in range(11):
            state = torch.zeros(14)
            state[p] = 1.0
            state[3 + f] = 1.0
            states.append(state)
    return torch.stack(states).to(DEVICE)

def canonical_mixed_norm_attack(model, images, labels, epsilon=0.15, alpha=0.01, steps=40, K=0):
    orig = images.clone().detach()
    batch_size = images.shape[0]
    
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    states_tensor = build_all_states()
    
    best_loss = torch.full((batch_size,), -float('inf'), device=DEVICE)
    best_adv = orig.clone()
    
    orig_cat = orig[:, 4:18]
    orig_p = orig_cat[:, :3].argmax(dim=1)
    orig_f = orig_cat[:, 3:].argmax(dim=1)
    
    for i in range(len(states_tensor)):
        state = states_tensor[i]
        
        state_p = state[:3].argmax()
        state_f = state[3:].argmax()
        
        flips_p = (orig_p != state_p).int()
        flips_f = (orig_f != state_f).int()
        total_flips = flips_p + flips_f
        
        reachable_mask = total_flips <= K
        
        if not reachable_mask.any():
            continue
            
        adv = orig.clone().detach()
        # Set all samples that can reach this state to this state
        # (For samples that can't reach it, it doesn't matter what we do since their loss won't be recorded)
        adv.data[:, 4:18] = state.unsqueeze(0).expand(batch_size, -1)
        
        # PGD continuous optimization
        for _ in range(steps):
            adv.requires_grad_(True)
            out = model(adv)
            loss = loss_fn(out, labels)
            
            # Mask out gradients for unreachable samples to save compute? (backward() will sum them anyway)
            loss = (loss * reachable_mask).sum() / reachable_mask.sum().clamp(min=1)
            
            model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                grad = adv.grad
                if CONTINUOUS_COLS:
                    adv_cont = adv[:, CONTINUOUS_COLS] + alpha * grad[:, CONTINUOUS_COLS].sign()
                    eta = (adv_cont - orig[:, CONTINUOUS_COLS]).clamp(-epsilon, epsilon)
                    adv.data[:, CONTINUOUS_COLS] = (orig[:, CONTINUOUS_COLS] + eta).clamp(0.0, 1.0)
                    
            adv = adv.detach()
            
        # Final loss evaluation
        with torch.no_grad():
            out = model(adv)
            final_loss = loss_fn(out, labels)
            
        update_mask = reachable_mask & (final_loss > best_loss)
        best_loss = torch.where(update_mask, final_loss, best_loss)
        best_adv = torch.where(update_mask.unsqueeze(1), adv, best_adv)
        
    return best_adv

def evaluate_model(model_name, weight_path, loader, epsilon=0.15, alpha=0.01, steps=40, Ks=[0]):
    model = TabularMLP().to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    print("=" * 60)
    print(f"EVALUATING MODEL: {model_name}")
    print("=" * 60)
    
    checkpoint_file = f"results/results_{model_name.replace(' ', '_')}.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            results = json.load(f)
    else:
        results = {"clean_correct": 0, "total": 0, "robust_correct": {str(K): 0 for K in Ks}, "completed_batches": 0}
        
    if results["completed_batches"] == len(loader):
        print("Model already fully evaluated from checkpoint.")
        return results
        
    start_batch = results["completed_batches"]
    
    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        if batch_idx < start_batch:
            continue
            
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        
        with torch.no_grad():
            pred = model(batch_x).argmax(dim=1)
        results["clean_correct"] += (pred == batch_y).sum().item()
        
        for K in Ks:
            # Check if this K was already evaluated for this batch? No, we checkpoint per batch.
            set_seed(42 + batch_idx)
            t0 = time.perf_counter()
            adv = canonical_mixed_norm_attack(model, batch_x, batch_y, epsilon, alpha, steps, K=K)
            with torch.no_grad():
                pred = model(adv).argmax(dim=1)
            results["robust_correct"][str(K)] += (pred == batch_y).sum().item()
            dt = time.perf_counter() - t0
            print(f"Batch {batch_idx+1}/{len(loader)} | K={K} | Time: {dt:.2f}s | Acc: {(pred == batch_y).sum().item()/len(batch_y)*100:.2f}%")
            
        results["total"] += len(batch_y)
        results["completed_batches"] += 1
        
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f)
            
    print(f"\nFinal Results for {model_name}:")
    print(f"Clean Accuracy: {results['clean_correct']/results['total']*100:.4f}%")
    for K in Ks:
        print(f"Mixed-Norm (K={K}) Robust Accuracy: {results['robust_correct'][str(K)]/results['total']*100:.4f}%")
    print()
    return results

def main():
    print("Loading test dataset...")
    data = np.loadtxt("./data/nsl-kdd-test.csv", delimiter=',')
    X = torch.tensor(data[:, :18], dtype=torch.float32)
    y = torch.tensor(data[:, 18], dtype=torch.long)
    
    set_seed(42)
    indices = torch.randperm(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    loader = DataLoader(TensorDataset(X_shuffled, y_shuffled), batch_size=2000, shuffle=False)
    
    models = [
        ("Baseline", "models/model.pth"),
        ("Legacy FGSM-Hardened", "models/model_adv.pth"),
        ("New Curriculum PGD-Hardened", "models/model_adv_pgd_curriculum.pth")
    ]
    
    print("Phase 1: Validating K=0 Anchor on Legacy Model")
    evaluate_model("Legacy FGSM-Hardened K0-Anchor", "models/model_adv.pth", loader, Ks=[0])
    
    print("Phase 2: Full Sweep")
    for name, path in models:
        evaluate_model(name, path, loader, Ks=[0, 1, 2])

if __name__ == "__main__":
    main()
