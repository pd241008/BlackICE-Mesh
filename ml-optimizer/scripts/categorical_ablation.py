import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import CONTINUOUS_COLS, CATEGORICAL_GROUPS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def categorical_only_attack(model, images, labels, epsilon=0.15, alpha=0.01, steps=40):
    orig = images.clone().detach()
    adv = images.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    
    for _ in range(steps):
        adv.requires_grad_(True)
        out = model(adv)
        loss = loss_fn(out, labels)
        model.zero_grad()
        loss.backward()
        grad = adv.grad
        
        with torch.no_grad():
            # ZERO continuous update (freeze continuous features)
            # Only update categorical features
            for group_idx in CATEGORICAL_GROUPS:
                adv_cat = adv[:, group_idx] + alpha * grad[:, group_idx].sign()
                eta_cat = (adv_cat - orig[:, group_idx]).clamp(-epsilon, epsilon)
                adv.data[:, group_idx] = (orig[:, group_idx] + eta_cat).clamp(0.0, 1.0)
                
        adv = adv.detach()
    return adv

def main():
    print("Loading test dataset...")
    data = np.loadtxt("./data/nsl-kdd-test.csv", delimiter=',')
    X = torch.tensor(data[:, :18], dtype=torch.float32)
    y = torch.tensor(data[:, 18], dtype=torch.long)
    
    loader = DataLoader(TensorDataset(X, y), batch_size=1000, shuffle=False)
    
    model = TabularMLP().to(DEVICE)
    model.load_state_dict(torch.load("models/model_adv.pth", map_location=DEVICE, weights_only=True))
    model.eval()

    # Empirical check: categorical-only ablation
    print("--- Categorical-Only Ablation ---")
    correct = 0
    total = 0
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        adv = categorical_only_attack(model, batch_x, batch_y, 0.15, 0.01, 40)
        
        with torch.no_grad():
            pred = model(adv).argmax(dim=1)
        correct += (pred == batch_y).sum().item()
        total += batch_y.size(0)
        
    acc = correct / total * 100
    print(f"Total Samples: {total}")
    print(f"Categorical-Only Robust Accuracy: {acc:.4f}%")

if __name__ == "__main__":
    main()
