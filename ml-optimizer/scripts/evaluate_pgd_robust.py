#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import CONTINUOUS_COLS, CATEGORICAL_GROUPS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_CSV = "./data/nsl-kdd-test.csv"

def pgd_attack_eval(model, images, labels, epsilon=0.15, alpha_cont=0.01, alpha_cat=1.0, steps=40):
    """Full-strength PGD attack for evaluation (40 steps, properly scaled)"""
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    ori_images = images.clone().detach()
    
    # Random start on continuous features
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

def load_data(csv_path):
    data = np.loadtxt(csv_path, delimiter=',')
    X = torch.tensor(data[:, :18], dtype=torch.float32)
    y = torch.tensor(data[:, 18], dtype=torch.long)
    return X, y

def evaluate_model(model_path, X, y, name):
    print(f"\n======================================")
    print(f"EVALUATING MODEL: {name}")
    print(f"======================================")
    
    model = TabularMLP().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True), strict=True)
    model.eval()
    
    loader = DataLoader(TensorDataset(X, y), batch_size=500, shuffle=False)
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        
        with torch.no_grad():
            clean_pred = model(batch_x).argmax(dim=1)
        clean_correct += (clean_pred == batch_y).sum().item()
        
        adv_data = pgd_attack_eval(model, batch_x, batch_y, epsilon=0.15, alpha_cont=0.01, alpha_cat=1.0, steps=40)
        
        with torch.no_grad():
            adv_pred = model(adv_data).argmax(dim=1)
        adv_correct += (adv_pred == batch_y).sum().item()
        total += batch_y.size(0)
        
    clean_acc = clean_correct / total * 100
    adv_acc = adv_correct / total * 100
    
    print(f"Total Samples: {total}")
    print(f"Clean Accuracy: {clean_acc:.2f}% ({clean_correct}/{total})")
    print(f"Robust Accuracy (Fixed PGD): {adv_acc:.2f}% ({adv_correct}/{total})")

def main():
    print("Loading test dataset...")
    X, y = load_data(TEST_CSV)
    
    models_to_test = {
        "Baseline": "models/model.pth",
        "Legacy FGSM-Hardened": "models/model_adv.pth",
        "New Curriculum PGD-Hardened": "models/model_adv_pgd_curriculum.pth"
    }
    
    for name, path in models_to_test.items():
        try:
            evaluate_model(path, X, y, name)
        except Exception as e:
            print(f"Failed to evaluate {name}: {e}")

if __name__ == "__main__":
    main()
