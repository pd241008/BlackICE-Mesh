import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import CONTINUOUS_COLS, CATEGORICAL_GROUPS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_CSV = "./data/nsl-kdd-test.csv"

def pgd_attack_clamped(model, images, labels, epsilon=0.15, alpha_cont=0.01, alpha_cat=1.0, steps=40):
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    ori_images = images.clone().detach()
    
    if CONTINUOUS_COLS:
        pass # Removed random start to test deterministic behavior
        
    for i in range(steps):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        cost = loss_fn(outputs, labels)
        cost.backward()
        grad = images.grad
        
        # Continuous bounded
        if CONTINUOUS_COLS:
            adv_cont = images[:, CONTINUOUS_COLS] + alpha_cont * grad[:, CONTINUOUS_COLS].sign()
            eta = torch.clamp(adv_cont - ori_images[:, CONTINUOUS_COLS], min=-epsilon, max=epsilon)
            adv_cont_snapped = torch.clamp(ori_images[:, CONTINUOUS_COLS] + eta, min=0.0, max=1.0)
            images.data[:, CONTINUOUS_COLS] = adv_cont_snapped
            
        # Categorical STRICTLY epsilon bounded
        for cat_group in CATEGORICAL_GROUPS:
            # gradient step with alpha_cat
            adv_cat = images[:, cat_group] + alpha_cat * grad[:, cat_group].sign()
            
            # THE NEW CLAMP: strictly enforce epsilon bound on categorical branch before snap
            eta_cat = torch.clamp(adv_cat - ori_images[:, cat_group], min=-epsilon, max=epsilon)
            adv_cat_clamped = torch.clamp(ori_images[:, cat_group] + eta_cat, min=0.0, max=1.0)
            
            # Snap (argmax -> one_hot)
            nearest_idx = torch.argmax(adv_cat_clamped, dim=1)
            snapped_tensor = F.one_hot(nearest_idx, num_classes=len(cat_group)).float()
            images.data[:, cat_group] = snapped_tensor
                
        images = images.detach()
        
    return images

def load_data(csv_path):
    data = np.loadtxt(csv_path, delimiter=',')
    X = torch.tensor(data[:, :18], dtype=torch.float32)
    y = torch.tensor(data[:, 18], dtype=torch.long)
    return X, y

def main():
    print("Loading test dataset...")
    X, y = load_data(TEST_CSV)
    
    model = TabularMLP().to(DEVICE)
    model.load_state_dict(torch.load("models/model_adv.pth", map_location=DEVICE, weights_only=True), strict=True)
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
        
        adv_data = pgd_attack_clamped(model, batch_x, batch_y, epsilon=0.15, alpha_cont=0.01, alpha_cat=1.0, steps=40)
        
        with torch.no_grad():
            adv_pred = model(adv_data).argmax(dim=1)
        adv_correct += (adv_pred == batch_y).sum().item()
        total += batch_y.size(0)
        
    clean_acc = clean_correct / total * 100
    adv_acc = adv_correct / total * 100
    
    print(f"Total Samples: {total}")
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Epsilon-Clamped OHCP Robust Accuracy: {adv_acc:.2f}%")

if __name__ == "__main__":
    main()
