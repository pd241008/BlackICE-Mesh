#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

import argparse

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import get_train_loader, get_config
from app.ml.utils.checkpoint import save_model_checkpoint, load_model_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
LR = 1e-3
EPSILON = 0.15
PGD_STEPS = 10  # Training PGD steps (typically fewer than eval for speed, but enough to find good adv examples)

def pgd_attack_train(model, images, labels, epsilon, alpha_cont, alpha_cat, steps, config):
    """PGD attack adapted for the training loop with configurable alpha_cat"""
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    ori_images = images.clone().detach()
    
    # Random start on continuous features
    if config.CONTINUOUS_COLS:
        random_noise = torch.empty_like(ori_images[:, config.CONTINUOUS_COLS]).uniform_(-epsilon, epsilon)
        images[:, config.CONTINUOUS_COLS] = torch.clamp(ori_images[:, config.CONTINUOUS_COLS] + random_noise, 0.0, 1.0)
        
    for i in range(steps):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        cost = loss_fn(outputs, labels)
        cost.backward()
        grad = images.grad
        
        if config.CONTINUOUS_COLS:
            adv_cont = images[:, config.CONTINUOUS_COLS] + alpha_cont * grad[:, config.CONTINUOUS_COLS].sign()
            eta = torch.clamp(adv_cont - ori_images[:, config.CONTINUOUS_COLS], min=-epsilon, max=epsilon)
            adv_cont_snapped = torch.clamp(ori_images[:, config.CONTINUOUS_COLS] + eta, min=0.0, max=1.0)
            images.data[:, config.CONTINUOUS_COLS] = adv_cont_snapped
            
        if alpha_cat > 0:
            for cat_group in config.CATEGORICAL_GROUPS:
                adv_cat = images[:, cat_group] + alpha_cat * grad[:, cat_group].sign()
                nearest_idx = torch.argmax(adv_cat, dim=1)
                snapped_tensor = F.one_hot(nearest_idx, num_classes=len(cat_group)).float()
                images.data[:, cat_group] = snapped_tensor
                
        images = images.detach()
        
    return images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nsl-kdd')
    args = parser.parse_args()
    
    config = get_config(args.dataset)

    print(f"Starting PGD Adversarial Training on {DEVICE} for {args.dataset}")
    print("Using Mixed Loss: 0.5 Clean + 0.5 Adversarial")
    
    model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
    
    # Warm-start from baseline
    baseline_path = f"models/model_{args.dataset.replace('-', '_')}.pth"
    if os.path.exists(baseline_path):
        print(f"Warm-starting from baseline weights: {baseline_path}")
        load_model_checkpoint(model, baseline_path, device=DEVICE)
    else:
        print("Baseline weights not found. Starting from scratch.")
        
    train_loader = get_train_loader(dataset_name=args.dataset, batch_size=32768)
    optimizer = Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_clean_correct = 0
        total_adv_correct = 0
        total_samples = 0
        
        # Curriculum learning for alpha_cat
        if epoch < 10:
            current_alpha_cat = 0.01  # Basically zero flip chance, train on continuous first
        elif epoch < 30:
            # Scale from 0.01 to 1.0 linearly over epochs 10 to 30
            progress = (epoch - 10) / 20.0
            current_alpha_cat = 0.01 + progress * (1.0 - 0.01)
        else:
            current_alpha_cat = 1.0  # Full strength categorical attack
            
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} | alpha_cat={current_alpha_cat:.2f} ---")
        
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.long().view(-1).to(DEVICE)
            
            # Clean pass
            output_clean = model(data)
            loss_clean = F.cross_entropy(output_clean, target)
            clean_preds = output_clean.argmax(dim=1)
            total_clean_correct += (clean_preds == target).sum().item()
            
            # Generate adversarial examples
            model.eval() # Eval mode for generation
            adv_data = pgd_attack_train(
                model, data, target, 
                epsilon=EPSILON, 
                alpha_cont=0.01, 
                alpha_cat=current_alpha_cat, 
                steps=PGD_STEPS,
                config=config
            )
            model.train() # Back to train mode
            
            # Adversarial pass
            output_adv = model(adv_data)
            loss_adv = F.cross_entropy(output_adv, target)
            adv_preds = output_adv.argmax(dim=1)
            total_adv_correct += (adv_preds == target).sum().item()
            
            # Mixed loss update
            loss = 0.5 * loss_clean + 0.5 * loss_adv
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
        avg_loss = total_loss / total_samples
        clean_acc = total_clean_correct / total_samples * 100
        adv_acc = total_adv_correct / total_samples * 100
        
        print(f"Avg Loss: {avg_loss:.4f} | Clean Acc: {clean_acc:.2f}% | Train Robust Acc: {adv_acc:.2f}%")
        
        # Degeneracy check
        if clean_acc < 58.0 and adv_acc < 58.0:
            print("WARNING: Model may be collapsing to majority class!")
            
    save_path = f"models/model_adv_pgd_curriculum_{args.dataset.replace('-', '_')}.pth"
    save_model_checkpoint(model, save_path)
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    main()
