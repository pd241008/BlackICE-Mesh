import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
import itertools
from torch.utils.data import DataLoader

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import get_test_loader, get_config
from app.ml.utils.checkpoint import load_model_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def canonical_mixed_norm_attack(model, images, labels, config, epsilon=0.15, alpha=0.01, steps=40, K=0):
    batch_size = images.shape[0]
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    if not config.CATEGORICAL_GROUPS:
        adv = images.clone().detach()
        for _ in range(steps):
            adv.requires_grad_(True)
            out = model(adv)
            loss = loss_fn(out, labels)
            model.zero_grad()
            loss.mean().backward()
            with torch.no_grad():
                grad = adv.grad
                if config.CONTINUOUS_COLS:
                    adv_cont = adv[:, config.CONTINUOUS_COLS] + alpha * grad[:, config.CONTINUOUS_COLS].sign()
                    eta = (adv_cont - images[:, config.CONTINUOUS_COLS]).clamp(-epsilon, epsilon)
                    adv.data[:, config.CONTINUOUS_COLS] = (images[:, config.CONTINUOUS_COLS] + eta).clamp(0.0, 1.0)
            adv = adv.detach()
        return adv
        
    cat_indices = [idx for group in config.CATEGORICAL_GROUPS for idx in group]
    cat_min = min(cat_indices)
    cat_max = max(cat_indices)
    rel_groups = [[idx - cat_min for idx in group] for group in config.CATEGORICAL_GROUPS]
    
    orig_cat = images[:, cat_min:cat_max+1]
    
    # Generate all K-reachable categorical states for each sample
    states_per_sample = [orig_cat]
    
    if K >= 1:
        for rel_group in rel_groups:
            for i in range(len(rel_group)):
                new_state = orig_cat.clone()
                new_state[:, rel_group] = 0.0
                new_state[:, rel_group[i]] = 1.0
                states_per_sample.append(new_state)

    # Stack them: (num_states, batch_size, cat_width)
    all_states = torch.stack(states_per_sample, dim=0)
    num_states = all_states.shape[0]
    
    # Expand images and labels
    expanded_images = images.unsqueeze(0).expand(num_states, -1, -1).clone()
    expanded_images[:, :, cat_min:cat_max+1] = all_states
    expanded_images = expanded_images.reshape(num_states * batch_size, -1).detach()
    
    expanded_labels = labels.unsqueeze(0).expand(num_states, -1).reshape(-1)
    
    adv = expanded_images.clone()
    orig_expanded_cont = expanded_images[:, config.CONTINUOUS_COLS] if config.CONTINUOUS_COLS else None
    
    # Process in chunks to prevent OOM
    chunk_size = 16000
    final_advs = []
    final_losses = []
    
    for start_idx in range(0, adv.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, adv.shape[0])
        adv_chunk = adv[start_idx:end_idx].clone().detach()
        labels_chunk = expanded_labels[start_idx:end_idx]
        orig_cont_chunk = orig_expanded_cont[start_idx:end_idx] if orig_expanded_cont is not None else None
        
        for _ in range(steps):
            adv_chunk.requires_grad_(True)
            out = model(adv_chunk)
            loss = loss_fn(out, labels_chunk)
            
            model.zero_grad()
            loss.sum().backward()
            
            with torch.no_grad():
                grad = adv_chunk.grad
                if config.CONTINUOUS_COLS:
                    adv_cont = adv_chunk[:, config.CONTINUOUS_COLS] + alpha * grad[:, config.CONTINUOUS_COLS].sign()
                    eta = (adv_cont - orig_cont_chunk).clamp(-epsilon, epsilon)
                    adv_chunk.data[:, config.CONTINUOUS_COLS] = (orig_cont_chunk + eta).clamp(0.0, 1.0)
                    
            adv_chunk = adv_chunk.detach()
            
        with torch.no_grad():
            out = model(adv_chunk)
            loss = loss_fn(out, labels_chunk)
            
        final_advs.append(adv_chunk)
        final_losses.append(loss)
        
    adv = torch.cat(final_advs, dim=0)
    losses = torch.cat(final_losses, dim=0)
    
    losses = losses.view(num_states, batch_size)
    adv = adv.view(num_states, batch_size, -1)
    
    best_loss, best_idx = losses.max(dim=0)
    best_adv = adv[best_idx, torch.arange(batch_size)]
    
    return best_adv


def evaluate_model(model_name, weight_path, dataset_name, loader, config, epsilon=0.15, alpha=0.01, steps=40, Ks=[0]):
    model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
    if not os.path.exists(weight_path):
        print(f"Skipping {model_name}, weight path not found: {weight_path}")
        return None
        
    load_model_checkpoint(model, weight_path, device=DEVICE)
    model.eval()
    
    print("=" * 60)
    print(f"EVALUATING {model_name} on {dataset_name}")
    print("=" * 60)
    
    checkpoint_file = f"results/results_{dataset_name.replace('-','_')}_{model_name.replace(' ', '_')}.json"
    os.makedirs("results", exist_ok=True)
    
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
            set_seed(42 + batch_idx)
            t0 = time.perf_counter()
            adv = canonical_mixed_norm_attack(model, batch_x, batch_y, config, epsilon, alpha, steps, K=K)
            with torch.no_grad():
                pred = model(adv).argmax(dim=1)
            results["robust_correct"][str(K)] += (pred == batch_y).sum().item()
            dt = time.perf_counter() - t0
            print(f"Batch {batch_idx+1}/{len(loader)} | K={K} | Time: {dt:.2f}s | Acc: {(pred == batch_y).sum().item()/len(batch_y)*100:.2f}%")
            
        results["total"] += len(batch_y)
        results["completed_batches"] += 1
        
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f)
            
    print(f"\nFinal Results for {model_name} on {dataset_name}:")
    print(f"Clean Accuracy: {results['clean_correct']/results['total']*100:.4f}%")
    for K in Ks:
        print(f"Mixed-Norm (K={K}) Robust Accuracy: {results['robust_correct'][str(K)]/results['total']*100:.4f}%")
    print()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nsl-kdd')
    args = parser.parse_args()
    
    config = get_config(args.dataset)
    
    print(f"Loading test dataset for {args.dataset}...")
    # Set seed before shuffling data
    set_seed(42)
    loader = get_test_loader(args.dataset, batch_size=2000)
    safe_name = args.dataset.replace('-', '_')
    models = [
        ("Hardened", f"models/model_adv_{safe_name}.pth")
    ]
    
    # Check if models exist (for NSL-KDD the paths might just be model.pth without dataset name)
    if args.dataset == 'nsl-kdd' and not os.path.exists(models[0][1]):
        models = [
            ("Hardened", "models/model_adv.pth")
        ]
    
    print(f"Phase 0.1 Verification for {args.dataset}")
    for name, path in models:
        evaluate_model(name, path, args.dataset, loader, config, Ks=[0, 1])

if __name__ == "__main__":
    main()
