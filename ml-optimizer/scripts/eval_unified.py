import os
import json
import torch
import torch.nn.functional as F

from app.ml.attacks.unified_pgd import unified_pgd_attack
from app.ml.attacks.eval_protocol import EVAL_EPSILON as EPSILON
from app.ml.attacks.eval_protocol import EVAL_ALPHA_CONT as ALPHA_CONT
from app.ml.attacks.eval_protocol import EVAL_PGD_STEPS as PGD_STEPS

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import get_test_loader, get_config
from app.ml.utils.checkpoint import load_model_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_unified(model, dataloader, config, K=1):
    model.eval()
    clean_correct = 0
    robust_correct = 0
    total = 0
    
    for data, target in dataloader:
        data, target = data.to(DEVICE), target.long().view(-1).to(DEVICE)
        
        # Clean Eval
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)
            correct_mask = (pred == target)
            clean_correct += correct_mask.sum().item()
            
        # We only evaluate on samples that were already correct
        data = data[correct_mask]
        target = target[correct_mask]
        
        if data.size(0) == 0:
            continue
            
        # PGD Eval
        # We need to evaluate over combinations of K groups, but for this unified pass we just do K=1 exhaustively.
        # Wait, the evaluator in eval_mixed_norm.py handles combinations. Let's just implement a simple K=1 exhaustive loop.
        if K == 0:
            # Just continuous attack
            adv_data = unified_pgd_attack(
                model, data, target, 
                epsilon=EPSILON, alpha=ALPHA_CONT, alpha_cat=0.0, steps=PGD_STEPS,
                continuous_cols=config.CONTINUOUS_COLS, categorical_groups=config.CATEGORICAL_GROUPS
            )
            with torch.no_grad():
                adv_out = model(adv_data)
                adv_pred = adv_out.argmax(dim=1)
                robust_correct += (adv_pred == target).sum().item()
        elif K == 1:
            if not config.CATEGORICAL_GROUPS:
                # Same as K=0 if no cat groups
                adv_data = unified_pgd_attack(
                    model, data, target, 
                    epsilon=EPSILON, alpha=ALPHA_CONT, alpha_cat=0.0, steps=PGD_STEPS,
                    continuous_cols=config.CONTINUOUS_COLS, categorical_groups=[]
                )
                with torch.no_grad():
                    adv_out = model(adv_data)
                    adv_pred = adv_out.argmax(dim=1)
                    robust_correct += (adv_pred == target).sum().item()
            else:
                # Exhaustive K=1 (including K=0 base case)
                
                # Base case: K=0
                adv_data_k0 = unified_pgd_attack(
                    model, data, target, 
                    epsilon=EPSILON, alpha=ALPHA_CONT, alpha_cat=0.0, steps=PGD_STEPS,
                    continuous_cols=config.CONTINUOUS_COLS, categorical_groups=[]
                )
                with torch.no_grad():
                    adv_out_k0 = model(adv_data_k0)
                    adv_pred_k0 = adv_out_k0.argmax(dim=1)
                    survivors = (adv_pred_k0 == target)
                
                # And with all exactly-1-flip attacks
                for group in config.CATEGORICAL_GROUPS:
                    adv_data = unified_pgd_attack(
                        model, data, target, 
                        epsilon=EPSILON, alpha=ALPHA_CONT, alpha_cat=1.0, steps=PGD_STEPS,
                        continuous_cols=config.CONTINUOUS_COLS, categorical_groups=[group]
                    )
                    with torch.no_grad():
                        adv_out = model(adv_data)
                        adv_pred = adv_out.argmax(dim=1)
                        survivors = survivors & (adv_pred == target)
                robust_correct += survivors.sum().item()
                
        total += data.size(0)  # total samples we attacked (which were clean_correct)
        
    return clean_correct, robust_correct, total

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Specific seed to evaluate')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Datasets to evaluate (default: all). E.g. --datasets cicids2017')
    args = parser.parse_args()

    all_datasets = ['nsl-kdd', 'cicids2017', 'unsw_nb15']
    datasets = [d for d in all_datasets if args.datasets is None or d in args.datasets]
    methods = ['hardened', 'curriculum', 'rsc']
    
    os.makedirs("results/unified", exist_ok=True)
    
    for dataset in datasets:
        print(f"\nEvaluating Unified Models for {dataset}")
        config = get_config(dataset)
        loader = get_test_loader(dataset, batch_size=2000)
        safe_name = dataset.replace('-', '_')
        
        for method in methods:
            seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
            path = f"models/unified/model_adv_{method}_{safe_name}{seed_suffix}.pth"
            if not os.path.exists(path):
                print(f"Skipping {method} - path not found: {path}")
                continue
            
            json_path = f"results/unified/eval_{method}_{safe_name}{seed_suffix}.json"
            if os.path.exists(json_path):
                print(f"Skipping {method} - eval already exists at {json_path}")
                continue
                
            model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
            load_model_checkpoint(model, path, device=DEVICE)
            
            clean_correct, robust_correct_k0, total_k0 = evaluate_unified(model, loader, config, K=0)
            _, robust_correct_k1, _ = evaluate_unified(model, loader, config, K=1)
            
            total_eval = len(loader.dataset)
            clean_acc = clean_correct / total_eval * 100
            rob_k0_acc = robust_correct_k0 / total_eval * 100
            rob_k1_acc = robust_correct_k1 / total_eval * 100
            
            print(f"[{method}] Clean: {clean_acc:.2f}% | K=0: {rob_k0_acc:.2f}% | K=1: {rob_k1_acc:.2f}%")
            
            with open(f"results/unified/eval_{method}_{safe_name}{seed_suffix}.json", "w") as f:
                json.dump({
                    "clean_acc": clean_acc,
                    "k0_acc": rob_k0_acc,
                    "k1_acc": rob_k1_acc
                }, f)

if __name__ == "__main__":
    main()
