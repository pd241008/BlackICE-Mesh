import os
import argparse
import sys
import torch
import torch.nn.functional as F
from torch.optim import Adam

# Inject the artifact dir into sys.path to import unified_pgd
from app.ml.attacks.unified_pgd import unified_pgd_attack

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import get_train_loader, get_config
from app.ml.utils.checkpoint import save_model_checkpoint, load_model_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 32768
LR = 1e-3
EPSILON = 0.15
PGD_STEPS = 10
ALPHA_CONT = 0.01

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['nsl-kdd', 'cicids2017', 'unsw_nb15'])
    parser.add_argument('--method', type=str, required=True, choices=['hardened', 'curriculum', 'rsc'])
    parser.add_argument('--subset_size', type=int, default=None, help='Randomly subset the training data to speed up execution')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        
    config = get_config(args.dataset)
    safe_name = args.dataset.replace('-', '_')
    
    # We save these in models/unified_... to avoid touching old models
    os.makedirs("models/unified", exist_ok=True)
    seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
    save_path = f"models/unified/model_adv_{args.method}_{safe_name}{seed_suffix}.pth"
    
    print(f"Starting Unified Training on {DEVICE} | Dataset: {args.dataset} | Method: {args.method} | Subset: {args.subset_size} | Seed: {args.seed}")
    
    model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
    baseline_path = f"models/model_{safe_name}.pth"
    if os.path.exists(baseline_path):
        print(f"Warm-starting from baseline weights: {baseline_path}")
        load_model_checkpoint(model, baseline_path, device=DEVICE)
    else:
        print("Baseline weights not found. Starting from scratch.")
        
    train_loader = get_train_loader(dataset_name=args.dataset, batch_size=BATCH_SIZE)
    
    if args.subset_size is not None and args.subset_size < len(train_loader.dataset):
        # Create a SubsetRandomSampler or just manually subset the dataset
        import torch.utils.data as data
        indices = torch.randperm(len(train_loader.dataset))[:args.subset_size]
        subset = data.Subset(train_loader.dataset, indices)
        train_loader = data.DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Subsampled dataset to {len(train_loader.dataset)} samples.")
        
    optimizer = Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_clean_correct = 0
        total_adv_correct = 0
        total_samples = 0
        
        # Determine alpha_cat
        if args.method == 'curriculum':
            if epoch < 10:
                current_alpha_cat = 0.01
            elif epoch < 30:
                progress = (epoch - 10) / 20.0
                current_alpha_cat = 0.01 + progress * (1.0 - 0.01)
            else:
                current_alpha_cat = 1.0
        else:
            current_alpha_cat = 1.0 # Full strength for Hardened and RSC
            
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} | alpha_cat={current_alpha_cat:.2f} ---")
        
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.long().view(-1).to(DEVICE)
            
            # Clean pass
            output_clean = model(data)
            loss_clean = F.cross_entropy(output_clean, target)
            clean_preds = output_clean.argmax(dim=1)
            total_clean_correct += (clean_preds == target).sum().item()
            
            # Generate adversarial examples
            model.eval()
            adv_data = unified_pgd_attack(
                model, data, target, 
                epsilon=EPSILON, 
                alpha=ALPHA_CONT, 
                alpha_cat=current_alpha_cat, 
                steps=PGD_STEPS,
                continuous_cols=config.CONTINUOUS_COLS,
                categorical_groups=config.CATEGORICAL_GROUPS,
                rsc=(args.method == 'rsc')
            )
            model.train()
            
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
        
    save_model_checkpoint(model, save_path)
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    main()
