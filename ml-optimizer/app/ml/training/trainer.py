import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import get_train_loader, get_config
from app.ml.attacks.pgd import pgd_attack
from app.ml.utils.checkpoint import save_model_checkpoint, load_model_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(dataset_name='nsl-kdd', epochs=50, lr=1e-3, save_path="models/model.pth"):
    config = get_config(dataset_name)
    model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)

    if os.path.exists(save_path):
        print(f"Found pretrained model at {save_path}. Loading it instead of training.")
        load_model_checkpoint(model, save_path, device=DEVICE)
        model.eval()
        return model

    train_loader = get_train_loader(dataset_name)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for data, target in train_loader:
            data, target = data.to(DEVICE), target.long().view(-1).to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Clean Train] Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    save_model_checkpoint(model, save_path)

    return model


def adversarial_train(model, dataset_name, train_loader, epsilon=0.1, epochs=50, lr=1e-3, save_path="models/model_adv.pth", rsc=False):
    if os.path.exists(save_path):
        print(f"Found pretrained adversarial model at {save_path}. Loading it instead of training.")
        load_model_checkpoint(model, save_path, device=DEVICE)
        model.eval()
        return model

    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)
    config = get_config(dataset_name)

    model.train()

    for epoch in range(epochs):
        total_loss_epoch = 0
        current_epsilon = min(epsilon, 0.02 + (epsilon - 0.02) * (epoch / 10.0)) if epochs >= 10 else epsilon

        for data, target in train_loader:
            data, target = data.to(DEVICE), target.long().view(-1).to(DEVICE)

            output = model(data)
            loss_clean = F.cross_entropy(output, target)

            adv_data = pgd_attack(model, data, target, epsilon=current_epsilon, alpha=0.01, steps=10, 
                                  continuous_cols=config.CONTINUOUS_COLS, categorical_groups=config.CATEGORICAL_GROUPS, rsc=rsc)
            adv_output = model(adv_data)
            loss_adv = F.cross_entropy(adv_output, target)

            loss = 0.5 * loss_clean + 0.5 * loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()

        avg_loss = total_loss_epoch / len(train_loader)
        print(f"[Adv Train] Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    save_model_checkpoint(model, save_path)

    model.eval()
    return model


def train_multiple_models(dataset_name='nsl-kdd', num_models=3, epochs=20):
    config = get_config(dataset_name)
    for i in range(num_models):
        print(f"\nTraining model {i+1}")

        save_path = f"models/model_{i}.pth"
        if os.path.exists(save_path):
            print(f"Found pretrained ensemble model at {save_path}. Skipping training.")
            continue

        torch.manual_seed(42 + i)
        random.seed(42 + i)

        model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
        train_loader = get_train_loader(dataset_name)
        lr_list = [1e-3, 1e-3, 1e-3]
        optimizer = Adam(model.parameters(), lr=lr_list[i % len(lr_list)])

        model.train()

        for epoch in range(epochs):
            total_loss = 0

            for data, target in train_loader:
                data, target = data.to(DEVICE), target.long().view(-1).to(DEVICE)

                noise = torch.randn_like(data) * (0.02 * (i + 1))
                data_noisy = data + noise
                data_noisy = torch.clamp(data_noisy, 0.0, 1.0)

                optimizer.zero_grad()

                output = model(data_noisy)
                loss = F.cross_entropy(output, target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[Model {i+1}] Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        save_model_checkpoint(model, save_path)

    print("\nAll ensemble models trained successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nsl-kdd')
    parser.add_argument('--rsc', action='store_true', help='Enable Randomized Subset Constraints (RSC) during training')
    args = parser.parse_args()
    
    safe_name = args.dataset.replace('-', '_')
    base_save_path = f"models/model_{safe_name}.pth"
    adv_save_path = f"models/model_adv_{'rsc_' if args.rsc else ''}{safe_name}.pth"
    
    train_multiple_models(dataset_name=args.dataset, num_models=3)
    
    # Train single baseline
    model = train_model(dataset_name=args.dataset, save_path=base_save_path)
    
    # Train hardened
    loader = get_train_loader(args.dataset)
    adversarial_train(model, args.dataset, loader, epochs=50, save_path=adv_save_path, rsc=args.rsc)
