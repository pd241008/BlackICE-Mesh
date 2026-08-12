import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random

from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import get_train_loader
from app.ml.attacks.fgsm import fgsm_attack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(epochs=50, lr=1e-3, save_path="app/ml/model.pth"):
    model = TabularMLP().to(DEVICE)

    if os.path.exists(save_path):
        print(f"Found pretrained model at {save_path}. Loading it instead of training.")
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model.eval()
        return model

    train_loader = get_train_loader()
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

    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

    return model


def adversarial_train(model, train_loader, epsilon=0.1, epochs=50, lr=1e-3, save_path="app/ml/model_adv.pth"):
    if os.path.exists(save_path):
        print(f"Found pretrained adversarial model at {save_path}. Loading it instead of training.")
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model.eval()
        return model

    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss_epoch = 0
        current_epsilon = min(epsilon, 0.02 + (epsilon - 0.02) * (epoch / 10.0)) if epochs >= 10 else epsilon

        for data, target in train_loader:
            data, target = data.to(DEVICE), target.long().view(-1).to(DEVICE)

            output = model(data)
            loss_clean = F.cross_entropy(output, target)

            adv_data = fgsm_attack(model, data, target, current_epsilon)
            adv_output = model(adv_data)
            loss_adv = F.cross_entropy(adv_output, target)

            loss = 0.5 * loss_clean + 0.5 * loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()

        avg_loss = total_loss_epoch / len(train_loader)
        print(f"[Adv Train] Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Adversarial Model saved at {save_path}")

    model.eval()
    return model


def train_multiple_models(num_models=3, epochs=20):
    for i in range(num_models):
        print(f"\nTraining model {i+1}")

        save_path = f"app/ml/model_{i}.pth"
        if os.path.exists(save_path):
            print(f"Found pretrained ensemble model at {save_path}. Skipping training.")
            continue

        torch.manual_seed(42 + i)
        random.seed(42 + i)

        model = TabularMLP().to(DEVICE)
        train_loader = get_train_loader()
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

        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

    print("\nAll ensemble models trained successfully!")


if __name__ == "__main__":
    train_multiple_models(num_models=3)
    train_model()
