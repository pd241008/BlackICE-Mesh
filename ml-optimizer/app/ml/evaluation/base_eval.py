# backend/app/ml/evaluate.py
# Robust Accuracy Evaluation

import torch

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    device = next(model.parameters()).device
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total
