import torch
from app.ml.attacks.pgd import pgd_attack

def evaluate_pgd(model, data_loader, epsilon=0.1, alpha=0.01, steps=40, max_samples=100, continuous_cols=None, categorical_groups=None):
    model.eval()
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    device = next(model.parameters()).device
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        if total >= max_samples:
            break
            
        with torch.no_grad():
            output = model(data)
            init_pred = output.argmax(dim=1)
        
        clean_correct += (init_pred == target).sum().item()
        
        adv_data = pgd_attack(model, data, target, epsilon, alpha, steps, continuous_cols, categorical_groups)
        
        with torch.no_grad():
            adv_output = model(adv_data)
            adv_pred = adv_output.argmax(dim=1)
        
        adv_correct += (adv_pred == target).sum().item()
        total += target.size(0)
        
    return clean_correct, adv_correct, total
