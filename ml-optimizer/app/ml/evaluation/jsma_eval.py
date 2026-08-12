import torch
import torch.nn.functional as F
from app.ml.attacks.jsma import jsma_attack


def evaluate_jsma(model, data_loader, theta=0.4, max_samples=50):
    clean_correct = 0
    adv_correct = 0
    total = 0

    perturbed_features = []
    confidence_drop = 0

    device = next(model.parameters()).device
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)

        if total >= max_samples:
            break

        output = model(data)
        pred = output.argmax(dim=1)
        clean_correct += (pred == target).sum().item()

        orig_probs = F.softmax(output, dim=1)
        orig_conf = orig_probs.gather(1, target.unsqueeze(1)).squeeze()

        adv = jsma_attack(model, data, target, theta=theta)

        adv_output = model(adv)
        adv_pred = adv_output.argmax(dim=1)
        adv_correct += (adv_pred == target).sum().item()

        adv_probs = F.softmax(adv_output, dim=1)
        adv_conf = adv_probs.gather(1, target.unsqueeze(1)).squeeze()

        confidence_drop += (orig_conf - adv_conf).sum().item()

        diff = (adv - data).abs() > 1e-5
        num_perturbed = diff.sum().item()

        perturbed_features.append(num_perturbed)

        total += target.size(0)

    avg_perturb = sum(perturbed_features) / len(perturbed_features)
    avg_conf_drop = confidence_drop / total

    return (
        clean_correct,
        adv_correct,
        total,
        round(avg_perturb, 2),
        round(avg_conf_drop, 4)
    )
