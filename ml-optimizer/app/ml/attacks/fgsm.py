import torch
import torch.nn.functional as F


def fgsm_attack(model, image, label, epsilon):

    image = image.clone().detach().requires_grad_(True)

    # Forward
    output = model(image)
    loss = F.cross_entropy(output, label)

    # Backward
    model.zero_grad()
    loss.backward()

    data_grad = image.grad.detach()

    # Basic Epsilon step
    # (No scaling needed for standard Tabular Min-Max)

    # FGSM
    perturbed = image + epsilon * data_grad.sign()

    # ✅ Clamp for Tabular Min-Max
    min_val = 0.0
    max_val = 1.0

    perturbed = torch.clamp(perturbed, min_val, max_val)

    return perturbed.detach()
