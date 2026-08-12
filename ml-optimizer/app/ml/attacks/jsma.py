# backend/app/ml/jsma.py

import torch
import torch.nn.functional as F


def compute_jacobian(model, image, target_class):
    image = image.clone().detach().requires_grad_(True)

    output = model(image)

    model.zero_grad()
    output[0, target_class].backward()

    return image.grad.detach()


def jsma_attack(model, image, label, theta=0.4, max_iter=80):

    perturbed = image.clone().detach()

    true_label = label[0].item()

    # Target: least likely class
    with torch.no_grad():
        initial_output = model(perturbed)
        target_class = torch.argmin(initial_output, dim=1)[0].item()

    for _ in range(max_iter):

        perturbed = perturbed.clone().detach().requires_grad_(True)

        output = model(perturbed)
        pred = output.argmax(dim=1)

        # Stop if misclassified
        if pred[0].item() != true_label:
            break

        grads = compute_jacobian(model, perturbed, target_class)

        saliency = grads.abs().view(-1)

        # ✅ Select top-k pixels
        topk = torch.topk(saliency, k=10).indices

        perturbed_flat = perturbed.view(-1).clone()

        # ✅ Update ALL top-k pixels
        for i in topk:
            perturbed_flat[i] += theta

        # ✅ Clamp in normalized Tabular space
        min_val = 0.0
        max_val = 1.0

        perturbed_flat = torch.clamp(perturbed_flat, min_val, max_val)

        perturbed = perturbed_flat.view_as(perturbed).detach()

    return perturbed
