import torch
import torch.nn.functional as F


def fgsm_attack(model, image, label, epsilon):
    image = image.clone().detach().requires_grad_(True)

    output = model(image)
    loss = F.cross_entropy(output, label)

    model.zero_grad()
    loss.backward()

    data_grad = image.grad.detach()

    perturbed = image + epsilon * data_grad.sign()
    perturbed = torch.clamp(perturbed, 0.0, 1.0)

    return perturbed.detach()
