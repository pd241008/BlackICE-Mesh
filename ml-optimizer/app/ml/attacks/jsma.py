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

    with torch.no_grad():
        initial_output = model(perturbed)
        target_class = torch.argmin(initial_output, dim=1)[0].item()

    for _ in range(max_iter):
        perturbed = perturbed.clone().detach().requires_grad_(True)

        output = model(perturbed)
        pred = output.argmax(dim=1)

        if pred[0].item() != true_label:
            break

        grads = compute_jacobian(model, perturbed, target_class)
        saliency = grads.abs().view(-1)

        topk = torch.topk(saliency, k=10).indices

        perturbed_flat = perturbed.view(-1).clone()

        for i in topk:
            perturbed_flat[i] += theta

        perturbed_flat = torch.clamp(perturbed_flat, 0.0, 1.0)
        perturbed = perturbed_flat.view_as(perturbed).detach()

    return perturbed
