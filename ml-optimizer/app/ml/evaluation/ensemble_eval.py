import torch
from app.ml.attacks.fgsm import fgsm_attack


def evaluate_ensemble(
    model,
    data_loader,
    epsilon=0.1,
    max_samples=300
):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    clean_correct = 0
    adv_correct = 0
    total = 0

    for data, target in data_loader:

        if total >= max_samples:
            break

        data = data.to(device)
        target = target.to(device)

        # CLEAN
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)

        clean_correct += (
            pred == target
        ).sum().item()

        # small noise improves ensemble diversity
        noise = torch.randn_like(data) * 0.005
        noisy_data = data + noise

        # ATTACK
        adv_data = fgsm_attack(
            model,
            noisy_data,
            target,
            epsilon
        )

        with torch.no_grad():
            adv_output = model(adv_data)
            adv_pred = adv_output.argmax(dim=1)

        adv_correct += (
            adv_pred == target
        ).sum().item()

        total += target.size(0)

    clean_acc = clean_correct / total
    robust_acc = adv_correct / total

    return clean_acc, robust_acc
