import numpy as np
from app.ml.evaluation.fgsm_eval import evaluate_fgsm

def robustness_curve(model, data_loader, epsilons):
    results = []

    for eps in epsilons:
        clean_correct, adv_correct, total = evaluate_fgsm(
            model, data_loader, epsilon=eps, max_samples=50
        )
        results.append({
            "epsilon": eps,
            "adversarial_accuracy": round(adv_correct / total, 3)
        })

    return results