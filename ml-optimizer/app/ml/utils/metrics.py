# backend/app/ml/metrics.py
# Adversarial Metrics (ASR, Accuracy Drop)

def attack_success_rate(clean_correct, adv_correct, total):
    return round((clean_correct - adv_correct) / total, 3)


def accuracy_drop(clean_acc, adv_acc):
    return round(clean_acc - adv_acc, 3)
