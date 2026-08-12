"""
BlackICE-Mesh | ml-optimizer worker
GPU-bound PyTorch simulation engine. Consumes adversarial jobs from RabbitMQ,
executes the retained Min-Max adversarial training / attack / evaluation logic,
and republishes telemetry (Clean Accuracy, Robust Accuracy, ASR) to the gateway.
"""
import json
import os
import traceback

import pika
import torch

from app.ml.attacks.pgd import pgd_attack
from app.ml.data.loader import (
    CATEGORICAL_GROUPS,
    CONTINUOUS_COLS,
    get_test_loader,
    get_train_loader,
)
from app.ml.evaluation.ensemble_eval import evaluate_ensemble
from app.ml.evaluation.fgsm_eval import evaluate_fgsm
from app.ml.evaluation.jsma_eval import evaluate_jsma
from app.ml.evaluation.pgd_eval import evaluate_pgd
from app.ml.models.architecture import TabularMLP
from app.ml.models.ensemble import EnsembleModel
from app.ml.training.trainer import train_multiple_models
from app.ml.utils.metrics import accuracy_drop, attack_success_rate

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "blackice")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "blackice")
JOB_QUEUE = os.getenv("JOB_QUEUE", "ml.jobs")
RESULTS_EXCHANGE = os.getenv("RESULTS_EXCHANGE", "ml.results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResourceManager:
    """Loads the hardened models and data loaders once at boot."""

    def __init__(self):
        self.base_model = None
        self.ensemble_model = None
        self.train_loader = None
        self.test_loader = None

    def load(self):
        print(f"[ml-optimizer] device={DEVICE}")
        self.base_model = TabularMLP().to(DEVICE)
        self.base_model.load_state_dict(
            torch.load("app/ml/model.pth", map_location=DEVICE, weights_only=True)
        )
        self.base_model.eval()

        self.ensemble_model = EnsembleModel(num_models=3).to(DEVICE)
        self.train_loader = get_train_loader()
        self.test_loader = get_test_loader()
        print("[ml-optimizer] resources loaded")


RESOURCES = ResourceManager()


def _pct(x):
    return round(x * 100.0, 2)


def handle_attack_fgsm(payload):
    eps = float(payload.get("epsilon", 0.15))
    clean, adv, total = evaluate_fgsm(RESOURCES.base_model, RESOURCES.test_loader, epsilon=eps)
    return {
        "attack_type": "FGSM",
        "epsilon": eps,
        "samples": total,
        "clean_accuracy": _pct(clean / total),
        "robust_accuracy": _pct(adv / total),
        "attack_success_rate": _pct(attack_success_rate(clean, adv, total)),
        "relative_drop": _pct(accuracy_drop(clean / total, adv / total) / max(clean / total, 1e-9)),
    }


def handle_attack_pgd(payload):
    eps = float(payload.get("epsilon", 0.1))
    alpha = float(payload.get("alpha", 0.01))
    steps = int(payload.get("steps", 40))
    clean, adv, total = evaluate_pgd(
        RESOURCES.base_model, RESOURCES.test_loader,
        epsilon=eps, alpha=alpha, steps=steps,
        continuous_cols=CONTINUOUS_COLS, categorical_groups=CATEGORICAL_GROUPS,
    )
    return {
        "attack_type": "PGD",
        "epsilon": eps,
        "alpha": alpha,
        "steps": steps,
        "samples": total,
        "clean_accuracy": _pct(clean / total),
        "robust_accuracy": _pct(adv / total),
        "attack_success_rate": _pct(attack_success_rate(clean, adv, total)),
        "relative_drop": _pct(accuracy_drop(clean / total, adv / total) / max(clean / total, 1e-9)),
    }


def handle_attack_jsma(payload):
    theta = float(payload.get("theta", 0.4))
    clean, adv, total, avg_perturb, conf_drop = evaluate_jsma(
        RESOURCES.base_model, RESOURCES.test_loader, theta=theta
    )
    return {
        "attack_type": "JSMA",
        "theta": theta,
        "samples": total,
        "clean_accuracy": _pct(clean / total),
        "robust_accuracy": _pct(adv / total),
        "attack_success_rate": _pct(attack_success_rate(clean, adv, total)),
        "confidence_drop": conf_drop,
        "perturbed_features": avg_perturb,
    }


def handle_defence_adversarial(payload):
    from app.ml.training.trainer import adversarial_train

    eps = float(payload.get("epsilon", 0.15))
    base = RESOURCES.base_model
    robust = TabularMLP().to(DEVICE)
    robust.load_state_dict(base.state_dict())
    adversarial_train(robust, RESOURCES.train_loader, epsilon=eps, epochs=20)

    clean, adv, total = evaluate_fgsm(robust, RESOURCES.test_loader, epsilon=eps)
    return {
        "defence_method": "adversarial_training",
        "epsilon": eps,
        "clean_accuracy": _pct(clean / total),
        "robust_accuracy": _pct(adv / total),
        "attack_success_rate": _pct(attack_success_rate(clean, adv, total)),
    }


def handle_defence_ensemble(payload):
    eps = float(payload.get("epsilon", 0.15))
    clean_acc, robust_acc = evaluate_ensemble(RESOURCES.ensemble_model, RESOURCES.test_loader, epsilon=eps)
    return {
        "defence_method": "ensemble",
        "num_models": RESOURCES.ensemble_model.num_models,
        "epsilon": eps,
        "clean_accuracy": _pct(clean_acc),
        "robust_accuracy": _pct(robust_acc),
        "attack_success_rate": round((clean_acc - robust_acc) * 100.0, 2),
    }


def handle_evaluate_baseline(payload):
    eps = float(payload.get("epsilon", 0.15))
    clean, adv, total = evaluate_fgsm(RESOURCES.base_model, RESOURCES.test_loader, epsilon=eps)
    return {
        "mode": "baseline",
        "epsilon": eps,
        "samples": total,
        "clean_accuracy": _pct(clean / total),
        "robust_accuracy": _pct(adv / total),
        "attack_success_rate": _pct(attack_success_rate(clean, adv, total)),
    }


def handle_train_ensemble(payload):
    num_models = int(payload.get("num_models", 3))
    epochs = int(payload.get("epochs", 20))
    train_multiple_models(num_models=num_models, epochs=epochs)
    return {"mode": "train", "num_models": num_models, "epochs": epochs, "status": "complete"}


HANDLERS = {
    "attack.fgsm": handle_attack_fgsm,
    "attack.pgd": handle_attack_pgd,
    "attack.jsma": handle_attack_jsma,
    "defence.adversarial": handle_defence_adversarial,
    "defence.ensemble": handle_defence_ensemble,
    "evaluate.baseline": handle_evaluate_baseline,
    "train.ensemble": handle_train_ensemble,
}


def _connect():
    creds = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    return pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=creds)
    )


def on_message(ch, method, props, body):
    try:
        job = json.loads(body)
        job_type = job.get("type")
        handler = HANDLERS.get(job_type)
        if handler is None:
            raise ValueError(f"unknown job type: {job_type}")

        result = {"job_id": job.get("job_id"), "type": job_type, **handler(job.get("payload", {}))}
        status = "ok"
    except Exception as exc:
        traceback.print_exc()
        result = {"job_id": job.get("job_id") if "job" in locals() else None,
                  "type": job.get("type") if "job" in locals() else "unknown",
                  "error": str(exc)}
        status = "error"

    ch.basic_publish(
        exchange=RESULTS_EXCHANGE,
        routing_key=f"result.{status}",
        properties=pika.BasicProperties(correlation_id=props.correlation_id, content_type="application/json"),
        body=json.dumps(result),
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    RESOURCES.load()

    while True:
        try:
            conn = _connect()
            channel = conn.channel()
            channel.queue_declare(queue=JOB_QUEUE, durable=True)
            channel.exchange_declare(exchange=RESULTS_EXCHANGE, exchange_type="direct", durable=True)
            channel.basic_consume(queue=JOB_QUEUE, on_message_callback=on_message, auto_ack=False)
            print(f"[ml-optimizer] consuming {JOB_QUEUE} ...")
            channel.start_consuming()
        except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError) as exc:
            print(f"[ml-optimizer] broker unavailable ({exc}); retrying in 5s")
            conn.sleep(5)


if __name__ == "__main__":
    main()
