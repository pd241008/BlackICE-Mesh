"""Foolbox-based independent verification of robust accuracy.

Runs third-party adversarial attack implementations (Foolbox 3.x) against the
unified checkpoints to cross-validate the hand-rolled attacks in this repo
(PGD in app/ml/attacks/pgd.py, CW in app/ml/attacks/cw.py).

Threat-model mapping:
  - Perturbations are restricted to CONTINUOUS_COLS only. The TabularMLP is
    wrapped so the categorical one-hot blocks of each sample stay frozen at
    their original values while Foolbox optimizes the continuous block.
    This corresponds to K=0 under the unified mixed-norm threat model.
  - bounds=(0, 1) matches the dataset configs' MIN_VAL/MAX_VAL.

Attacks used:
  - L2CarliniWagnerAttack: optimization-based reference for cw.py
  - L2DeepFoolAttack: gradient-based minimal-norm reference
  - LinfPGDAttack: sanity baseline vs app/ml/attacks/pgd.py

Results land in results/foolbox/.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import foolbox as fb

from app.ml.data.loader import get_test_loader, get_config
from app.ml.models.architecture import TabularMLP
from app.ml.utils.checkpoint import load_model_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHODS = ["hardened", "curriculum", "rsc"]


class ContinuousColumnWrapper(nn.Module):
    """Exposes a sub-network over continuous columns only.

    forward(x_cont) rebuilds the full feature vector by re-inserting the
    categorical slice frozen at its original values (assign the per-batch
    tensor to .cat_fixed before attacking), then delegates to the wrapped
    model.
    """

    def __init__(self, model, cont_cols, cat_cols):
        super().__init__()
        self.model = model
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.cat_fixed = None

    def forward(self, x_cont):
        assert self.cat_fixed is not None, "set_cat_fixed() must be called per batch"
        x_full = x_cont.new_empty((x_cont.shape[0],
                                   len(self.cont_cols) + len(self.cat_cols)))
        x_full[:, self.cont_cols] = x_cont
        x_full[:, self.cat_cols] = self.cat_fixed.to(x_cont.dtype)
        return self.model(x_full)


def split_columns(config):
    cont_cols = list(config.CONTINUOUS_COLS)
    cat_cols = [idx for group in config.CATEGORICAL_GROUPS for idx in group]
    return cont_cols, cat_cols


def build_attacks(args):
    available = {
        "cw-l2": lambda: fb.attacks.L2CarliniWagnerAttack(
            steps=args.steps, confidence=args.confidence,
            binary_search_steps=args.binary_search_steps,
            stepsize=args.lr),
        "deepfool-l2": lambda: fb.attacks.L2DeepFoolAttack(steps=args.steps),
        "pgd-linf": lambda: fb.attacks.LinfPGD(
            rel_stepsize=max(args.epsilon / args.steps * 2.5, 1e-3),
            steps=args.steps),
    }
    return {name: available[name]() for name in args.attacks}


def find_checkpoint(safe_name, method, seed):
    base_dir = "models/unified"
    if seed is None:
        candidates = [
            os.path.join(base_dir, f"model_adv_{method}_{safe_name}.pth"),
            f"models/model_adv_{method}_{safe_name}.pth",
        ]
    else:
        candidates = [
            os.path.join(base_dir, f"model_adv_{method}_{safe_name}_seed{seed}.pth"),
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Foolbox cross-validation of robust accuracy "
                    "(continuous-column threat model, K=0)")
    parser.add_argument("--dataset", type=str, default="nsl-kdd")
    parser.add_argument("--methods", type=str, nargs="+", default=METHODS,
                        choices=METHODS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[None])
    parser.add_argument("--attacks", type=str, nargs="+",
                        default=["cw-l2", "deepfool-l2", "pgd-linf"],
                        choices=["cw-l2", "deepfool-l2", "pgd-linf"])
    parser.add_argument("--epsilon", type=float, default=0.15,
                        help="L-inf bound for pgd-linf")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="CW stepsize")
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--binary-search-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit-batches", type=int, default=1,
                        help="Number of test batches to attack (CW is slow)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = get_config(args.dataset)
    safe_name = args.dataset.replace("-", "_")
    cont_cols, cat_cols = split_columns(config)

    attacks = build_attacks(args)
    print(f"Foolbox version: {fb.__version__} | attacks: {list(attacks)} | "
          f"device: {DEVICE}")

    os.makedirs("results/foolbox", exist_ok=True)

    for method in args.methods:
        for seed in args.seeds:
            weight_path = find_checkpoint(safe_name, method, seed)
            if weight_path is None:
                print(f"[SKIP] No checkpoint for {method} seed={seed}")
                continue

            tag = method if seed is None else f"{method}_seed{seed}"
            result_file = (f"results/foolbox/results_{safe_name}_{tag}.json")

            if os.path.exists(result_file) and not args.overwrite:
                with open(result_file) as f:
                    existing = json.load(f)
                if all(existing["attacks"][a]["completed_batches"]
                       >= args.limit_batches for a in args.attacks):
                    print(f"[SKIP] {result_file} already complete.")
                    continue
            else:
                existing = {"clean_correct": 0, "total": 0, "completed_batches": 0,
                            "attacks": {a: {"correct": 0, "completed_batches": 0}
                                        for a in args.attacks}}

            model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
            load_model_checkpoint(model, weight_path, device=DEVICE)
            model.eval()

            wrapper = ContinuousColumnWrapper(model, cont_cols, cat_cols)
            fmodel = fb.PyTorchModel(wrapper, bounds=(0.0, 1.0), device=DEVICE)

            loader = get_test_loader(args.dataset, batch_size=args.batch_size)
            start_batch = existing["completed_batches"]

            for batch_idx, (batch_x, batch_y) in enumerate(loader):
                if batch_idx < start_batch:
                    continue
                if batch_idx >= start_batch + args.limit_batches:
                    break

                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                wrapper.cat_fixed = batch_x[:, cat_cols].detach()
                x_cont = batch_x[:, cont_cols]

                with torch.no_grad():
                    clean_pred = fmodel(x_cont).argmax(dim=1)
                existing["clean_correct"] += (clean_pred == batch_y).sum().item()
                existing["total"] += len(batch_y)

                for name, attack in attacks.items():
                    state = existing["attacks"][name]
                    if state["completed_batches"] >= args.limit_batches:
                        continue
                    t0 = time.perf_counter()
                    if name == "pgd-linf":
                        _, clipped, _ = attack(fmodel, x_cont, batch_y,
                                               epsilons=[args.epsilon])
                        adv = clipped[0]
                    else:
                        _, adv, _ = attack(fmodel, x_cont, batch_y, epsilons=None)
                    with torch.no_grad():
                        adv_pred = fmodel(adv).argmax(dim=1)
                    correct = (adv_pred == batch_y).sum().item()
                    state["correct"] += correct
                    state["completed_batches"] += 1
                    dt = time.perf_counter() - t0
                    print(f"{tag} [{name}] batch {batch_idx + 1} | "
                          f"Time: {dt:.2f}s | Robust acc: "
                          f"{correct / len(batch_y) * 100:.2f}%")

                existing["completed_batches"] += 1
                with open(result_file, "w") as f:
                    json.dump(existing, f)

            total = max(existing["total"], 1)
            print(f"\nFinal Foolbox Results for {tag} on {args.dataset}:")
            print(f"Clean Accuracy: {existing['clean_correct'] / total * 100:.4f}%")
            for name in args.attacks:
                acc = existing["attacks"][name]["correct"] \
                    / max(total, 1) * 100
                print(f"{name} Robust Accuracy: {acc:.4f}%")
            print()


if __name__ == "__main__":
    main()
