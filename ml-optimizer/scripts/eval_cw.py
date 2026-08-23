import argparse
import json
import os
import time

import torch

from app.ml.attacks.cw import cw_mixed_norm_attack
from app.ml.data.loader import get_test_loader, get_config
from app.ml.models.architecture import TabularMLP
from app.ml.utils.checkpoint import load_model_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHODS = ["hardened", "curriculum", "rsc"]


def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)


def find_checkpoint(dataset_name, safe_name, method, seed):
    """Resolve unified checkpoint path for a method/seed (seed=None -> base)."""
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


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _delta_stats(adv, orig, config):
    """L-inf / L2 magnitudes of the perturbation on the continuous subspace."""
    cols = list(config.CONTINUOUS_COLS)
    if not cols:
        return {"max_linf": 0.0, "mean_l2": 0.0}
    d = (adv - orig)[:, cols]
    return {
        "max_linf": d.abs().max().item(),
        "mean_l2": d.norm(dim=1).mean().item(),
    }


def evaluate_model(method, weight_path, dataset_name, config, args, seed=None):
    set_seed(seed if seed is not None else 42)
    loader = get_test_loader(dataset_name, batch_size=args.batch_size)

    model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
    load_model_checkpoint(model, weight_path, device=DEVICE)
    model.eval()

    tag = method if seed is None else f"{method}_seed{seed}"
    result_file = f"results/cw/results_{dataset_name.replace('-', '_')}_{tag}.json"
    os.makedirs("results/cw", exist_ok=True)

    attack_config = {
        "checkpoint_sha256": _sha256(weight_path),
        "epsilon": args.epsilon,
        "steps": args.steps,
        "lr": args.lr,
        "kappa": args.kappa,
        "binary_search_steps": args.binary_search_steps,
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "Ks": args.Ks,
        "selection_metric": "post-hoc cross-entropy on final adversarial example "
                            "(worst candidate state per sample)",
        "linf_policy": "delta = eps*tanh(p) on continuous cols; bounded by "
                       "construction; max_linf verified per batch",
    }

    if os.path.exists(result_file) and not args.overwrite:
        with open(result_file) as f:
            results = json.load(f)
        if results.get("completed_batches") == len(loader):
            print(f"[SKIP] {result_file} already complete.")
            return results
    else:
        results = {"clean_correct": 0, "total": 0,
                   "robust_correct": {str(K): 0 for K in args.Ks},
                   "completed_batches": 0}

    results["attack_config"] = attack_config
    results.setdefault("batches", [])

    start_batch = results["completed_batches"]

    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        if batch_idx < start_batch:
            continue
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

        with torch.no_grad():
            pred = model(batch_x).argmax(dim=1)
        results["clean_correct"] += (pred == batch_y).sum().item()

        batch_record = {"batch_idx": batch_idx, "n": len(batch_y)}
        for K in args.Ks:
            t0 = time.perf_counter()
            adv = cw_mixed_norm_attack(
                model, batch_x, batch_y, config,
                epsilon=args.epsilon, steps=args.steps, lr=args.lr,
                kappa=args.kappa, binary_search_steps=args.binary_search_steps,
                chunk_size=args.chunk_size, K=K,
            )
            with torch.no_grad():
                pred = model(adv).argmax(dim=1)
            correct = (pred == batch_y).sum().item()
            results["robust_correct"][str(K)] += correct
            dt = time.perf_counter() - t0
            stats = _delta_stats(adv, batch_x, config)
            if stats["max_linf"] > args.epsilon + 1e-5:
                raise AssertionError(
                    f"L-inf budget violated on batch {batch_idx} K={K}: "
                    f"{stats['max_linf']:.6f} > {args.epsilon}")
            batch_record[f"K={K}"] = {
                "correct": correct,
                **{k: round(v, 6) for k, v in stats.items()},
                "seconds": round(dt, 2),
            }
            print(f"Batch {batch_idx + 1}/{len(loader)} | K={K} | "
                  f"Time: {dt:.2f}s | Acc: {correct / len(batch_y) * 100:.2f}% | "
                  f"max|d|oo: {stats['max_linf']:.4f}")

        results["total"] += len(batch_y)
        results["completed_batches"] += 1
        results["batches"].append(batch_record)
        with open(result_file, "w") as f:
            json.dump(results, f)

    total = max(results["total"], 1)
    print(f"\nFinal CW Results for {tag} on {dataset_name}:")
    print(f"Clean Accuracy: {results['clean_correct'] / total * 100:.4f}%")
    for K in args.Ks:
        acc = results["robust_correct"][str(K)] / total * 100
        print(f"CW Mixed-Norm (K={K}) Robust Accuracy: {acc:.4f}%")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Carlini-Wagner mixed-norm robust accuracy evaluation "
                    "(unified checkpoints)")
    parser.add_argument("--dataset", type=str, default="nsl-kdd")
    parser.add_argument("--methods", type=str, nargs="+", default=METHODS,
                        choices=METHODS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[None],
                        help="Seeds to evaluate; omit for base (non-seeded) runs")
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--binary-search-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--Ks", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="Evaluate only the first N batches (smoke tests)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = get_config(args.dataset)
    safe_name = args.dataset.replace("-", "_")

    for method in args.methods:
        for seed in args.seeds:
            path = find_checkpoint(args.dataset, safe_name, method, seed)
            if path is None:
                print(f"[SKIP] No checkpoint for {method} seed={seed} on {args.dataset}")
                continue
            print("=" * 60)
            print(f"EVALUATING {method} (seed={seed}) on {args.dataset}")
            print(f"Checkpoint: {path}")
            print("=" * 60)
            if args.limit_batches is not None:
                # Smoke-test mode: cap the loader without touching result files.
                _evaluate_limited(method, path, args.dataset, config, args, seed)
                continue
            evaluate_model(method, path, args.dataset, config, args, seed=seed)


def _evaluate_limited(method, weight_path, dataset_name, config, args, seed):
    """Bounded evaluation used for smoke tests; results printed only."""
    set_seed(42)
    full_loader = get_test_loader(dataset_name, batch_size=args.batch_size)
    loader = [b for i, b in zip(range(args.limit_batches), full_loader)]

    model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
    load_model_checkpoint(model, weight_path, device=DEVICE)
    model.eval()

    clean_correct = 0
    robust_correct = {K: 0 for K in args.Ks}
    total = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        with torch.no_grad():
            clean_correct += (model(batch_x).argmax(1) == batch_y).sum().item()
        for K in args.Ks:
            adv = cw_mixed_norm_attack(
                model, batch_x, batch_y, config,
                epsilon=args.epsilon, steps=args.steps, lr=args.lr,
                kappa=args.kappa, binary_search_steps=args.binary_search_steps,
                chunk_size=args.chunk_size, K=K)
            with torch.no_grad():
                robust_correct[K] += (model(adv).argmax(1) == batch_y).sum().item()
        total += len(batch_y)

    tag = method if seed is None else f"{method}_seed{seed}"
    print(f"[SMOKE] {tag}/{dataset_name}: clean {clean_correct / total * 100:.2f}% | "
          + " | ".join(f"K={K} {robust_correct[K] / total * 100:.2f}%" for K in args.Ks))


if __name__ == "__main__":
    main()
