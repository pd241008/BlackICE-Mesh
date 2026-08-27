"""Scalability study: wall-clock timing for exhaustive K=1 and K=2 evaluation.

Runs on GPU (falls back to CPU). Logs per-batch timing with candidate counts.
Same PGD-40 backend as eval_deepfool_k1.py; semantics: exhaustive-state AND.

The enumeration pattern matches eval_deepfool_k1.py exactly: the base
(no-flip) state is evaluated once explicitly, then every |g| one-hot state
per group (including the redundant original) is evaluated in the group loop.
For K=2, every |g_i| x |g_j| Cartesian product per pair is enumerated.

Samples are batched (not processed one-at-a-time) for GPU efficiency.
Wall-clock is measured per-batch; per-sample timing is derived.

Usage:
  python scripts/eval_scalability.py --dataset unsw_nb15 --method rsc --seed 42 \
      --K 1 --eligible-groups 0,1,2,3,4 --num-samples 500

  python scripts/eval_scalability.py --dataset unsw_nb15 --method rsc --seed 42 \
      --K 2 --eligible-groups 0,1,2,3,4 --num-samples 500
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ml.attacks.unified_pgd import unified_pgd_attack
from app.ml.attacks.eval_protocol import EVAL_EPSILON, EVAL_ALPHA_CONT, EVAL_PGD_STEPS
from app.ml.data.loader import get_test_loader, get_config
from app.ml.models.architecture import TabularMLP
from app.ml.utils.checkpoint import load_model_checkpoint

EPS = EVAL_EPSILON


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def enumerate_candidates(eligible_groups, K):
    """Enumerate candidate states matching eval_deepfool_k1.py's pattern."""
    candidates = [[]]
    if K >= 1:
        for gi, g in enumerate(eligible_groups):
            for j in range(len(g)):
                candidates.append([(gi, j)])
    if K >= 2:
        for gi, gj in itertools.combinations(range(len(eligible_groups)), 2):
            for ii in range(len(eligible_groups[gi])):
                for ij in range(len(eligible_groups[gj])):
                    candidates.append([(gi, ii), (gj, ij)])
    return candidates


def minimal_candidate_count(eligible_groups, K):
    """Minimal non-redundant count: 1 base + only-new flips."""
    count = 1
    if K >= 1:
        for g in eligible_groups:
            count += len(g) - 1
    if K >= 2:
        for gi, gj in itertools.combinations(range(len(eligible_groups)), 2):
            count += (len(eligible_groups[gi]) - 1) * (len(eligible_groups[gj]) - 1)
    return count


def build_batch_states(data, eligible_groups, candidate):
    """Apply a candidate flip to an entire batch tensor. Returns new tensor."""
    state = data.clone()
    for gi, col_idx in candidate:
        gcols = eligible_groups[gi]
        state[:, gcols] = 0.0
        state[:, gcols[col_idx]] = 1.0
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="unsw_nb15")
    p.add_argument("--method", default="rsc")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--K", type=int, required=True, choices=[1, 2])
    p.add_argument("--eligible-groups", type=str, required=True,
                   help="comma-separated group indices, e.g. '0,1,2'")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    eligible = [int(x) for x in args.eligible_groups.split(",")]
    config = get_config(args.dataset)
    safe_name = args.dataset.replace("-", "_")
    ckpt = f"models/unified/model_adv_{args.method}_{safe_name}_seed{args.seed}.pth"
    assert os.path.exists(ckpt), f"missing checkpoint {ckpt}"
    sha = sha256_file(ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLP(input_dim=config.FEATURE_DIM).to(device)
    load_model_checkpoint(model, ckpt, device=str(device))
    model.to(device).eval()

    cont_cols = list(config.CONTINUOUS_COLS)
    all_groups = [list(g) for g in config.CATEGORICAL_GROUPS]
    eligible_groups = [all_groups[i] for i in eligible]

    candidates = enumerate_candidates(eligible_groups, args.K)
    num_enumerated = len(candidates)
    num_minimal = minimal_candidate_count(eligible_groups, args.K)

    print(f"[CONFIG] dataset={args.dataset} K={args.K} eligible={eligible}")
    print(f"  enumerated candidates: {num_enumerated} "
          f"(minimal non-redundant: {num_minimal})")
    print(f"  device: {device} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    def state_survival_batch(x_batch, y_batch):
        """Run PGD-40 on entire batch, return bool survival tensor."""
        adv = unified_pgd_attack(
            model, x_batch, y_batch, epsilon=EPS, alpha=EVAL_ALPHA_CONT,
            alpha_cat=0.0, steps=EVAL_PGD_STEPS,
            continuous_cols=cont_cols, categorical_groups=[])
        with torch.no_grad():
            return model(adv).argmax(1) == y_batch

    # Phase 1: Collect num_samples clean-correct samples
    loader = get_test_loader(args.dataset, batch_size=args.batch_size)
    collected_x, collected_y = [], []
    clean_correct_total = 0
    total_seen = 0

    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        with torch.no_grad():
            ok = model(bx).argmax(1) == by
        clean_correct_total += int(ok.sum())
        total_seen += bx.size(0)
        data, target = bx[ok], by[ok]
        if data.size(0) > 0:
            collected_x.append(data)
            collected_y.append(target)
        if sum(c.size(0) for c in collected_x) >= args.num_samples:
            break

    all_x = torch.cat(collected_x)[:args.num_samples]
    all_y = torch.cat(collected_y)[:args.num_samples]
    n = all_x.size(0)
    print(f"[COLLECTED] {n} clean-correct samples from {total_seen} total "
          f"(clean_acc={100*clean_correct_total/max(total_seen,1):.2f}%)")

    # Phase 2: Run all candidates on the full batch, time per candidate
    candidate_times = []
    t_batch_start = time.time()

    # Base pass
    t0 = time.time()
    surv_k0 = state_survival_batch(all_x, all_y)
    candidate_times.append({"candidate_idx": 0, "label": "base_K0",
                            "wallclock_seconds": round(time.time() - t0, 6)})

    # K=K AND accumulation
    surv_kk = surv_k0.clone()
    for ci, cand in enumerate(candidates[1:], start=1):
        state = build_batch_states(all_x, eligible_groups, cand)
        t0 = time.time()
        surv_kk &= state_survival_batch(state, all_y)
        candidate_times.append({"candidate_idx": ci, "label": str(cand),
                                "wallclock_seconds": round(time.time() - t0, 6)})

    total_wall = time.time() - t_batch_start
    avg_per_candidate = total_wall / num_enumerated
    avg_per_sample = total_wall / n

    k0_count = int(surv_k0.sum())
    kk_count = int(surv_kk.sum())

    print(f"[DONE] {n} samples x {num_enumerated} candidates = "
          f"{total_wall:.1f}s ({avg_per_sample:.4f}s/sample)")

    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "checkpoint_sha256": sha,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_note": ("Wall-clock times use GPU inference and are not directly "
                        "comparable to the production evaluator (eval_deepfool_k1.py) "
                        "which hardcodes CPU. The reported quantity is the relative "
                        "scaling trend across group counts, not an absolute cost figure."),
        "K": args.K,
        "eligible_groups": eligible,
        "num_candidates_enumerated": num_enumerated,
        "num_candidates_minimal": num_minimal,
        "redundant_overhead": num_enumerated - num_minimal,
        "num_samples": n,
        "total_samples_seen": total_seen,
        "clean_accuracy_pct": round(100 * clean_correct_total / max(total_seen, 1), 4),
        "total_wallclock_seconds": round(total_wall, 3),
        "avg_wallclock_per_sample": round(avg_per_sample, 6),
        "avg_wallclock_per_candidate": round(avg_per_candidate, 6),
        "pgd_steps": EVAL_PGD_STEPS,
        "epsilon": EPS,
        "k0_survivors": k0_count,
        "kk_survivors": kk_count,
        "k0_robust_pct": round(100 * k0_count / max(n, 1), 4),
        "kk_robust_pct": round(100 * kk_count / max(n, 1), 4),
        "candidate_times": candidate_times,
    }

    if args.out is None:
        args.out = (f"results/scalability_{safe_name}_K{args.K}"
                    f"_g{'_'.join(map(str, eligible))}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
