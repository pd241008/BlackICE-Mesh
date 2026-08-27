"""JSMA vs Exhaustive evaluator comparison on categorical groups.

Implements a group-level JSMA (Jacobian Saliency Map Attack) that:
1. Computes gradient of target-class logit w.r.t. input
2. Aggregates per-group saliency (max |grad| over one-hot columns)
3. Greedily flips the most salient group to its most adversarial column
4. Flips up to K groups total

Then compares against the exhaustive K=1 evaluator:
- Agreement rate: both agree robust / both agree vulnerable
- Divergence: JSMA says robust but exhaustive says vulnerable
- Missed-flip rate: JSMA misses flips that exhaustive catches

Usage:
  python scripts/eval_jsma_vs_exhaustive.py --dataset unsw_nb15 --method rsc \
      --seed 42 --K 1 --num-samples 500
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
    """Enumerate candidate states (matching eval_deepfool_k1.py)."""
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


def build_state(data, eligible_groups, candidate):
    state = data.clone()
    for gi, col_idx in candidate:
        gcols = eligible_groups[gi]
        state[:, gcols] = 0.0
        state[:, gcols[col_idx]] = 1.0
    return state


def jsma_group_attack(model, x, y, eligible_groups, K,
                      continuous_cols, eps_schedule=None):
    """Group-level JSMA: greedily flip K groups by saliency.

    Returns:
        adv: adversarial example (tensor)
        flips: list of (group_idx, col_idx) actually applied
        saliency_trace: list of per-step saliency scores
    """
    if eps_schedule is None:
        eps_schedule = [EPS]

    n_groups = len(eligible_groups)
    device = x.device
    perturbed = x.clone().detach()
    flips = []
    saliency_trace = []

    for step in range(K):
        perturbed = perturbed.clone().detach().requires_grad_(True)
        output = model(perturbed)
        pred = output.argmax(1)

        if pred[0].item() != y[0].item():
            break

        # Target = class with lowest logit (most adversarial)
        with torch.no_grad():
            target_class = model(perturbed).argmin(1).item()

        # Forward + backward through the full graph for gradient
        output = model(perturbed)
        grad_out = output[0, target_class]
        model.zero_grad()
        grad_out.backward()
        grads = perturbed.grad.detach()

        # Per-group saliency: max |grad| over one-hot columns
        # Also compute best adversarial column per group
        group_saliency = []
        group_best_col = []
        for gi, gcols in enumerate(eligible_groups):
            g = grads[0, gcols]  # shape: |g|
            # Saliency = max |gradient| in this group
            sal = g.abs().max().item()
            # Best adversarial column: the one that maximizes gradient
            # (for increasing the adversarial target's logit)
            best_col = g.argmax().item()
            group_saliency.append(sal)
            group_best_col.append(best_col)

        # Pick the group with highest saliency
        best_gi = max(range(n_groups), key=lambda i: group_saliency[i])
        best_col = group_best_col[best_gi]

        # Skip if this group was already flipped
        if any(fi == best_gi for fi, _ in flips):
            # Find next best
            remaining = [i for i in range(n_groups)
                         if not any(fi == i for fi, _ in flips)]
            if not remaining:
                break
            best_gi = max(remaining, key=lambda i: group_saliency[i])
            best_col = group_best_col[best_gi]

        saliency_trace.append({
            "step": step,
            "group_saliency": group_saliency,
            "chosen_group": best_gi,
            "chosen_col": best_col,
        })

        # Apply the flip
        gcols = eligible_groups[best_gi]
        with torch.no_grad():
            perturbed[0, gcols] = 0.0
            perturbed[0, gcols[best_col]] = 1.0
        flips.append((best_gi, best_col))

    return perturbed.detach(), flips, saliency_trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="unsw_nb15")
    p.add_argument("--method", default="rsc")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    config = get_config(args.dataset)
    safe_name = args.dataset.replace("-", "_")
    ckpt = f"models/unified/model_adv_{args.method}_{safe_name}_seed{args.seed}.pth"
    assert os.path.exists(ckpt), f"missing {ckpt}"
    sha = sha256_file(ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLP(input_dim=config.FEATURE_DIM).to(device)
    load_model_checkpoint(model, ckpt, device=str(device))
    model.to(device).eval()

    cont_cols = list(config.CONTINUOUS_COLS)
    all_groups = [list(g) for g in config.CATEGORICAL_GROUPS]
    n_groups = len(all_groups)

    # Use all groups as eligible
    eligible = list(range(n_groups))
    eligible_groups = [all_groups[i] for i in eligible]

    candidates = enumerate_candidates(eligible_groups, args.K)
    num_enumerated = len(candidates)

    print(f"[CONFIG] dataset={args.dataset} K={args.K} eligible={eligible}")
    print(f"  exhaustive candidates: {num_enumerated}, JSMA: greedy {args.K} steps")
    print(f"  device: {device}")

    def state_survival_batch(x_batch, y_batch):
        adv = unified_pgd_attack(
            model, x_batch, y_batch, epsilon=EPS, alpha=EVAL_ALPHA_CONT,
            alpha_cat=0.0, steps=EVAL_PGD_STEPS,
            continuous_cols=cont_cols, categorical_groups=[])
        with torch.no_grad():
            return model(adv).argmax(1) == y_batch

    # Collect clean-correct samples
    loader = get_test_loader(args.dataset, batch_size=args.batch_size)
    collected_x, collected_y = [], []
    clean_total = 0
    seen = 0

    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        with torch.no_grad():
            ok = model(bx).argmax(1) == by
        clean_total += int(ok.sum())
        seen += bx.size(0)
        d, t = bx[ok], by[ok]
        if d.size(0) > 0:
            collected_x.append(d)
            collected_y.append(t)
        if sum(c.size(0) for c in collected_x) >= args.num_samples:
            break

    all_x = torch.cat(collected_x)[:args.num_samples]
    all_y = torch.cat(collected_y)[:args.num_samples]
    n = all_x.size(0)
    print(f"[COLLECTED] {n} samples, clean_acc={100*clean_total/max(seen,1):.2f}%")

    # Phase 1: Exhaustive K=1 evaluation (batched)
    print(f"[EXHAUSTIVE] Running {num_enumerated} candidates...")
    t0 = time.time()
    surv_k0 = state_survival_batch(all_x, all_y)
    surv_kk = surv_k0.clone()
    for cand in candidates[1:]:
        state = build_state(all_x, eligible_groups, cand)
        surv_kk &= state_survival_batch(state, all_y)
    t_exhaustive = time.time() - t0

    # Phase 2: JSMA group-level evaluation (per-sample)
    # JSMA picks a flip via greedy saliency, then we apply PGD-40 to that
    # candidate state — same evaluation backend as exhaustive, different
    # selection strategy.
    print(f"[JSMA] Running greedy {args.K}-step saliency + PGD-40 on {n} samples...")
    t0 = time.time()
    jsma_survivors = []
    jsma_flips_per_sample = []
    for si in range(n):
        x = all_x[si:si+1]
        y = all_y[si:si+1]

        # JSMA selects a flip via gradient saliency
        adv, flips, trace = jsma_group_attack(
            model, x, y, eligible_groups, args.K,
            continuous_cols=cont_cols)

        # Apply the same PGD-40 backend as exhaustive
        if flips:
            jsma_survived = state_survival_batch(adv, y)
        else:
            # JSMA found nothing to flip; check original
            jsma_survived = torch.tensor([True], device=device)

        jsma_survivors.append(bool(jsma_survived))
        jsma_flips_per_sample.append(len(flips))

        if si % 100 == 0 and si > 0:
            print(f"  [{si}/{n}]")

    t_jsma = time.time() - t0

    # Compare
    exhaustive_robust = surv_kk.cpu().numpy()
    jsma_robust = torch.tensor(jsma_survivors).numpy()

    agree_robust = sum(exhaustive_robust & jsma_robust)
    agree_vulnerable = sum(~exhaustive_robust & ~jsma_robust)
    jsma_robust_exhaustive_vulnerable = sum(~exhaustive_robust & jsma_robust)
    jsma_vulnerable_exhaustive_robust = sum(exhaustive_robust & ~jsma_robust)

    agreement = (agree_robust + agree_vulnerable) / n

    print(f"\n{'='*60}")
    print(f"RESULTS (K={args.K}, {n} samples)")
    print(f"{'='*60}")
    print(f"Exhaustive robust: {int(exhaustive_robust.sum())}/{n} "
          f"({100*exhaustive_robust.mean():.1f}%)")
    print(f"JSMA robust:       {sum(jsma_survivors)}/{n} "
          f"({100*sum(jsma_survivors)/n:.1f}%)")
    print(f"Agreement:         {agree_robust+agree_vulnerable}/{n} "
          f"({100*agreement:.1f}%)")
    print(f"  Both robust:     {agree_robust}")
    print(f"  Both vulnerable: {agree_vulnerable}")
    print(f"Divergence:        {n - agree_robust - agree_vulnerable}")
    print(f"  JSMA=robust, Exhaustive=vulnerable: "
          f"{jsma_robust_exhaustive_vulnerable} "
          "(JSMA overestimates robustness)")
    print(f"  JSMA=vulnerable, Exhaustive=robust: "
          f"{jsma_vulnerable_exhaustive_robust} "
          "(JSMA underestimates robustness)")
    print(f"\nTiming:")
    print(f"  Exhaustive: {t_exhaustive:.2f}s "
          f"({t_exhaustive/n:.4f}s/sample, {num_enumerated} PGD passes/sample)")
    print(f"  JSMA:       {t_jsma:.2f}s "
          f"({t_jsma/n:.4f}s/sample, 1 PGD pass/sample)")
    print(f"  Speedup:    {t_exhaustive/t_jsma:.1f}x")

    avg_flips = sum(jsma_flips_per_sample) / max(len(jsma_flips_per_sample), 1)
    bc = torch.tensor(jsma_flips_per_sample).bincount()
    dist = {int(k): int(v) for k, v in enumerate(bc)}
    print(f"\nJSMA flips: avg={avg_flips:.2f}, distribution={dist}")

    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "checkpoint_sha256": sha,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "K": args.K,
        "num_samples": n,
        "clean_accuracy_pct": round(100 * clean_total / max(seen, 1), 4),
        "exhaustive": {
            "num_candidates": num_enumerated,
            "wallclock_seconds": round(t_exhaustive, 3),
            "per_sample_seconds": round(t_exhaustive / n, 6),
            "robust_count": int(exhaustive_robust.sum()),
            "robust_pct": round(100 * float(exhaustive_robust.mean()), 4),
        },
        "jsma": {
            "wallclock_seconds": round(t_jsma, 3),
            "per_sample_seconds": round(t_jsma / n, 6),
            "robust_count": sum(jsma_survivors),
            "robust_pct": round(100 * sum(jsma_survivors) / n, 4),
            "avg_flips": round(avg_flips, 2),
        },
        "comparison": {
            "agreement_pct": round(100 * agreement, 4),
            "agree_robust": int(agree_robust),
            "agree_vulnerable": int(agree_vulnerable),
            "divergence_count": int(n - agree_robust - agree_vulnerable),
            "jsma_overestimates_robustness": int(jsma_robust_exhaustive_vulnerable),
            "jsma_underestimates_robustness": int(jsma_vulnerable_exhaustive_robust),
        },
        "sample_records": [
            {
                "sample_id": si,
                "exhaustive_robust": bool(exhaustive_robust[si]),
                "jsma_robust": jsma_survivors[si],
                "agree": bool(exhaustive_robust[si]) == jsma_survivors[si],
            }
            for si in range(n)
        ],
    }

    if args.out is None:
        args.out = f"results/jsma_vs_exhaustive_{safe_name}_K{args.K}.json"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
