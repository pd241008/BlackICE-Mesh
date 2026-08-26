"""Stage C: CROWN vs IBP comparison on real checkpoints — full mixed-norm threat model.

For each (dataset, method, seed) triple:
  1. Load checkpoint, run CROWN backward bound on clean-correct samples
  2. Enumerate all one-hot categorical states (K=1 certificate)
  3. Compare with IBP forward bound (same enumeration)
  4. Report K=0 and K=1 certified rates for both methods

CROWN ≥ IBP is required by theory (CROWN uses tighter linear relaxations).
"""
import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ml.attacks.eval_protocol import EVAL_EPSILON
from app.ml.data.loader import get_test_loader, get_config
from app.ml.models.architecture import TabularMLP
from app.ml.utils.checkpoint import load_model_checkpoint
from scripts.crown_bound import crown_margin_lower, crown_margin_lower_k1, config_cont_slice
from scripts.certified_bound import ibp_margin_lower_bounds

KNOWN_TOTALS = {"cicids2017": 623869, "unsw_nb15": 508010, "nsl_kdd": 22543}
EPS = EVAL_EPSILON


def ibp_k1(model, x_lo_base, x_hi_base, true_cls, groups):
    """IBP K=1 certificate: base AND every one-hot state."""
    k0 = ibp_margin_lower_bounds(model, x_lo_base, x_hi_base, true_cls)
    k1_min = k0.clone()
    for group in groups:
        for j in group:
            slo, shi = x_lo_base.clone(), x_hi_base.clone()
            slo[:, group] = 0.0; shi[:, group] = 0.0
            slo[:, j] = 1.0; shi[:, j] = 1.0
            m_state = ibp_margin_lower_bounds(model, slo, shi, true_cls)
            k1_min = torch.min(k1_min, m_state)
    return k0, k1_min


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-batches", type=int, default=None)
    args = p.parse_args()

    safe_name = args.dataset.replace("-", "_")
    tag = f"{args.method}_seed{args.seed}"
    ckpt = f"models/unified/model_adv_{args.method}_{safe_name}_seed{args.seed}.pth"
    if not os.path.exists(ckpt):
        print(json.dumps({"error": f"checkpoint not found: {ckpt}"}))
        sys.exit(1)

    sha = sha256_file(ckpt)
    config = get_config(args.dataset)
    model = TabularMLP(input_dim=config.FEATURE_DIM)
    load_model_checkpoint(model, ckpt, device='cpu')
    model.eval()
    groups = [list(g) for g in config.CATEGORICAL_GROUPS]

    config_cont_slice._cols = list(config.CONTINUOUS_COLS)

    n_total = 0
    crown_k0_bounds = []
    crown_k1_bounds = []
    ibp_k0_bounds = []
    ibp_k1_bounds = []

    loader = get_test_loader(args.dataset, batch_size=args.batch_size)
    with torch.no_grad():
        for bi, (bx, by) in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            ok = model(bx).argmax(1) == by
            data, target = bx[ok], by[ok]
            n = data.size(0)
            n_total += bx.size(0)
            if n == 0:
                continue

            cr_k0, cr_k1 = crown_margin_lower_k1(model, data, target, EPS, groups)

            x_lo = data.clone()
            x_hi = data.clone()
            x_lo[:, config.CONTINUOUS_COLS] = (data[:, config.CONTINUOUS_COLS] - EPS).clamp(0, 1)
            x_hi[:, config.CONTINUOUS_COLS] = (data[:, config.CONTINUOUS_COLS] + EPS).clamp(0, 1)
            ib_k0, ib_k1 = ibp_k1(model, x_lo, x_hi, target, groups)

            crown_k0_bounds.append(cr_k0)
            crown_k1_bounds.append(cr_k1)
            ibp_k0_bounds.append(ib_k0)
            ibp_k1_bounds.append(ib_k1)

    all_cr_k0 = torch.cat(crown_k0_bounds)
    all_cr_k1 = torch.cat(crown_k1_bounds)
    all_ib_k0 = torch.cat(ibp_k0_bounds)
    all_ib_k1 = torch.cat(ibp_k1_bounds)
    n = all_cr_k0.size(0)

    def stats(bounds):
        return {
            "mean": round(float(bounds.mean()), 4),
            "median": round(float(bounds.median()), 4),
            "std": round(float(bounds.std()), 4),
            "min": round(float(bounds.min()), 4),
            "pct_positive": round(100 * float((bounds > 0).sum()) / max(n, 1), 2),
        }

    summary = {
        "checkpoint": {"path": ckpt, "sha256": sha},
        "dataset": args.dataset, "method": args.method, "seed": args.seed,
        "epsilon": EPS, "threat_model": "mixed-norm: L∞ continuous + L0 categorical exhaustive",
        "n_total": n_total, "n_clean_correct": n,
        "crown_k0": stats(all_cr_k0),
        "crown_k1": stats(all_cr_k1),
        "ibp_k0": stats(all_ib_k0),
        "ibp_k1": stats(all_ib_k1),
        "soundness_check": {
            "crown_k0_le_exhaustive_empirical": "see comparison script",
            "crown_k1_le_ibp_k1": bool((all_cr_k1 <= all_ib_k1 + 1e-6).all()),
            "crown_k1_ge_ibp_k1": bool((all_cr_k1 >= all_ib_k1 - 1e-6).all()),
        },
    }

    os.makedirs("results/certificates", exist_ok=True)
    out_path = f"results/certificates/crown_vs_ibp_{safe_name}_{tag}.json"
    json.dump(summary, open(out_path, 'w'), indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
