"""State-exhaustive interval-bound-propagation certificate for the mixed-norm threat model.

Certified-K1(sample) := clean-correct AND for EVERY enumerated categorical state s
(original + every single-column one-hot state per group, matching
cw_exhaustive_k1_survivors enumeration): the L-inf interval neighborhood of the
continuous columns at state s is CERTIFIED robust, i.e. interval propagation
through the network yields min-margin > 0:

    min_{x' in [x_lo, x_hi]} (z_true(x') - max_{o != true} z_o(x')) > 0

where [x_lo, x_hi] clamps x +/- EPSILON into [0,1] on continuous columns and the
categorical block is held EXACTLY at state s (discrete, no relaxation).

IBP through TabularMLP (fc1 -> BN(eval, frozen affine) -> ReLU -> fc2 -> ReLU -> fc3)
is exact interval arithmetic: affine layers transform intervals coordinate-wise by
the sign of their weights; ReLU maps [lo,hi] -> [relu(lo), relu(hi)]. Sound by
construction; looseness comes only from cross-feature independence.

Correctness gates:
  - any certified sample MUST be in the persisted empirical PGD-40 exhaustive
    survivor masks (certified subset of attacked-robust); violation => BUG.
Constants imported from app.ml.attacks.eval_protocol.
"""

import argparse
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ml.attacks.eval_protocol import EVAL_EPSILON
from app.ml.data.loader import get_test_loader, get_config
from app.ml.models.architecture import TabularMLP
from app.ml.utils.checkpoint import load_model_checkpoint


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def ibp_margin_lower_bounds(model, x_lo, x_hi, true_cls):
    """Lower bound of (z_true - max_other) over the input box, per sample."""
    def affine(l, u, w, b):
        cent = (l + u) / 2
        rad = (u - l) / 2
        wc = w @ cent.unsqueeze(-1) if w.dim() == 2 else None
        # vectorized: out_cent = W cent + b ; out_rad = |W| rad
        out_c = F.linear(cent, w, b)
        out_r = F.linear(rad, w.abs())
        return out_c - out_r, out_c + out_r

    l1, u1 = affine(x_lo, x_hi, model.fc1.weight, model.fc1.bias)
    # BN eval: y = gamma * (z - mu) / sqrt(var+eps) + beta  (elementwise affine)
    gam = model.bn1.weight / torch.sqrt(model.bn1.running_var + model.bn1.eps)
    bet = model.bn1.bias - gam * model.bn1.running_mean
    shift = gam >= 0
    l1b = torch.where(shift, l1, u1) * gam + bet
    u1b = torch.where(shift, u1, l1) * gam + bet
    l2, u2 = torch.relu(l1b), torch.relu(u1b)
    l3, u3 = affine(l2, u2, model.fc2.weight, model.fc2.bias)
    l4, u4 = torch.relu(l3), torch.relu(u3)
    lf, uf = affine(l4, u4, model.fc3.weight, model.fc3.bias)

    k = true_cls
    lf_true = lf[torch.arange(lf.size(0)), k]
    uf_masked = uf.clone()
    uf_masked[torch.arange(uf.size(0)), k] = -float('inf')
    uf_other_max = uf_masked.max(dim=1).values
    return lf_true - uf_other_max


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cicids2017")
    p.add_argument("--method", default="rsc")
    p.add_argument("--seed", type=int, default=53)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    config = get_config(args.dataset)
    safe_name = args.dataset.replace("-", "_")
    tag = f"{args.method}_seed{args.seed}"
    ckpt = f"models/unified/model_adv_{args.method}_{safe_name}_seed{args.seed}.pth"
    sha = sha256_file(ckpt)
    print(f"[HASH] {sha}", flush=True)

    ref_path = f"results/foolbox/masks_exh_k1_pgd40_{safe_name}_{tag}.pt"
    ref_masks = torch.load(ref_path, weights_only=False)["k1"] \
        if os.path.exists(ref_path) else None
    if ref_masks is not None:
        print(f"[REF] empirical pgd40-exhaustive masks: {len(ref_masks)} batches", flush=True)

    model = TabularMLP(input_dim=config.FEATURE_DIM)
    load_model_checkpoint(model, ckpt, device='cpu')
    model.eval()
    groups = [list(g) for g in config.CATEGORICAL_GROUPS]
    EPS = EVAL_EPSILON

    KNOWN_TOTALS = {"cicids2017": 623869, "unsw_nb15": 508010, "nsl-kdd": 22543}
    expected_n = KNOWN_TOTALS.get(safe_name)

    n_total = clean_correct = attacked = 0
    cert_k0 = cert_k1 = 0
    mask_cursor = 0
    gate_violations = []
    records = []

    loader = get_test_loader(args.dataset, batch_size=args.batch_size)
    with torch.no_grad():
        for bi, (bx, by) in enumerate(loader):
            ok = model(bx).argmax(1) == by
            data, target = bx[ok], by[ok]
            n_total += bx.size(0); clean_correct += int(ok.sum())
            n = data.size(0)
            if n == 0:
                continue
            ar = torch.arange(n)

            x_lo_base, x_hi_base = data.clone(), data.clone()
            x_lo_base[:, config.CONTINUOUS_COLS] = (data[:, config.CONTINUOUS_COLS] - EPS).clamp(0, 1)
            x_hi_base[:, config.CONTINUOUS_COLS] = (data[:, config.CONTINUOUS_COLS] + EPS).clamp(0, 1)

            m_base = ibp_margin_lower_bounds(model, x_lo_base, x_hi_base, target) > 0
            m_all = m_base.clone()
            for group in groups:
                for j in group:
                    slo, shi = x_lo_base.clone(), x_hi_base.clone()
                    slo[:, group] = 0.0; shi[:, group] = 0.0
                    slo[:, j] = 1.0; shi[:, j] = 1.0
                    m_all &= ibp_margin_lower_bounds(model, slo, shi, target) > 0

            cert_k0 += int(m_base.sum()); cert_k1 += int(m_all.sum()); attacked += n

            if ref_masks is not None:
                emp = ref_masks[mask_cursor].bool()
                mask_cursor += 1
                viol = int((m_all & ~emp).sum())
                if viol:
                    gate_violations.append({"batch_idx": bi, "violations": viol})
            records.append({"batch_idx": bi, "n": int(n),
                            "cert_k0": int(m_base.sum()), "cert_k1": int(m_all.sum())})

    if expected_n is not None:
        assert n_total == expected_n, (
            f"n_total={n_total} != known test-set size {expected_n} for {safe_name}; "
            f"accumulator corruption suspected (resume double-count?)")

    summary = {
        "checkpoint": {"path": ckpt, "sha256": sha},
        "certificate": "state-exhaustive IBP: certified-K1 iff certified-Linf at base AND every one-hot state",
        "epsilon": EPS,
        "n_total": n_total, "n_attacked": attacked,
        "certified_k0_pct_of_attacked": round(100 * cert_k0 / max(attacked, 1), 4),
        "certified_k1_pct_of_attacked": round(100 * cert_k1 / max(attacked, 1), 4),
        "gate": {"ref": "pgd40-exhaustive masks", "violations_total": sum(v["violations"] for v in gate_violations),
                 "detail": gate_violations[:5]},
        "records": records,
    }
    out = f"results/certificates/cert_ibp_{safe_name}_{tag}.json"
    os.makedirs("results/certificates", exist_ok=True)
    json.dump(summary, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
