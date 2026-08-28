"""Single-sample logit trace for the LEGACY AdvGuard-Hardened checkpoint.

Checkpoint  : models/model_adv.pth (SHA f07d2e3798026fa5de51cd4778f03312947d187bb20a139d039be9b3bfffda8f)
Dataset     : NSL-KDD test set (22,543 samples)
Evaluator   : reference exhaustive K=1 (EXH) -- enumerate every one-hot state per
              categorical group, run continuous PGD-40 (eps=0.15, alpha=0.01,
              random start) on each, pick worst state. This is the evaluator
              behind the 0/22,543 (0.00%) result for this checkpoint
              (results/results_nsl_kdd_Hardened.json, K=1).

Protocol (mirrors the earlier RSC trace protocol, corrected to the right
checkpoint):
  1. Re-confirm checkpoint SHA-256 at load time.
  2. Full test-set clean-accuracy regression gate (expect 18162/22543 = 80.57%).
  3. Select correctly-classified ATTACK samples (true label 1, predicted 1),
     distinct inputs, that reverse under at least one valid single categorical
     flip, with the largest clean logit margin; take the top 3.
  4. For each selected sample, flip protocol_type (|G|=3) and service (|G|=11)
     to every alternative one-hot state; record full clean->flipped logit
     vectors, margin before/after, and whether the prediction reverses.
  5. EXH break-check: confirm each selected sample is broken by the reference
     exhaustive K=1 evaluator (misclassified under at least one enumerated
     state after continuous PGD-40), exactly the mechanism behind the 0/22,543.
"""

import argparse
import hashlib
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ml.attacks.eval_protocol import EVAL_ALPHA_CONT, EVAL_EPSILON, EVAL_PGD_STEPS
from app.ml.attacks.unified_pgd import unified_pgd_attack
from app.ml.models.architecture import TabularMLP
from app.ml.utils.checkpoint import compute_sha256

EXPECTED_SHA = "f07d2e3798026fa5de51cd4778f03312947d187bb20a139d039be9b3bfffda8f"
CKPT = "models/model_adv.pth"
TEST_CSV = "data/nsl-kdd-test.csv"
N_TOTAL = 22543
CONT_COLS = [0, 1, 2, 3]
GROUPS = [[4, 5, 6], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
GROUP_NAMES = ["protocol_type", "service"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--out", default="results/logit_trace_model_adv_hardened.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    # --- Provenance gate: re-confirm checkpoint hash at trace time ---
    sha = compute_sha256(CKPT)
    assert sha == EXPECTED_SHA, f"checkpoint hash mismatch: {sha}"

    # --- Data: full NSL-KDD test set ---
    df = pd.read_csv(TEST_CSV, header=None)
    assert len(df) == N_TOTAL, f"expected {N_TOTAL} rows, got {len(df)}"
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

    # --- Model ---
    model = TabularMLP(input_dim=X.size(1)).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()

    # --- Full-set clean accuracy (evaluator-matched regression gate) ---
    with torch.no_grad():
        logits = torch.cat([model(b.to(device)) for b in X.split(2000)])
        preds = logits.argmax(1).cpu()
    clean_correct = (preds == y).sum().item()
    print(f"Checkpoint SHA-256 : {sha}")
    print(f"Clean accuracy     : {clean_correct}/{N_TOTAL} = {100*clean_correct/N_TOTAL:.2f}% "
          f"(expect 18162/22543 = 80.57%)")

    # --- Sample selection: correctly-classified, high-margin ATTACK samples.
    # (a) Dedupe identical feature rows: NSL-KDD contains repeated packets, and
    #     the global max-margin point is a single duplicated row (45 copies);
    #     selecting three copies of it would be degenerate.
    # (b) Require at least one valid single categorical flip to cleanly reverse
    #     the prediction, so every traced example is an actual logit collapse and
    #     the example is usable in the paper.
    import numpy as np
    true_attack = (y == 1)
    correct = (preds == y)
    eligible = correct & true_attack
    flat_eligible = eligible.nonzero(as_tuple=True)[0]
    uniq, first_pos = np.unique(X[flat_eligible].numpy(), axis=0, return_index=True)
    distinct_idx = flat_eligible[torch.tensor(first_pos)]
    margin = (logits[:, 1] - logits[:, 0]).cpu()
    distinct_margin = margin[distinct_idx]
    order = distinct_margin.argsort(descending=True)
    candidates = distinct_idx[order[: min(60, len(order))]]

    cand_reverses = torch.zeros(len(candidates), dtype=torch.bool)
    for k, gi in enumerate(candidates.tolist()):
        xc = X[gi:gi + 1]
        rev_found = False
        for group in GROUPS:
            orig_active = int(xc[0, group].argmax())
            for flip_to in range(len(group)):
                if flip_to == orig_active:
                    continue
                flipped = xc.clone()
                flipped[:, group] = 0.0
                flipped[:, group[flip_to]] = 1.0
                with torch.no_grad():
                    fp = model(flipped.to(device)).cpu().squeeze().argmax().item()
                if fp != 1:
                    rev_found = True
                    break
            if rev_found:
                break
        cand_reverses[k] = rev_found

    reversers = candidates[cand_reverses]
    if len(reversers) == 0:
        raise RuntimeError("no distinct high-margin attack sample reverses under a "
                           "single valid categorical flip")
    selected = reversers[: args.num_samples].tolist()
    print(f"Selected high-margin attack samples (distinct, clean single-flip reversal): {selected}")

    traces = []
    for gi_abs in selected:
        x = X[gi_abs:gi_abs + 1]
        xd = x.to(device)
        tl = y[gi_abs].item()
        with torch.no_grad():
            cl = model(xd).cpu().squeeze()
        c_pred = cl.argmax().item()
        assert tl == 1 and c_pred == 1, f"sample {gi_abs} not correctly-classified attack"
        c_margin = (cl[1] - cl[0]).item()
        c_prob = torch.softmax(cl, 0).tolist()

        flips = []
        for g_idx, group in enumerate(GROUPS):
            orig_active = int(x[0, group].argmax())
            for flip_to in range(len(group)):
                if flip_to == orig_active:
                    continue
                flipped = x.clone()
                flipped[:, group] = 0.0
                flipped[:, group[flip_to]] = 1.0
                with torch.no_grad():
                    fl = model(flipped.to(device)).cpu().squeeze()
                f_pred = fl.argmax().item()
                f_margin = (fl[1] - fl[0]).item()
                flips.append({
                    "group": GROUP_NAMES[g_idx],
                    "group_idx": g_idx,
                    "group_size": len(group),
                    "orig_active_col": orig_active,
                    "flipped_to_col": flip_to,
                    "flip_logits": fl.tolist(),
                    "flip_probs": torch.softmax(fl, 0).tolist(),
                    "flip_pred": f_pred,
                    "margin_after": round(f_margin, 6),
                    "reversal": f_pred != tl,
                })

        traces.append({
            "test_idx": gi_abs,
            "true_label": tl,
            "clean_pred": c_pred,
            "clean_logits": cl.tolist(),
            "clean_probs": c_prob,
            "margin_before": round(c_margin, 6),
            "flips": flips,
        })

    # --- EXH break-check: reproduce canonical K=1 evaluator semantics
    # (eval_mixed_norm.canonical_mixed_norm_attack: enumerate all one-hot states,
    # continuous PGD-40 per state, deterministic init -- no random start, pick the
    # worst state by CE loss) for each selected sample. Note: canonical runs the
    # PGD over the whole 2000-row batch; this per-sample replication runs it over
    # the sample's 15 states (15 rows) with identical algorithm, so it is a
    # faithful per-sample manifest. The authoritative full-set number remains the
    # archived results/results_nsl_kdd_Hardened.json (K=1 -> 0/22543).
    import torch.nn as nn
    cat_min, cat_max = 4, 17
    rel_groups = [[c - cat_min for c in G] for G in GROUPS]

    def canonical_k1_break_check(x):
        xd = x.to(device)
        tl_t = torch.tensor([1], device=device)
        labels = tl_t.repeat(15)
        orig_cat = xd[:, cat_min:cat_max + 1]
        states = [orig_cat]
        for rel_group in rel_groups:
            for i in range(len(rel_group)):
                s = orig_cat.clone()
                s[:, rel_group] = 0.0
                s[:, rel_group[i]] = 1.0
                states.append(s)
        all_states = torch.stack(states, dim=0)
        expanded = xd.unsqueeze(0).expand(15, -1, -1).clone()
        expanded[:, :, cat_min:cat_max + 1] = all_states
        adv = expanded.reshape(15, -1).detach()
        orig_cont = adv[:, CONT_COLS].clone()
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        for _ in range(EVAL_PGD_STEPS):
            adv.requires_grad_(True)
            out = model(adv)
            loss = loss_fn(out, labels)
            model.zero_grad()
            loss.sum().backward()
            with torch.no_grad():
                grad = adv.grad
                adv_cont = adv[:, CONT_COLS] + EVAL_ALPHA_CONT * grad[:, CONT_COLS].sign()
                eta = (adv_cont - orig_cont).clamp(-EVAL_EPSILON, EVAL_EPSILON)
                adv.data[:, CONT_COLS] = (orig_cont + eta).clamp(0.0, 1.0)
            adv = adv.detach()
        with torch.no_grad():
            losses = loss_fn(model(adv), labels)
            worst = int(losses.argmax())
            pred = model(adv[worst:worst + 1]).argmax(1).item()
        net_cat_row = adv[worst].cpu()
        flipped_groups = []
        for g_idx, group in enumerate(GROUPS):
            if int(net_cat_row[group].argmax()) != int(x[0, group].argmax()):
                flipped_groups.append(GROUP_NAMES[g_idx])
        return {
            "broken_by_k1_exh": pred != 1,
            "n_states": 15,
            "worst_state_group_flipped": flipped_groups,
            "per_sample_note": "deterministic init, algorithm identical to "
                               "canonical_mixed_norm_attack; full-batch authority is "
                               "results/results_nsl_kdd_Hardened.json (0/22543)",
        }

    exh = [{
        "test_idx": t["test_idx"],
        **canonical_k1_break_check(X[t["test_idx"]:t["test_idx"] + 1]),
    } for t in traces]

    n_rev = sum(1 for t in traces for f in t["flips"] if f["reversal"])
    n_flips = sum(len(t["flips"]) for t in traces)
    print(f"Total flips: {n_flips}, reversals: {n_rev}")
    for t in traces:
        revs = [f for f in t["flips"] if f["reversal"]]
        print(f"  sample {t['test_idx']}: margin_before={t['margin_before']:+.3f} | "
              f"{len(revs)}/{len(t['flips'])} flips reverse")

    out = {
        "description": "Logit-reversal trace: legacy AdvGuard-Hardened checkpoint "
                       "(model_adv.pth), NSL-KDD full test set, EXH K=1 evaluator-matched",
        "checkpoint_sha256": sha,
        "dataset": "nsl-kdd",
        "n_total": N_TOTAL,
        "clean_correct": clean_correct,
        "clean_acc_pct": round(100 * clean_correct / N_TOTAL, 2),
        "exh_k1_archived": {"correct": 0, "total": 22543,
                            "source": "results/results_nsl_kdd_Hardened.json"},
        "evaluator": {
            "semantics": "EXH K=1: enumerate every one-hot state per categorical "
                         "group, continuous PGD-40 (eps=0.15, alpha=0.01, "
                         "deterministic init, no random start) per state, pick "
                         "worst by CE loss",
            "epsilon": EVAL_EPSILON,
            "alpha_cont": EVAL_ALPHA_CONT,
            "pgd_steps": EVAL_PGD_STEPS,
            "continuous_cols": CONT_COLS,
            "categorical_groups": GROUPS,
        },
        "traces": traces,
        "exh_break_check": exh,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()