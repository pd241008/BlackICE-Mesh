"""Exhaustive-state K=1 evaluation with third-party / alternative optimizers.

Fixes a semantics asymmetry found Aug 23: cw_exhaustive_k1_survivors enumerates
EVERY one-hot state per categorical group (base + |G| state passes), while
eval_unified.py's K=1 anchor makes ONE gradient-snapped flip pass per group.
Cross-attack comparisons must hold semantics fixed, so this script evaluates
DeepFool (Foolbox Linf variant) AND plain PGD-40 under the SAME exhaustive
enumeration as the CW harness.

Survivor := clean-correct AND survives base pass AND survives EVERY enumerated
state pass (sample broken if ANY optimizer/state breaks it).

DeepFool notes (locked design decisions):
  - LinfDeepFoolAttack output projected onto the eps ball before scoring
    (weakening-only; asserted per pass).
  - Scoring = misclassification of final adversarial example; DeepFool's
    internal minimal-distance objective never used for selection.

Nesting references loaded from results/cw/masks_<ds>_<tag>.pt (cw = exhaustive;
pgd = SNAP semantics — labeled asymmetric, do not mix up in tables).
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch
import foolbox as fb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ml.attacks.unified_pgd import unified_pgd_attack
from app.ml.attacks.eval_protocol import EVAL_EPSILON, EVAL_ALPHA_CONT, EVAL_PGD_STEPS
from app.ml.data.loader import get_test_loader, get_config
from app.ml.models.architecture import TabularMLP
from app.ml.utils.checkpoint import load_model_checkpoint
from scripts.eval_foolbox import ContinuousColumnWrapper, split_columns

EPS = EVAL_EPSILON


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", choices=["deepfool", "pgd40"], required=True)
    p.add_argument("--dataset", default="cicids2017")
    p.add_argument("--method", default="rsc")
    p.add_argument("--seed", type=int, default=53)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--df-steps", type=int, default=50)
    p.add_argument("--limit-batches", type=int, default=None,
                   help="process at most N eligible (attacked>0) batches")
    p.add_argument("--suffix", default="",
                   help="filename suffix for probe/subset runs (kept separate from main artifacts)")
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()

    config = get_config(args.dataset)
    safe_name = args.dataset.replace("-", "_")
    tag = f"{args.method}_seed{args.seed}"
    ckpt = f"models/unified/model_adv_{args.method}_{safe_name}_seed{args.seed}.pth"
    assert os.path.exists(ckpt), f"missing checkpoint {ckpt}"
    sha = sha256_file(ckpt)

    out_json = f"results/foolbox/exh_k1_{args.attack}_{safe_name}_{tag}{args.suffix}.json"
    out_masks = f"results/foolbox/masks_exh_k1_{args.attack}_{safe_name}_{tag}{args.suffix}.pt"
    os.makedirs("results/foolbox", exist_ok=True)

    model = TabularMLP(input_dim=config.FEATURE_DIM)
    load_model_checkpoint(model, ckpt, device='cpu')
    model.eval()
    cont_cols, cat_cols = split_columns(config)
    groups = [list(g) for g in config.CATEGORICAL_GROUPS]
    wrapper = ContinuousColumnWrapper(model, cont_cols, cat_cols)
    fmodel = fb.PyTorchModel(wrapper, bounds=(0.0, 1.0), device='cpu')
    df_attack = fb.attacks.LinfDeepFoolAttack(steps=args.df_steps) \
        if args.attack == "deepfool" else None

    def state_survival(x_full_state, target):
        """One optimizer pass with cat block frozen at x_full_state's values."""
        if args.attack == "pgd40":
            adv = unified_pgd_attack(
                model, x_full_state, target, epsilon=EPS, alpha=EVAL_ALPHA_CONT,
                alpha_cat=0.0, steps=EVAL_PGD_STEPS,
                continuous_cols=cont_cols, categorical_groups=[])
        else:
            wrapper.cat_fixed = x_full_state[:, cat_cols].detach()
            x_cont = x_full_state[:, cont_cols].detach()
            _, adv_fb, _ = df_attack(fmodel, x_cont, target, epsilons=None)
            delta = (adv_fb - x_cont).clamp(-EPS, EPS)
            assert delta.abs().max().item() <= EPS + 1e-5, "eps projection failed"
            adv = x_full_state.clone()
            adv[:, cont_cols] = (x_cont + delta).clamp(0.0, 1.0)
        with torch.no_grad():
            return model(adv).argmax(1) == target

    # resume scaffolding
    start_idx_set, agg = set(), None
    if os.path.exists(out_json) and not args.fresh:
        with open(out_json) as f:
            saved = json.load(f)
        start_idx_set = set(saved.get("processed_batch_idxs", []))
        agg = {k: saved[k] for k in ("n_total", "clean_correct", "attacked",
                                     "k0_survivors", "k1_survivors")}
        if os.path.exists(out_masks):
            masks = torch.load(out_masks, weights_only=False)["k1"]
        records = list(saved.get("records", []))
        print(f"[RESUME] {len(start_idx_set)} batches already done "
              f"({len(masks)} masks restored)", flush=True)
    if agg is None:
        agg = {"n_total": 0, "clean_correct": 0, "attacked": 0,
               "k0_survivors": 0, "k1_survivors": 0}
    masks, records, processed_order = [], [], []

    # reference masks (cursor-aligned by attacked-sample sequence)
    ref_path = f"results/cw/masks_{safe_name}_{tag}.pt"
    ref_cw = torch.load(ref_path, weights_only=False)["cw"] \
        if os.path.exists(ref_path) else None
    if ref_cw is not None:
        print(f"[REF] cw mask batches: {len(ref_cw)} (exhaustive semantics)", flush=True)
    done_count = len(start_idx_set)
    limit = args.limit_batches
    t0 = time.time()
    loader = get_test_loader(args.dataset, batch_size=args.batch_size)

    for bi, (bx, by) in enumerate(loader):
        if limit is not None and done_count >= limit:
            break
        with torch.no_grad():
            ok = model(bx).argmax(1) == by
        data, target = bx[ok], by[ok]
        n = data.size(0)
        if bi not in start_idx_set:
            agg["n_total"] += bx.size(0)
            agg["clean_correct"] += int(ok.sum())
        if n == 0 or bi in start_idx_set:
            continue

        target_y = target
        arange_n = torch.arange(n)
        surv_k0 = state_survival(data, target_y)
        surv_k1 = surv_k0.clone()
        for group in groups:
            gcols = [c for c in group]
            for j in gcols:  # exhaustive: every one-hot state (superset incl. original)
                state = data.clone()
                state[:, gcols] = 0.0
                state[:, j] = 1.0
                surv_k1 &= state_survival(state, target_y)

        m_cw = ref_cw[len(masks)] if ref_cw is not None else None
        rec = {"batch_idx": bi, "n": int(n),
               "k0": int(surv_k0.sum()), "k1": int(surv_k1.sum())}
        if m_cw is not None:
            rec.update({
                "vs_cw_both": int((surv_k1.cpu() & m_cw).sum()),
                "vs_cw_only_this": int((surv_k1.cpu() & ~m_cw).sum()),
                "vs_cw_only_cw": int((~surv_k1.cpu() & m_cw).sum()),
            })
        records.append(rec); masks.append(surv_k1.cpu()); processed_order.append(bi)
        done_count += 1
        agg["attacked"] += int(n)
        agg["k0_survivors"] += int(surv_k0.sum())
        agg["k1_survivors"] += int(surv_k1.sum())

        if done_count % 25 == 0 or (limit is not None and done_count >= limit):
            k1pct = round(100 * agg["k1_survivors"] / max(agg["attacked"], 1), 4)
            print(f"[{time.time()-t0:.0f}s] batches={done_count} attacked={agg['attacked']} "
                  f"k1_robust={agg['k1_survivors']} ({k1pct}% of attacked)", flush=True)
            json.dump({
                "checkpoint_sha256": sha, "attack_backend": args.attack,
                "config": {"df_steps": args.df_steps if args.attack == "deepfool" else None,
                           "pgd_steps": EVAL_PGD_STEPS if args.attack == "pgd40" else None,
                           "epsilon": EPS, "semantics": "exhaustive-state AND (base + every one-hot state per group)",
                           "batch_size": args.batch_size, "partial": True},
                **agg, "processed_batch_idxs": sorted(set(start_idx_set) | set(processed_order)),
                "records": records,
            }, open(out_json, 'w'))
            if masks:
                torch.save({"k1": masks}, out_masks + '.part')
                os.replace(out_masks + '.part', out_masks)

    flat = torch.cat(masks) if masks else torch.zeros(0, dtype=torch.bool)
    # Regression gate: a COMPLETE run must account for the full test set exactly.
    # Catches accumulator double-count bugs (e.g. resume-path n_total inflation).
    KNOWN_TOTALS = {"cicids2017": 623869, "unsw_nb15": 508010, "nsl-kdd": 22543}
    expected_n = KNOWN_TOTALS.get(args.dataset)
    is_complete = limit is None
    if is_complete and expected_n is not None:
        assert agg["n_total"] == expected_n, (
            f"n_total={agg['n_total']} != known test-set size {expected_n} for {args.dataset}; "
            f"accumulator corruption suspected (resume double-count?)")
    summary = {
        "checkpoint_sha256": sha, "attack_backend": args.attack,
        "config": {"df_steps": args.df_steps if args.attack == "deepfool" else None,
                   "pgd_steps": EVAL_PGD_STEPS if args.attack == "pgd40" else None,
                   "epsilon": EPS, "semantics": "exhaustive-state AND",
                   "batch_size": args.batch_size,
                   "partial": limit is not None},
        **agg,
        "k1_pct_of_attacked": round(100 * agg["k1_survivors"] / max(agg["attacked"], 1), 4),
        "processed_batch_idxs": sorted(set(start_idx_set) | set(processed_order)),
        "records": records,
    }
    json.dump(summary, open(out_json, 'w'))
    torch.save({"k1": masks}, out_masks)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
