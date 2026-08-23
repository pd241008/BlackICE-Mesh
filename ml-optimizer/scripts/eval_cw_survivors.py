"""Exhaustive K=1 C&W evaluation with survivor-set capture.

Runs the canonical K=1 protocol (AND over continuous-only and every
single-flip state) for both C&W (cw_exhaustive_k1_survivors) and the
canonical PGD path (mirroring eval_unified.py), records per-batch raw
output + checkpoint provenance + perturbation magnitudes, and persists
per-sample survivor masks for nested-set comparison.
"""
import argparse
import hashlib
import json
import os
import time

import torch

from app.ml.attacks.cw import cw_exhaustive_k1_survivors
from app.ml.attacks.unified_pgd import unified_pgd_attack
from app.ml.data.loader import get_test_loader, get_config
from app.ml.models.architecture import TabularMLP
from app.ml.utils.checkpoint import load_model_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 0.15
ALPHA_CONT = 0.01
PGD_STEPS = 10


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pgd_k1_survivors(model, data, target, config):
    """Mirror of eval_unified.py K=1: base K=0 attack AND every group pass."""
    adv0 = unified_pgd_attack(
        model, data, target,
        epsilon=EPSILON, alpha=ALPHA_CONT, alpha_cat=0.0, steps=PGD_STEPS,
        continuous_cols=config.CONTINUOUS_COLS, categorical_groups=[],
    )
    with torch.no_grad():
        survivors = model(adv0).argmax(dim=1) == target
    for group in config.CATEGORICAL_GROUPS:
        adv = unified_pgd_attack(
            model, data, target,
            epsilon=EPSILON, alpha=ALPHA_CONT, alpha_cat=1.0, steps=PGD_STEPS,
            continuous_cols=config.CONTINUOUS_COLS, categorical_groups=[group],
        )
        with torch.no_grad():
            survivors &= model(adv).argmax(dim=1) == target
    return survivors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cicids2017")
    p.add_argument("--method", default="rsc")
    p.add_argument("--seed", type=int, default=53)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--binary-search-steps", type=int, default=3)
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()

    config = get_config(args.dataset)
    safe_name = args.dataset.replace("-", "_")
    tag = f"{args.method}_seed{args.seed}"
    ckpt = f"models/unified/model_adv_{args.method}_{safe_name}_seed{args.seed}.pth"
    assert os.path.exists(ckpt), f"missing checkpoint {ckpt}"

    out_dir = "results/cw"
    os.makedirs(out_dir, exist_ok=True)
    result_file = f"{out_dir}/k1_survivors_{safe_name}_{tag}.json"
    masks_file = f"{out_dir}/masks_{safe_name}_{tag}.pt"

    header = {
        "checkpoint": ckpt,
        "checkpoint_sha256": _sha256(ckpt),
        "dataset": args.dataset,
        "attack_config": {
            "epsilon": EPSILON, "cw_steps": args.steps,
            "cw_binary_search_steps": args.binary_search_steps,
            "pgd_steps": PGD_STEPS, "pgd_alpha": ALPHA_CONT,
            "semantics": "K=1 survivor = AND(K=0 survival, every single-flip "
                         "state survival); clean-correct samples only",
        },
    }

    loader = get_test_loader(args.dataset, batch_size=args.batch_size)
    if args.limit_batches is not None:
        loader = [b for i, b in zip(range(args.limit_batches), loader)]

    start_batch = 0
    state = {"batches": [], "cw_masks": [], "pgd_masks": [],
             "clean_correct": 0, "attacked": 0}
    if os.path.exists(result_file) and not args.fresh:
        with open(result_file) as f:
            saved = json.load(f)
        start_batch = saved["completed_batches"]
        state["batches"] = saved["batches"]
        state["clean_correct"] = saved["clean_correct"]
        state["attacked"] = saved["attacked"]
        if os.path.exists(masks_file):
            blob = torch.load(masks_file, weights_only=False)
            # Keep masks for batches already counted in completed_batches.
            state["cw_masks"] = blob["cw"][:start_batch]
            state["pgd_masks"] = blob["pgd"][:start_batch]
            assert len(state["cw_masks"]) >= start_batch or not blob["cw"], \
                "mask file out of sync with result file"
        print(f"[RESUME] from batch {start_batch}")

    model = TabularMLP(input_dim=config.FEATURE_DIM).to(DEVICE)
    load_model_checkpoint(model, ckpt, device=DEVICE)
    model.eval()
    cont_cols = list(config.CONTINUOUS_COLS)

    def flush(completed):
        doc = dict(header)
        doc.update({
            "completed_batches": completed,
            "clean_correct": state["clean_correct"],
            "attacked": state["attacked"],
            "batches": state["batches"],
        })
        with open(result_file, "w") as f:
            json.dump(doc, f, indent=1)
        torch.save({"cw": state["cw_masks"], "pgd": state["pgd_masks"]},
                   masks_file)

    t_start = time.perf_counter()
    for bi, (bx, by) in enumerate(loader):
        if bi < start_batch:
            continue
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        with torch.no_grad():
            clean_ok = model(bx).argmax(dim=1) == by
        state["clean_correct"] += int(clean_ok.sum())
        data, target = bx[clean_ok], by[clean_ok]
        rec = {"batch_idx": bi, "n_clean_ok": int(clean_ok.sum())}

        if data.size(0) > 0:
            # CW side (canonical AND-semantics).
            t0 = time.perf_counter()
            cw_surv = cw_exhaustive_k1_survivors(
                model, data, target, config, epsilon=EPSILON,
                steps=args.steps,
                binary_search_steps=args.binary_search_steps,
                chunk_size=8192,
            )
            dt_cw = time.perf_counter() - t0

            # PGD side (canonical mirror).
            pgd_surv = pgd_k1_survivors(model, data, target, config)

            rec.update({
                "cw_k1_robust": int(cw_surv.sum()),
                "pgd_k1_robust": int(pgd_surv.sum()),
                "both": int((cw_surv & pgd_surv).sum()),
                "pgd_only": int((pgd_surv & ~cw_surv).sum()),
                "cw_only": int((cw_surv & ~pgd_surv).sum()),
                "seconds_cw": round(dt_cw, 1),
            })
            state["cw_masks"].append(cw_surv.cpu())
            state["pgd_masks"].append(pgd_surv.cpu())
            state["attacked"] += data.size(0)

        state["batches"].append(rec)
        flush(bi + 1)
        eta_min = (time.perf_counter() - t_start) / max(1, bi - start_batch + 1) \
            * (len(loader) - bi - 1) / 60
        print(f"batch {bi+1}/{len(loader)} {rec} ETA={eta_min:.0f}min", flush=True)

    # Final nesting summary.
    cw_all = torch.cat(state["cw_masks"]) if state["cw_masks"] else torch.empty(0, dtype=torch.bool)
    pgd_all = torch.cat(state["pgd_masks"]) if state["pgd_masks"] else torch.empty(0, dtype=torch.bool)
    if len(cw_all):
        both = int((cw_all & pgd_all).sum())
        summary = {
            "attacked": int(len(cw_all)),
            "cw_k1_robust_pct": round(float(cw_all.float().mean()) * 100, 4),
            "pgd_k1_robust_pct": round(float(pgd_all.float().mean()) * 100, 4),
            "survive_both": both,
            "survive_pgd_only": int((pgd_all & ~cw_all).sum()),
            "survive_cw_only": int((cw_all & ~pgd_all).sum()),
            "jaccard": round(both / int((cw_all | pgd_all).sum()), 6),
        }
        doc = dict(header); doc.update({"summary": summary})
        with open(result_file.replace(".json", "_SUMMARY.json"), "w") as f:
            json.dump(doc, f, indent=1)
        print("SUMMARY:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
