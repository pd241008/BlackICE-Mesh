"""Single-sample logit trace for NSL-KDD categorical flips.

Loads a trained checkpoint, picks correctly-classified samples, flips
protocol_type and service to every alternative one-hot state, and logs
the logit vectors before and after each flip. Reports whether the
prediction reverses.

Produces the raw data for Section IV-C's categorical-flip example.
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ml.models.architecture import TabularMLP
from app.ml.data.loader import get_test_loader, get_config
from app.ml.utils.checkpoint import load_model_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nsl-kdd")
    p.add_argument("--method", default="rsc")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-samples", type=int, default=20,
                   help="number of correctly-classified samples to trace")
    p.add_argument("--out", default="results/logit_trace_nsl_kdd.json")
    args = p.parse_args()

    config = get_config(args.dataset)
    safe_name = args.dataset.replace("-", "_")
    ckpt = f"models/unified/model_adv_{args.method}_{safe_name}_seed{args.seed}.pth"
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    model = TabularMLP(input_dim=config.FEATURE_DIM)
    load_model_checkpoint(model, ckpt, device="cpu")
    model.eval()

    loader = get_test_loader(args.dataset, batch_size=2000)
    groups = [list(g) for g in config.CATEGORICAL_GROUPS]

    # Collect correctly-classified samples
    traces = []
    for bx, by in loader:
        with torch.no_grad():
            logits = model(bx)
            preds = logits.argmax(1)
        ok = preds == by
        if ok.sum() == 0:
            continue
        correct_idxs = ok.nonzero(as_tuple=True)[0]
        for idx in correct_idxs:
            if len(traces) >= args.num_samples:
                break
            x = bx[idx:idx+1]
            y = by[idx:idx+1]
            with torch.no_grad():
                clean_logits = model(x)
            clean_pred = clean_logits.argmax(1).item()
            clean_prob = torch.softmax(clean_logits, dim=1).squeeze()
            traces.append({
                "sample_idx": len(traces),
                "true_label": y.item(),
                "clean_pred": clean_pred,
                "clean_logits": clean_logits.squeeze().tolist(),
                "clean_probs": clean_prob.tolist(),
                "flips": []
            })
        if len(traces) >= args.num_samples:
            break

    print(f"Collected {len(traces)} correctly-classified samples")

    # For each sample, flip each categorical group to every alternative state
    for t in traces:
        idx = t["sample_idx"]
        x = bx[idx:idx+1]
        y = by[idx:idx+1]

        for g_idx, group in enumerate(groups):
            orig_onehot = x[:, group].clone()
            orig_active = orig_onehot.argmax(1).item()

            for flip_to in range(len(group)):
                if flip_to == orig_active:
                    continue  # skip original state
                flipped = x.clone()
                flipped[:, group] = 0.0
                flipped[:, group[flip_to]] = 1.0
                with torch.no_grad():
                    flip_logits = model(flipped)
                flip_pred = flip_logits.argmax(1).item()
                flip_prob = torch.softmax(flip_logits, dim=1).squeeze()
                reversal = flip_pred != y.item()
                t["flips"].append({
                    "group_idx": g_idx,
                    "group_size": len(group),
                    "orig_active_col": orig_active,
                    "flipped_to_col": flip_to,
                    "flip_logits": flip_logits.squeeze().tolist(),
                    "flip_probs": flip_prob.tolist(),
                    "flip_pred": flip_pred,
                    "reversal": reversal
                })

    # Print summary
    total_flips = sum(len(t["flips"]) for t in traces)
    total_reversals = sum(1 for t in traces for f in t["flips"] if f["reversal"])
    print(f"Total flips: {total_flips}, Reversals: {total_reversals} ({100*total_reversals/max(total_flips,1):.1f}%)")

    # Print a concrete example for the paper (first reversal found, or first flip)
    for t in traces:
        for f in t["flips"]:
            if f["reversal"]:
                gname = ['protocol_type','service'][f['group_idx']] if f['group_idx'] < 2 else f"G{f['group_idx']}"
                print(f"\n--- REVERSAL EXAMPLE ---")
                print(f"Sample {t['sample_idx']}, Group {f['group_idx']} ({gname})"
                      f", flip col {f['orig_active_col']} -> {f['flipped_to_col']}")
                print(f"Clean logits: {t['clean_logits']}")
                print(f"Clean pred: {t['clean_pred']}, true: {t['true_label']}")
                print(f"Flip logits: {f['flip_logits']}")
                print(f"Flip pred: {f['flip_pred']}")
                break
        else:
            continue
        break

    # Also print per-group reversal rates
    for g_idx, group in enumerate(groups):
        group_name = ["protocol_type", "service"][g_idx] if g_idx < 2 else f"G{g_idx}"
        flips_g = [f for t in traces for f in t["flips"] if f["group_idx"] == g_idx]
        revs_g = [f for f in flips_g if f["reversal"]]
        print(f"Group {g_idx} ({group_name}, |G|={len(group)}): "
              f"{len(flips_g)} flips, {len(revs_g)} reversals "
              f"({100*len(revs_g)/max(len(flips_g),1):.1f}%)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(traces, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
