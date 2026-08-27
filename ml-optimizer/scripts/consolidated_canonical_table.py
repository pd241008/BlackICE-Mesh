"""Canonical 5-seed consolidated evaluation: both conventions, identical seeds.

Reads existing results from:
  1. eval_deepfool_k1.py (exhaustive-state AND, α_cat=0.0) → clean-correct denominator
  2. eval_unified.py (gradient-snapped K=1, α_cat=1.0) → full test set denominator

Produces a single consolidated table with BOTH conventions on the SAME checkpoints.
No re-runs needed — pure aggregation of existing results.

Usage:
  python scripts/consolidated_canonical_table.py
"""

import json
import os
import sys

KNOWN_TOTALS = {"nsl_kdd": 22543, "cicids2017": 623869, "unsw_nb15": 508010}
DATASETS = ["nsl_kdd", "cicids2017", "unsw_nb15"]
METHODS = ["hardened", "curriculum", "rsc"]
METHOD_LABELS = {"hardened": "Hardened", "curriculum": "Curriculum", "rsc": "RSC"}
DS_LABELS = {"nsl_kdd": "NSL-KDD", "cicids2017": "CICIDS2017", "unsw_nb15": "UNSW-NB15"}

# Canonical seed sets: first 5 available per dataset
CANONICAL_SEEDS = {
    "nsl_kdd": [42, 43, 44],        # only 3 checkpoints exist
    "cicids2017": [42, 43, 44, 45, 46],
    "unsw_nb15": [42, 43, 44, 46, 47],
}


def load_exhaustive(ds, method, seed):
    """Load eval_deepfool_k1.py pgd40 result."""
    path = f"results/foolbox/exh_k1_pgd40_{ds}_{method}_seed{seed}.json"
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return {
        "n_total": d["n_total"],
        "clean_correct": d["clean_correct"],
        "attacked": d["attacked"],
        "k0_survivors": d["k0_survivors"],
        "k1_survivors": d["k1_survivors"],
        "checkpoint_sha256": d.get("checkpoint_sha256", "unknown"),
    }


def load_unified(ds, method, seed):
    """Load eval_unified.py result."""
    path = f"results/unified/eval_{method}_{ds}_seed{seed}.json"
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return {
        "clean_acc_pct": d["clean_acc"],
        "k0_acc_pct": d["k0_acc"],
        "k1_acc_pct": d["k1_acc"],
    }


def main():
    all_rows = []
    missing = []

    for ds in DATASETS:
        n_total = KNOWN_TOTALS[ds]
        seeds = CANONICAL_SEEDS[ds]
        for method in METHODS:
            for seed in seeds:
                ex = load_exhaustive(ds, method, seed)
                un = load_unified(ds, method, seed)
                if ex is None:
                    missing.append(f"exhaustive {ds}/{method}/seed{seed}")
                    continue
                if un is None:
                    missing.append(f"unified {ds}/{method}/seed{seed}")
                    continue

                cc = ex["clean_correct"]
                att = ex["attacked"]
                k0_att = ex["k0_survivors"]
                k1_att = ex["k1_survivors"]

                row = {
                    "dataset": ds,
                    "dataset_label": DS_LABELS[ds],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": seed,
                    "n_total": n_total,
                    "clean_correct": cc,
                    "clean_acc_pct": round(100 * cc / n_total, 2),
                    "checkpoint_sha256": ex["checkpoint_sha256"],
                    # Exhaustive (eval_deepfool_k1.py): α_cat=0.0, exhaustive enumeration
                    "exh_k0_of_attacked": k0_att,
                    "exh_k0_pct_of_attacked": round(100 * k0_att / max(att, 1), 2),
                    "exh_k0_pct_of_total": round(100 * k0_att / n_total, 2),
                    "exh_k1_of_attacked": k1_att,
                    "exh_k1_pct_of_attacked": round(100 * k1_att / max(att, 1), 2),
                    "exh_k1_pct_of_total": round(100 * k1_att / n_total, 2),
                    # Gradient-snapped (eval_unified.py): α_cat=1.0, full test set denom
                    "snap_k0_pct_of_total": round(un["k0_acc_pct"], 2),
                    "snap_k1_pct_of_total": round(un["k1_acc_pct"], 2),
                }
                all_rows.append(row)

    if missing:
        print(f"MISSING ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
        print()

    # === Print consolidated table ===
    print("=" * 120)
    print("CANONICAL CONSOLIDATED TABLE — Both Evaluation Conventions on Identical Checkpoints")
    print("=" * 120)
    print()
    print("Convention 1: EXH = Exhaustive enumeration (eval_deepfool_k1.py, α_cat=0.0)")
    print("  Denominators: 'of_attacked' = clean-correct; 'of_total' = full test set")
    print("Convention 2: SNAP = Gradient-snapped (eval_unified.py, α_cat=1.0)")
    print("  Denominator: full test set")
    print()

    for ds in DATASETS:
        seeds = CANONICAL_SEEDS[ds]
        rows = [r for r in all_rows if r["dataset"] == ds]
        print(f"{'='*120}")
        print(f"{DS_LABELS[ds]} (n_total={KNOWN_TOTALS[ds]}, seeds={seeds})")
        print(f"{'='*120}")
        print(f"{'Method':<12} {'Seed':>4}  {'Clean%':>7}  {'EXH K0':>14}  {'EXH K1':>14}  {'SNAP K0':>10}  {'SNAP K1':>10}  {'EXH-SNAP gap':>12}")
        print(f"{'':12} {'':>4}  {'(full)':>7}  {'(of_att)':>14}  {'(of_att)':>14}  {'(full)':>10}  {'(full)':>10}  {'(K1,full)':>12}")
        print("-" * 120)

        for method in METHODS:
            m_rows = [r for r in rows if r["method"] == method]
            for r in m_rows:
                exh_k1_full = r["exh_k1_pct_of_total"]
                snap_k1_full = r["snap_k1_pct_of_total"]
                gap = exh_k1_full - snap_k1_full
                print(f"{METHOD_LABELS[method]:<12} {r['seed']:>4}  "
                      f"{r['clean_acc_pct']:>7.2f}  "
                      f"{r['exh_k0_of_attacked']:>6}/{r['clean_correct']:<6}={r['exh_k0_pct_of_attacked']:>5.2f}%  "
                      f"{r['exh_k1_of_attacked']:>6}/{r['clean_correct']:<6}={r['exh_k1_pct_of_attacked']:>5.2f}%  "
                      f"{r['snap_k0_pct_of_total']:>9.2f}%  "
                      f"{r['snap_k1_pct_of_total']:>9.2f}%  "
                      f"{gap:>+10.2f}pp")
            if m_rows:
                # Print mean row
                n = len(m_rows)
                mean_clean = sum(r["clean_acc_pct"] for r in m_rows) / n
                mean_exh_k1_att = sum(r["exh_k1_pct_of_attacked"] for r in m_rows) / n
                mean_exh_k1_tot = sum(r["exh_k1_pct_of_total"] for r in m_rows) / n
                mean_snap_k0 = sum(r["snap_k0_pct_of_total"] for r in m_rows) / n
                mean_snap_k1 = sum(r["snap_k1_pct_of_total"] for r in m_rows) / n
                mean_gap = mean_exh_k1_tot - mean_snap_k1
                print(f"{'  MEAN':<12} {'':>4}  "
                      f"{mean_clean:>7.2f}  "
                      f"{'':>6}/{'':<6}={mean_exh_k1_att:>5.2f}%  "
                      f"{'':>6}/{'':<6}={mean_exh_k1_att:>5.2f}%  "
                      f"{mean_snap_k0:>9.2f}%  "
                      f"{mean_snap_k1:>9.2f}%  "
                      f"{mean_gap:>+10.2f}pp")
                print()

    # === Summary table for paper ===
    print()
    print("=" * 120)
    print("SUMMARY TABLE (mean ± std across seeds)")
    print("=" * 120)
    print()
    header = f"{'Dataset':<14} {'Method':<12} {'n':>2}  {'Clean%':>7}  {'EXH K0%':>10}  {'EXH K1%':>10}  {'SNAP K0%':>10}  {'SNAP K1%':>10}  {'Gap K1':>10}"
    print(header)
    print("-" * 120)

    for ds in DATASETS:
        seeds = CANONICAL_SEEDS[ds]
        for method in METHODS:
            rows = [r for r in all_rows if r["dataset"] == ds and r["method"] == method]
            if not rows:
                continue
            n = len(rows)
            import statistics
            clean = [r["clean_acc_pct"] for r in rows]
            exh_k0 = [r["exh_k0_pct_of_attacked"] for r in rows]
            exh_k1 = [r["exh_k1_pct_of_attacked"] for r in rows]
            snap_k0 = [r["snap_k0_pct_of_total"] for r in rows]
            snap_k1 = [r["snap_k1_pct_of_total"] for r in rows]

            def fmt(vals):
                if len(vals) == 1:
                    return f"{vals[0]:.2f}"
                m = statistics.mean(vals)
                s = statistics.stdev(vals)
                return f"{m:.2f}±{s:.2f}"

            gap_vals = [r["exh_k1_pct_of_total"] - r["snap_k1_pct_of_total"] for r in rows]

            print(f"{DS_LABELS[ds]:<14} {METHOD_LABELS[method]:<12} {n:>2}  "
                  f"{fmt(clean):>7}  "
                  f"{fmt(exh_k0):>10}  "
                  f"{fmt(exh_k1):>10}  "
                  f"{fmt(snap_k0):>10}  "
                  f"{fmt(snap_k1):>10}  "
                  f"{fmt(gap_vals):>10}")

    # === Save consolidated JSON ===
    out = {
        "description": "Canonical 5-seed consolidated evaluation: exhaustive (EXH) and gradient-snapped (SNAP) on identical checkpoints",
        "conventions": {
            "EXH": "eval_deepfool_k1.py: exhaustive-state AND, α_cat=0.0, PGD-40. 'of_attacked' denom = clean-correct; 'of_total' = full test set.",
            "SNAP": "eval_unified.py: gradient-snapped K=1, α_cat=1.0, PGD-40. Denominator = full test set.",
        },
        "canonical_seeds": CANONICAL_SEEDS,
        "rows": all_rows,
    }
    os.makedirs("results/consolidated", exist_ok=True)
    with open("results/consolidated/canonical_both_conventions.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to results/consolidated/canonical_both_conventions.json")


if __name__ == "__main__":
    main()
