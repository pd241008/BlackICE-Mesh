"""Aggregate unified eval results into a running tally table.

Scans results/unified/eval_{method}_{dataset}[_seed{N}].json and emits:
  - results/unified/tally.csv  (one row per model run)
  - stdout summary with per-method mean and range, grouped by dataset

Usage: PYTHONPATH=. python scripts/tally_unified.py
"""

import csv
import json
import os
import re
from collections import defaultdict

RESULTS_DIR = "results/unified"
PATTERN = re.compile(
    r"eval_(hardened|curriculum|rsc)_(nsl_kdd|cicids2017|unsw_nb15)"
    r"(?:_seed(\d+))?\.json$"
)


def main():
    rows = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        m = PATTERN.match(fname)
        if not m:
            continue
        method, dataset, seed = m.group(1), m.group(2), m.group(3)
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            data = json.load(f)
        rows.append({
            "dataset": dataset,
            "method": method,
            "seed": seed if seed else "base",
            "clean_acc": round(data["clean_acc"], 2),
            "k0_acc": round(data["k0_acc"], 2),
            "k1_acc": round(data["k1_acc"], 2),
        })

    out_csv = os.path.join(RESULTS_DIR, "tally.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "method", "seed", "clean_acc", "k0_acc", "k1_acc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}\n")

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["method"])].append(r)

    def fmt(vals):
        return f"{min(vals):.2f}-{max(vals):.2f}% (mean {sum(vals)/len(vals):.2f}%, n={len(vals)})"

    for dataset in sorted({r["dataset"] for r in rows}):
        print(f"== {dataset} ==")
        for method in ("hardened", "curriculum", "rsc"):
            runs = grouped.get((dataset, method))
            if not runs:
                continue
            seeds = ", ".join(r["seed"] for r in sorted(runs, key=str))
            print(f"  {method:<11} clean: {fmt([r['clean_acc'] for r in runs])}")
            print(f"  {'':<11} K=0:   {fmt([r['k0_acc'] for r in runs])}")
            print(f"  {'':<11} K=1:   {fmt([r['k1_acc'] for r in runs])}")
            print(f"  {'':<11} runs:  {seeds}")
        print()


if __name__ == "__main__":
    main()
