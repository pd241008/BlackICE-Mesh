# ML Optimizer — Research Status

> **Last updated:** August 2026  
> **Phase:** Multi-Seed Statistical Validation (CICIDS2017 extended run in progress)

---

## What This Module Does

The `ml-optimizer` implements the adversarial robustness research pipeline for the BlackIce paper. It trains and evaluates three adversarial training methods (Hardened, Curriculum, RSC) across three network intrusion datasets (NSL-KDD, CICIDS2017, UNSW-NB15) under a unified mixed-norm threat model (L∞ on continuous features, L0 on categorical features).

---

## Current Results (n=3 seeds, seeds 42–44)

### K=1 Robust Accuracy — Multi-Seed Summary

| Dataset | Hardened | Curriculum | RSC |
|---|---|---|---|
| **NSL-KDD** | 14.7% ± 6.2pp | **22.5% ± 4.5pp** | 2.8% ± 2.5pp |
| **CICIDS2017** | 49.5% ± 15.5pp | 41.6% ± 27.5pp | **62.1% ± 13.7pp** |
| **UNSW-NB15** | **95.5% ± 1.2pp** | 91.2% ± 9.3pp | 94.2% ± 2.4pp |

### Interpretation

- **NSL-KDD:** Curriculum is the most stable and highest-performing method (statistically separable from RSC which fails badly).
- **CICIDS2017:** No statistically significant winner. All three methods show extreme initialization variance (±11–27pp). The n=3 ranking is not reliable — **extended seed run (seeds 45–54) in progress** to determine if any method genuinely dominates.
- **UNSW-NB15:** All three methods achieve high robustness (>91%). Hardened is marginally best; differences are within noise for n=3.

---

## Pipeline Architecture

```
data/                        ← Preprocessed Parquet datasets
models/unified/              ← Trained model checkpoints (per-seed)
results/unified/             ← Evaluation JSON results (per-seed)
app/ml/
  data/loader.py             ← StreamingParquetDataset (memory-efficient)
  data/{dataset}_config.py   ← Feature dimensions and categorical group definitions
  models/architecture.py     ← TabularMLP
  utils/checkpoint.py        ← SHA-256 verified model loading/saving
scratch/
  train_unified.py           ← Unified training script (all methods/datasets)
  eval_unified.py            ← Exhaustive K=0/K=1 evaluation
  unified_pgd.py             ← Mixed-norm PGD attack (L∞ + stochastic RSC masking)
  resume_multiseed_sequential.sh    ← Sequential multi-seed runner (OOM-safe)
  cicids2017_extended_seeds.sh      ← Extended seed runner for CICIDS2017 only
  aggregate_multiseed.py            ← Aggregate n=3 results
  aggregate_cicids2017_extended.py  ← Aggregate n=13 results + ANOVA
```

---

## Key Decisions & Incidents

| Document | Summary |
|---|---|
| [ADR-001](../docs/01-documentation/adrs/001-canonical-exhaustive-evaluation.md) | Why exhaustive discrete evaluation replaced greedy heuristics |
| [ADR-002](../docs/01-documentation/adrs/002-unified-adversarial-training.md) | Why all models were retrained from scratch with unified hyperparameters |
| [ADR-003](../docs/01-documentation/adrs/003-multi-seed-validation.md) | Multi-seed validation protocol before any ranking claim |
| [ADR-004](../docs/01-documentation/adrs/004-streaming-data-loader.md) | Memory-efficient streaming loader to prevent OOM crashes |
| [Postmortem-001](../docs/02-postmortems/001-phantom-robustness-ceiling.md) | Gradient masking artifacts inflating robust accuracy |
| [Postmortem-002](../docs/02-postmortems/002-dataset-dependent-failure-confounds.md) | Dataset confounds masking real method differences |
| [Postmortem-003](../docs/02-postmortems/003-fabricated-flops-and-pythonpath-chain.md) | Fabricated FLOPs estimate + PYTHONPATH silent failure chain |

---

## Next Steps

1. ⏳ **Extended CICIDS2017 seeds 46–54** — to be run overnight (seeds 45 Hardened+Curriculum already done; 45 RSC in progress as sample verification)
2. 📊 **Run `aggregate_cicids2017_extended.py`** — once all 13 seeds complete, report final ANOVA result
3. 📝 **Lock paper draft** — update CICIDS2017 section with n=13 statistical finding
