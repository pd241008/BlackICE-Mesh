# ML Optimizer — Research Status

> **Last updated:** August 25, 2026
> **Phase:** CROWN backward-pass bug fixed; all 27 checkpoints re-evaluated with sound relaxation; mixed-norm K=1 certification shows vacuous bounds for TabularMLP architecture

---

## What This Module Does

The `ml-optimizer` implements the adversarial robustness research pipeline for the BlackIce paper. It trains and evaluates three adversarial training methods (Hardened, Curriculum, RSC) across three network intrusion datasets (NSL-KDD, CICIDS2017, UNSW-NB15) under a unified mixed-norm threat model (L∞ on continuous features, L0 on categorical features).

---

## Current Results

### K=1 Exhaustive-State Robust Accuracy (primary protocol)

| Dataset | n | Hardened | Curriculum | RSC | Method winner |
|---|---|---|---|---|---|
| **NSL-KDD** | 3 | 0.0% ± 0.0 | 0.03% ± 0.05 | 0.0% ± 0.0 | Universal collapse — no method survives |
| **CICIDS2017** | 13 | **38.84% ± 16.1** | 7.55% ± 5.3 | 11.69% ± 24.9 | Hardened ≫ others (ANOVA p=8e-05) |
| **UNSW-NB15** | 9 | 73.79% ± 9.4 | 54.29% ≪ 13.7 | 75.43% ± 10.9 | Hardened ≈ RSC ≫ Curriculum |

### CROWN Certified Bounds at ε=0.15 (corrected — Aug 25)

**Sound CROWN backward pass** (sign-dependent crossing relaxation: r≥0 → dead ReLU, r<0 → secant). 0 soundness violations across all 27 checkpoints vs brute-force grid search.

**K=0 (continuous L∞ only):**

| Dataset | Hardened | Curriculum | RSC | IBP |
|---|---|---|---|---|
| **NSL-KDD** | 52-66% | 84-91% | 77-82% | 0.0% |
| **CICIDS2017** | 0% | 0% | 0% | 0.0% |
| **UNSW-NB15** | 0% | 0% | 0% | 0.0% |

**K=1 (full mixed-norm: L∞ continuous + L0 categorical exhaustive):**

| Dataset | Hardened | Curriculum | RSC | Exh K=1 range |
|---|---|---|---|---|
| **NSL-KDD** | 0% | 0% | 0% | 0-0.05% |
| **CICIDS2017** | 0% | 0% | 0% | 0.5-77.8% |
| **UNSW-NB15** | 0% | 0% | 0% | 18-85% |

CICIDS2017 and UNSW-NB15: CROWN K=0 is vacuous (0%) because most neurons are crossing-ReLUs, and the sound relaxation treats positive-slope crossings as dead (~400× looser than brute-force minimum). NSL-KDD K=0 is meaningful because more neurons are always-on/always-off.

### Interpretation

- **CROWN limitations:** Standard CROWN backward is sound but too conservative for TabularMLP with many crossing ReLUs. The gap between CROWN bound and brute-force minimum is 200-800×. This is an inherent limitation of per-neuron linear relaxation, not a bug — confirmed by Stage A (analytic) and Stage B (linear equivalence) passing perfectly.
- **NSL-KDD:** K=1 exhaustive collapse is universal across all three optimizers (PGD-40, CW, DeepFool); this is a trivial-flip vulnerability, not a robustness differentiator. CROWN K=0 certifies high margins on the continuous subspace, confirming the vulnerability is purely categorical.
- **CICIDS2017:** Hardened dominates under exhaustive semantics (38.8% >> others). Snap-protocol null (F=0.01, p=.99) hides this separation. Clean-accuracy bimodality means pooled clean means conflate "always fails class X" with "usually learns it."
- **UNSW-NB15:** Hardened ≈ RSC ≫ Curriculum. Curriculum weakest on both datasets where ranking is resolved. Protocol-necessity thesis replicates 2/2.

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

All core evaluation phases are COMPLETE. Remaining tasks are paper-writing support:

1. ✅ **Extended CICIDS2017 seeds 46–54** — COMPLETE Aug 23
2. ✅ **Statistical gate at n=13** — COMPLETE Aug 23
3. 📝 **Lock paper draft** — update CICIDS2017 section per item 2; add CROWN certification results (item 12); report NSL-KDD as explicitly limited (trivial collapse confirmed by 3 optimizers)
4. ✅ **UNSW-NB15 seed extension** — COMPLETE (seeds 46-51, n=9)
5. ✅ **Attack generalization (CW + DeepFool)** — COMPLETE Aug 23-25
6. ✅ **DeepFool/Foolbox run + K=1 semantics correction** — COMPLETE Aug 23
7. ✅ **Full 13×3 exhaustive-state multiseed sweep (CICIDS2017)** — COMPLETE Aug 23
8. ✅ **Certified bounds (IBP)** — COMPLETE Aug 23
9. ✅ **Verification round** — COMPLETE Aug 23
10. ✅ **UNSW-NB15 exhaustive-state sweep** — COMPLETE Aug 25
11. ✅ **NSL-KDD exhaustive-state sweep** — COMPLETE Aug 25
12. ✅ **CROWN certification bounds (backward-mode)** — COMPLETE Aug 25 (27/27 checkpoints)
    - Bug found/fixed: crossing ReLU used secant (upper bound) instead of sign-dependent relaxation
    - CROWN is sound (0 violations) but vacuously loose on CICIDS2017/UNSW-NB15 (many crossing neurons)
    - NSL-KDD K=0 = 52-91% meaningful; K=1 = 0% everywhere (categorical enumeration kills all)
13. ✅ **NSL-KDD DeepFool cross-check** — COMPLETE Aug 25 (confirms universal K=1 collapse)
