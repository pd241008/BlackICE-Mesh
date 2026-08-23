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

1. ✅ **Extended CICIDS2017 seeds 46–54** — COMPLETE Aug 23 08:33 (27/27 checkpoints + evals). Integrity audit PASSED: hash chain-of-custody 27/27, log inspection clean → [Audit Report 008](../docs/03-audit-reports/008-multiseed-4654-integrity-audit.md). Raw JSONs: `results/unified/eval_{method}_cicids2017_seed{42..54}.json`; hashes: `results/unified/checkpoint_hashes_seeds46-54.txt`. Key finding: **clean accuracy is bimodal (~89.8% vs ~80.5%) — driven by a single test class (~10% of test set) forming a generalization cliff; hardened fails it in every seed, curriculum/rsc mostly learn it.**
2. ✅ **Statistical gate at n=13** — one-way ANOVA + Bonferroni Welch (`results/unified/anova_cicids2017_n13.txt`, raw k1 + n=9 sensitivity in `anova_sensitivity_and_raw_k1.txt`): K=1 robust accuracy shows **NO method separation** (F(2,36)=0.26, p=0.77; unchanged excluding pre-recovery seeds, p=0.16); K=0 none either. Only surviving effect: Curriculum/RSC > Hardened on clean (+7–9pp, p≤0.001). **Paper headline must change: the CICIDS2017 method difference is a clean-accuracy effect, not a robustness effect** (earlier small-sample signal — seeds 42–44 plus the unnumbered base run, not a 4th seed — was seed noise). Denominator fully reconciled & provenance-audited: 39 seeded checkpoints (42–54 × 3), all hash-verified against save-time logs; seed-45 subset suspicion refuted (`Subset: None` in its training log); recovered scripts differ from originals by imports only → [Audit Report 008](../docs/03-audit-reports/008-multiseed-4654-integrity-audit.md) §Addendum.
3. 📝 **Lock paper draft** — update CICIDS2017 section per item 2; report per-cluster clean accuracy and the single-class cliff (audit report §4) instead of pooled means. Discussion must include: (a) RSC's seeds 42–44 K=1 values (70.05/69.97/46.29) sit toward the top of its own distribution — an explicit illustration of why n=3 favored it in hindsight; (b) CICIDS clean-accuracy bimodality means pooled clean means conflate "always fails class X" with "usually learns it".
4. 🏃 **UNSW-NB15 seed extension RUNNING** (`scripts/run_unsw_seeds_46_51.sh`, launched Aug 23 09:15, seeds 46–51 × 3 = n=3→9; supervised by parameterized `watchdog.sh` → `logs/unsw_ext/watchdog.log`). Triggered by effect-size gap analysis (`results/unified/gap_analysis_n3.txt`, rule: CI spans zero AND gap <5pp): UNSW K=1 Hardened-vs-Curriculum (+4.24pp, CI ±18) and Curriculum-vs-RSC (−2.94pp, CI ±19) are uncallable at n=3; UNSW clean is a precise null (all gaps ≤0.11pp, CI ±0.2pp) and needs no runs. **NSL-KDD deliberately NOT extended**: K=1 headline resolved (Curriculum≫RSC +19.7pp p=0.011; other gaps >5pp) — scoped as explicit limitation. Caveat to state honestly: at Curriculum's seed variance (~9pp), even n=9 may not render the ~3pp pair significant; goal is replacing ±19pp CIs with decision-grade bounds.
5. 🔬 **Attack generalization (baseline report §Future Work)** — `app/ml/attacks/cw.py` (Carlini–Wagner adapted to mixed-norm threat model) + `scripts/eval_cw.py` + `scripts/eval_foolbox.py`. **K=0 anchor gate PASSED Aug 23**: CW reproduces the canonical PGD K=0 number *exactly* (77.2435% both) on hash-pinned `model_adv_rsc_nsl_kdd_seed42.pth` (SHA fff152a4…, clean 85.79% both). Gate initially FAILED (CW 80.11% vs anchor 77.24%) — root cause: success checked only at end of each binary-search round, discarding transient flips; fixed with mid-round capture every 50 steps (like-for-like batch check: PGD 79.30% vs CW 79.10%). Earlier chat-claimed smoke-test numbers (86.1%/85.9%) were VOID — no persisted artifacts, checkpoint unidentifiable; superseded by this gated run. Design decisions locked: (a) **stopping rule** = fixed per-candidate budget (300 steps × 3 binary rounds, transient capture every 50), no convergence early-stop — equal optimizer budget across categorical candidates so worst-case selection isn't biased toward slow-converging states; (b) **DeepFool scoring** = post-hoc cross-entropy on its final example under the same selection metric as PGD/CW (DeepFool's internal min-distance objective never used for selection); (c) **first config** = RSC/CICIDS2017 seed53 (n=13-audited) once K≥1 exhaustive runs land; (d) **L∞ policy** = δ=ε·tanh(p) bounded by construction + per-batch assertion (observed max |δ|∞=0.052; DeepFool will use Foolbox's Linf variant). Attack-config provenance + per-batch raw records (incl. magnitudes) now persisted in `results/cw/*.json`. NOTE: NVML dropped again Aug 23 ~09:20 (new CUDA contexts fail; running trainers unaffected) — CW evals ran CPU-side, feasible since full NSL-KDD K=0 takes ~60s. Post-recovery experiment staged: while a live training context runs, fire a second trivial CUDA init and watch nvidia-smi — reproduces → serialized-context-init lock-file rule; doesn't → n=2 written off as coincidence.

**K=1 exhaustive C&W (rsc_cicids2017_seed53) — COMPLETE Aug 23, final in `results/cw/K1_FINAL_cicids2017_rsc_seed53.json`:** budget-convergence verified at K=1 same as K=0 (survivors identical across 300×3 → 600×6 → 600×6+c×100). **Budget-utilization finding (paper-worthy, own sentence):** successful CW perturbations use ~0.05–0.097 of the 0.15 L∞ ball at K=0 but saturate it (~0.149–0.150) at K=1 — flip states unlock samples continuous-only optimization cannot reach because the combined attack genuinely requires the full budget. **Headline: exhaustive CW is a strictly harder attacker than PGD-40 at K=1 here** — CW robust 3.12% of clean-correct (17,456/560,102) vs PGD ~55.2% (~49.6% canonical full-set denominator, matching the 49.58% anchor within ±0.06pp across 4 independent runs); CW survivors nest inside PGD survivors up to boundary noise (violations ≤51/17,456 = 0.29%, below PGD run-to-run churn; jaccard low only because set sizes differ ~17×). Contrast with NSL-KDD K=0 where the two attacks matched exactly. **Audit trail (both my errors, caught by anchor-gate discipline):** (1) first mirror used training-time steps=10 not eval-time 40 → inflated PGD survival to 84%; (2) residual "gap" after fixing was a denominator convention difference (`eval_unified.py` divides by FULL test set, mirrors by attacked count) — no batching/FP protocol sensitivity exists; ULP logit context-dependence measured and bounded <±0.15pp (methods footnote at most). Methods text note: K=1 probe ran on similar-not-identical batch to main run (460 vs 458 clean-correct); PGD side carries MC jitter ±~0.07pp at this n; CW side deterministic given data.
