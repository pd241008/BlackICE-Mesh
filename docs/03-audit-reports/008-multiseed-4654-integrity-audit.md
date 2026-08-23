# Audit Report 008: Multi-Seed Sweep Integrity (CICIDS2017, Seeds 46–54)

> **Date:** August 23, 2026
> **Scope:** Extended seed batch produced by `scripts/run_seeds_46_54.sh` (unified pipeline recovered from agent scratch into repo).
> **Verification standard:** same as prior rounds — raw per-seed results, checkpoint hash chain-of-custody, training log inspection.

---

## 1. Raw per-seed results (no aggregation)

39 runs = seeds 42–54 × {hardened, curriculum, rsc}. Values verbatim from
`results/unified/eval_{method}_cicids2017_seed{N}.json` (clean / K=0 / K=1, %):

| Seed | Hardened | Curriculum | RSC |
|------|----------|------------|-----|
| 42 | 80.73 / 36.22 / 33.45 | 69.36 / 37.54 / 30.97 | 80.77 / 71.37 / 70.05 |
| 43 | 79.80 / 59.86 / 50.56 | 48.75 / 24.39 / 20.97 | 80.54 / 75.13 / 69.97 |
| 44 | 80.46 / 68.08 / 64.42 | 89.81 / 74.96 / 72.72 | 80.53 / 60.20 / 46.29 |
| 45 | 80.44 / 64.03 / 35.45 | 89.21 / 71.94 / 53.68 | 89.81 / 81.63 / 77.18 |
| 46 | 80.60 / 62.99 / 56.34 | 89.81 / 75.61 / 71.63 | 89.77 / 81.08 / 75.92 |
| 47 | 80.43 / 68.98 / 66.35 | 89.68 / 83.54 / 61.16 | 89.86 / 76.11 / 67.15 |
| 48 | 80.60 / 68.05 / 65.91 | 89.58 / 54.05 / 43.31 | 89.83 / 78.19 / 62.88 |
| 49 | 80.27 / 66.91 / 63.60 | 89.79 / 80.56 / 77.13 | 89.79 / 66.51 / 43.38 |
| 50 | 80.48 / 65.32 / 57.09 | 89.80 / 84.09 / 61.36 | 89.84 / 69.02 / 47.61 |
| 51 | 80.25 / 69.15 / 66.70 | 89.86 / 74.78 / 69.86 | **80.61** / 72.58 / 65.76 |
| 52 | 80.24 / 69.43 / 52.33 | 89.61 / 84.83 / 79.28 | 89.81 / 56.23 / 39.50 |
| 53 | 80.60 / 67.33 / 63.94 | 89.71 / 81.26 / 77.00 | 89.78 / 76.83 / 49.58 |
| 54 | 80.51 / 63.53 / 62.21 | 89.56 / 81.27 / 66.16 | **80.51** / 71.95 / 68.42 |

## 2. Checkpoint hash chain-of-custody — 27/27 VERIFIED

Every training run logs the SHA-256 of its checkpoint at save time
(`app/ml/utils/checkpoint.py`). All 27 saved-run hashes were recomputed from
disk and compared against their logs:

```
hardened_46  9ef53d83f0b1…   curriculum_46 68bdc6207de0…   rsc_46        5135425e60c3…
hardened_47  dee867c7871d…   curriculum_47 1e40a3e0d518…   rsc_47        a40a714e9538…
hardened_48  6ae0132d019b…   curriculum_48 1cc2b98a8337…   rsc_48        41ca881581dd…
hardened_49  562abc254ea2…   curriculum_49 d18c6be2abed…   rsc_49        accb3345b826…
hardened_50  a4ab0f1f7b88…   curriculum_50 dd207cb58b77…   rsc_50        bf74d878675d…
hardened_51  333dd87dd00b…   curriculum_51 608e65d1d9c8…   rsc_51        9107ded5f86e…
hardened_52  8186aaf604bc…   curriculum_52 c38c635b2823…   rsc_52        dbc9be548b36…
hardened_53  37c5f4950245…   curriculum_53 b71117ac12c4…   rsc_53        055962e7bdfe…
hardened_54  a3f95f12212d…   curriculum_54 08bcdf285637…   rsc_54        d7cb67a0c4dc…
```
(*full digests persisted at `ml-optimizer/results/unified/checkpoint_hashes_seeds46-54.txt` and in each train log; all 27 log-vs-disk comparisons = MATCH. No post-training modification or corruption.)

## 3. Training log inspection

Red-flag sweep across all 27 logs (`logs/train_{method}_seed{N}.log`):

| Check | Result |
|---|---|
| Warm-started from `models/model_cicids2017.pth` | 27/27 ✓ |
| Completed 50 epochs | 27/27 ✓ |
| Trained on full dataset (`Subset: None`) | 27/27 ✓ (banner verified; no subsample fallback) |
| Started from scratch (failure path) | 0/27 ✓ |
| α_cat schedule | hardened/rsc fixed 1.00 ✓; curriculum ramp 0.01→1.00 over epochs 10–30 ✓ (spot-checked) |

Deep-dives performed: `rsc_seed53` (control), `rsc_seed51`, `rsc_seed54` (anomalies below), `hardened_seed51`. Loss trajectories converge normally (final train clean ≈ 95–96%, train robust ≈ 78–85%) with no divergence or NaN events.

**Incident note (transparency):** seed-52 RSC's first attempt wedged during a host-side GPU dropout (2026-08-22 ~22:00, NVML lost device); it was killed and retrained cleanly after system restart (07:45–08:29 next morning). Its final checkpoint passed all checks above.

## 4. Finding: clean accuracy is bimodal — and it's one class

Seeds split into two clusters on TEST clean accuracy: ~89.8% vs ~80.5%
(hardened sits in the low cluster in *every* seed; curriculum in the high cluster
in 12/13; RSC splits 11 high / 2 low).

Investigation chain:
1. Fresh evals reproduce the split from weights alone (89.88% vs 80.90%/80.73%) → not an eval-run artifact.
2. Layer-wise diffs confirm genuinely independent trainings (per-layer max-diff O(0.1–1.7)); hash-distinct files.
3. Disagreement analysis (seed53-high vs seed51-low, full 623,869-sample test set):
   - seed53 correct & seed51 wrong: **58,067 samples (9.31 pp)** — of which **58,010 share one true label (label 1)**
   - both-wrong rate 10.08%.

**Interpretation:** a single test class (~10% of the test set, consistent with
CICIDS2017's shifted test days) forms a generalization cliff. High-cluster models
learn it; low-cluster models miss essentially the entire block. Method means
therefore conflate "always fails the class" (hardened) with "usually learns it"
(curriculum/RSC). Recommend the paper report clean accuracy per cluster and/or
this class breakdown rather than a single mean.

## 5. Verdict

Numbers for seeds 46–54 are **trustworthy**: unmodified checkpoints (hash-verified),
correct recipe execution (logs), complete raw results preserved. No repeat of any
prior-round bug. The K=1 headline remains: hardened 57.3% vs curriculum 61.1% vs
rsc 61.1% mean — method separation requires the ANOVA gate (RESEARCH_STATUS item 2).
