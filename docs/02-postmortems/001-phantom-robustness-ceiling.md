# Postmortem: The Phantom Robustness Ceiling

> **Date:** August, 2026
> **Project:** Tabular Mixed-Norm Adversarial Defenses

---

## 💥 What we expected

When first evaluating adversarial robustness on tabular datasets (NSL-KDD, CICIDS2017), we expected mixed-norm attacks ($L_\infty$ on continuous, $L_0$ on categorical) to degrade model performance. However, initial evaluations reported a resilient "phantom ceiling" of ~77% robust accuracy. The categorical perturbations seemed to have zero effect on the model.

We expected:
1. The DACM (Discrete Adversarial Categorical Masking) hard-snapping heuristic to provide valid adversarial examples.
2. The model to be genuinely robust to these attacks.

## 📉 What actually happened

Rigorous auditing of the evaluation scripts uncovered the root cause: **the categorical attack channel was never activated.**

The 77.28% figure (17,421/22,543, NSL-KDD Hardened at ε=0.15) was mischaracterized in early audit reports as a "DACM-constrained attack." It is actually a **continuous-only PGD attack (K=0)**: the categorical snap (`argmax → one-hot`) exists in the code but is inactive because `alpha_cat=0.01` is below the flip threshold for one-hot vectors. The attack perturbs only 4 of 18 features (22% of the input), leaving all 14 categorical features frozen at their original state.

This was confirmed by reproducing 77.28% bit-for-bit with a K=0 continuous-only attack (no categorical perturbation, no random start, full test set). The number is arithmetically correct but measures the wrong thing.

Additional artifacts found during the investigation:
1. **Epsilon Leakage via Fractional Categories:** An attempt to bypass hard-snapping passed fractional one-hot values (e.g., `[0.85, 0.15]`) directly into the network. This violated the tabular threat model entirely.
2. **Greedy Optimization Oscillation:** When attacking $K>0$ categorical groups, a greedy projection heuristic forced the categorical selection to flip away from the original state at *every* step of the 40-step PGD attack, destabilizing the optimizer and inflating robust accuracy.
3. **Two Categorical-Channel Dead Paths:** An independent reimplementation attempt (`dacm_replication_test.py`) introduced two distinct failure modes: (a) `pgd_dacm_attack` computed categorical gradient steps but discarded them (snap called on `adv` instead of `adv_cat_proj` — Bug 3), and (b) `pgd_bpda_attack` never applied the categorical gradient step at all. Both produced K=0 results under different-sounding descriptions.

## 🧠 What we learned

**Heuristics and incomplete implementations in mixed-norm adversarial evaluation are dangerous. Even bug-free non-exhaustive evaluation is structurally misleading.**

The investigation produced a three-row staircase of increasing attack strength:

| Figure | What it actually is | Checkpoint |
|---|---|---|
| 77.28% | K=0, no random start, categorical dead (`alpha_cat=0.01`) | model_adv.pth (legacy) |
| 40.36% | K=0, random start, faithful AdvGuard snap (corrected defaults) | model_adv.pth (legacy) |
| 0.00% | K=1 exhaustive, full test set | model_adv.pth (legacy) |

All three use the same checkpoint, same denominator (correct/total), same epsilon (0.15). The progression is entirely from attack methodology getting strictly stronger.

Key findings:
- **Exhaustive Evaluation is Mandatory:** Because the valid discrete state space is extremely small in tabular data, greedy heuristics are unnecessary and harmful. We must use an exhaustive discrete search that iterates over all valid categorical combinations, holds the discrete state fixed, and runs $L_\infty$ PGD exclusively on the continuous features.
- **Small-Sample Fragility:** Even after fixing the methodology, sample size matters: n=100 yields 48.00% while n=22,543 yields 40.36% — a 7.64pp overestimate at small scale.
- **Continuous Features Add Noise:** An ablation study on the old baseline NSL-KDD model showed that zeroing all continuous features actually *increased* accuracy (from 80.57% to 82.61%). The decision boundaries are almost entirely load-bearing on 1-2 highly predictive categorical fields.
- **Three Evaluation Conventions:** Legacy (K=0, categorical dead), SNAP (gradient-snapped single flip, overestimates by 2–54pp), and EXH (exhaustive enumeration, canonical). See [ADR-001](../01-documentation/adrs/001-canonical-exhaustive-evaluation.md) for the full framework.

_See the corresponding ADR: [ADR-001: Canonical Exhaustive Mixed-Norm Evaluation](../01-documentation/adrs/001-canonical-exhaustive-evaluation.md)_
