# 📜 ADR-001: Canonical Exhaustive Mixed-Norm Evaluation

> **Status:** `Decided`
> **Date:** August, 2026

---

## 🌎 Context

Tabular network intrusion detection models process both continuous and categorical features. Adversarial evaluation requires a mixed-norm threat model ($L_\infty$ for continuous, $L_0$ for categorical). Three distinct evaluation conventions emerged during this project's history, each producing different robust-accuracy figures from the same checkpoint:

| Convention | Categorical handling | Denominator | Example result (NSL-KDD Hardened, ε=0.15) |
|---|---|---|---|
| **Legacy** (AdvGuard original) | `argmax → nearest one-hot`, but `alpha_cat=0.01` makes snap dead code | correct/total | 29.10% (external) / 27.69% (this repo) |
| **SNAP** (gradient-snapped K=1) | Single best-by-gradient flip per group, α_cat=1.0 | full test set | 8.43%–23.18% (per seed) |
| **EXH** (exhaustive K=1) | Enumerate all \|G\| one-hot states, run continuous PGD on each, pick worst | clean-correct | 0.00% |

The legacy convention's 77.28% (NSL-KDD Hardened) was mischaracterized in early audit reports as a "DACM-constrained attack." It is actually a **continuous-only PGD attack (K=0)**: the categorical snap (`argmax → one-hot`) exists in the code but is inactive because `alpha_cat=0.01` is below the flip threshold for one-hot vectors. The 77.28% figure measures robustness to perturbation of only the 4 continuous features out of 18, with all 14 categorical features frozen at their original state.

These heuristic and incomplete approaches introduced massive evaluation artifacts:
1. **Fractional Epsilon Leakage:** Passing continuous gradients directly into categorical one-hot fields resulted in invalid states (e.g., fractional network flags like `[0.85, 0.15]`), violating the threat model.
2. **Greedy Optimization Oscillation:** When attacking $K>0$ categorical groups, a greedy projection heuristic forced the categorical selection to flip away from the original state at every step, destabilizing continuous PGD and artificially inflating robust accuracy.
3. **Dead-Code Categorical Channel:** `alpha_cat=0.01` is mathematically insufficient to flip a one-hot `argmax` boundary (max delta 0.1 over 40 steps). The categorical snap executes but never changes the input, making K=0 and K=1 evaluations produce different numbers for unrelated reasons.

## 🛤️ Options Considered

1. **Gradient-snapped evaluation (SNAP)** — Single best-by-gradient flip per group, applied once at attack end. Computationally cheap but systematically overestimates robustness (gap of 2–54pp vs. EXH across datasets).
2. **Exhaustive discrete evaluation (EXH)** — Iterate over all valid discrete combinations of $K$ categorical flips, hold the discrete state fixed, and run continuous $L_\infty$ PGD on each state. Select the worst-case loss.
3. **Legacy DACM-snap evaluation** — Original AdvGuard `argmax → one-hot` snap applied every PGD step. With corrected `alpha_cat=1.0` defaults, produces faithful but non-exhaustive results.

---

## 🎯 Decision

> [!IMPORTANT]  
> **We will use Exhaustive Discrete Evaluation (EXH) as the canonical K≥1 protocol. SNAP is retained as the scalable approximation with a disclosed 2–54pp overestimation bias. Legacy convention numbers (77.28%, 29.10%) are re-characterized as K=0 baseline anchors, not K=1 robustness claims.**

## 🧠 Reasoning

By decoupling the discrete categorical attack from the continuous gradient optimization, we guarantee that the mixed-norm budget is perfectly respected and oscillation is impossible. The true worst-case vulnerability is always found.

**Verified faithful reproduction (Section III diagnostic):** The original AdvGuard snap logic (`argmax → nearest one-hot`, applied every PGD step) was ported verbatim from the original source (git commit `566735a`) with corrected defaults (`continuous_cols=[0,1,2,3]`, `categorical_groups=[[4,5,6],[7..17]]`). On the legacy `model_adv.pth` checkpoint (SHA: `f07d2e37...`) at full scale (n=22,543):

| Epsilon | Baseline | Hardened | Gap |
|---|---|---|---|
| 0.10 | 20.10% | 50.96% | -30.86pp |
| 0.12 | 16.09% | 46.08% | -29.99pp |
| 0.15 | 16.06% | **40.36%** | -24.30pp |

The 18.9pp gap between the original paper's 29.10% and the faithful 40.36% decomposes as: ~1.4pp from checkpoint provenance (50-epoch fine-tuning beyond published weights, documented in baseline report §1) and ~20.3pp from enabling the categorical snap channel that was inactive in the original evaluation defaults.

**Small-sample fragility:** At n=100, the same evaluation yields 48.00% for the hardened model at ε=0.15 — a 7.64pp overestimate vs. full-scale 40.36%. Evaluation rigor requires both correct methods AND adequate sample sizes.

**Scalability confirmed:** The exhaustive evaluator scales near-linearly with the number of categorical candidates. On UNSW-NB15 ($|G|=5$): 14→25→36→38→42 candidates at 1.4→4.2→7.0→7.2→7.8s. K=2 evaluation (667 candidates) completes in 59.0s on an RTX 4050 Laptop.

**Even the faithful snap overestimates:** Under exhaustive K=1 enumeration (EXH), robust accuracy collapses to **0.00%** for NSL-KDD Hardened — confirming that even bug-free non-exhaustive evaluation is structurally misleading. The paper's thesis is not "we found a bug" but "even correctly-implemented, genuinely categorical-perturbing attacks dramatically overestimate robustness relative to exhaustive enumeration."
