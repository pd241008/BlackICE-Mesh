# 📜 ADR-001: Canonical Exhaustive Mixed-Norm Evaluation

> **Status:** `Decided`
> **Date:** August, 2026

---

## 🌎 Context

Tabular network intrusion detection models process both continuous and categorical features. Adversarial evaluation requires a mixed-norm threat model ($L_\infty$ for continuous, $L_0$ for categorical). Early attempts to evaluate mixed-norm robustness relied on gradient masking (DACM hard-snapping during the forward pass) or greedy heuristics (Top-K projection during PGD optimization). 

These heuristic approaches introduced massive evaluation artifacts:
1. **Fractional Epsilon Leakage:** Passing continuous gradients directly into categorical one-hot fields resulted in invalid states (e.g., fractional network flags like `[0.85, 0.15]`), violating the threat model.
2. **Greedy Optimization Oscillation:** When attacking $K>0$ categorical groups, a greedy projection heuristic forced the categorical selection to flip away from the original state at every step. This caused the optimization to violently oscillate between classes (e.g., thrashing between network flags), completely destabilizing continuous PGD and artificially inflating robust accuracy.

## 🛤️ Options Considered

1. **Continue using Greedy Top-K Projection** - Computationally cheap, but fundamentally flawed due to optimization oscillation.
2. **Exhaustive Discrete Evaluation** - Iterate over all valid discrete combinations of $K$ categorical flips, hold the discrete state fixed, and run continuous $L_\infty$ PGD on each state. Select the worst-case loss.

---

## 🎯 Decision

> [!IMPORTANT]  
> **We will use Exhaustive Discrete Evaluation because the categorical state space is small enough (e.g., 14 valid states for NSL-KDD at $K=1$) to make brute-force search tractable and mathematically rigorous.**

## 🧠 Reasoning

By decoupling the discrete categorical attack from the continuous gradient optimization, we guarantee that the mixed-norm budget is perfectly respected and oscillation is impossible. The true worst-case vulnerability is always found. While it requires running PGD multiple times per sample (once for each valid discrete combination), the absolute size of the tabular threat space is small enough that the compute cost is acceptable for the guarantee of correctness.
