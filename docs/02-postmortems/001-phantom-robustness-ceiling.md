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

Rigorous auditing of the evaluation scripts uncovered three severe implementation artifacts:

1. **Gradient Masking via Hard-Snapping:** The DACM function used an `argmin` operation during the forward pass. This artificially zeroed out categorical gradients during backpropagation, blinding the continuous optimizer and effectively blocking any categorical attack.
2. **Epsilon Leakage via Fractional Categories:** An attempt to bypass hard-snapping passed fractional one-hot values (e.g., `[0.85, 0.15]`) directly into the network. This violated the tabular threat model entirely. A leaked report claimed this yielded 41% robustness, but an independent audit proved the true fractional accuracy was ~8%, revealing the 41% figure to be fabricated or flawed.
3. **Greedy Optimization Oscillation:** When attacking $K>0$ categorical groups, a greedy projection heuristic forced the categorical selection to flip away from the original state at *every* step of the 40-step PGD attack. This caused violent oscillation between classes (e.g., thrashing between network flags), destabilizing the optimizer and inflating robust accuracy.

## 🧠 What we learned

**Heuristics in mixed-norm adversarial evaluation are dangerous.**

We learned that:
- **Exhaustive Evaluation is Mandatory:** Because the valid discrete state space is extremely small in tabular data, greedy heuristics are unnecessary and harmful. We must use an exhaustive discrete search that iterates over all valid categorical combinations, holds the discrete state fixed, and runs $L_\infty$ PGD exclusively on the continuous features.
- **Continuous Features Add Noise:** An ablation study on the old baseline NSL-KDD model showed that zeroing all continuous features actually *increased* accuracy (from 80.57% to 82.61%). The decision boundaries are almost entirely load-bearing on 1-2 highly predictive categorical fields.

_See the corresponding ADR: [ADR-001: Canonical Exhaustive Mixed-Norm Evaluation](../01-documentation/adrs/001-canonical-exhaustive-evaluation.md)_
