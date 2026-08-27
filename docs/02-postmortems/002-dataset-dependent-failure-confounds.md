# Postmortem: Engineering Confounds in Adversarial Training

> **Date:** August, 2026
> **Project:** Tabular Mixed-Norm Adversarial Defenses

---

## 💥 What we expected

After fixing the evaluation artifacts, we evaluated three defense mechanisms across three datasets (NSL-KDD, CICIDS2017, UNSW-NB15):
1. **Standard Hardened Training**
2. **Curriculum Training (OHCP)**
3. **Randomized Subset Constraints (RSC)**

The initial results showed a catastrophic dataset-dependent collapse. RSC completely failed (0.00% robustness) on multi-group datasets (NSL-KDD, UNSW-NB15), while Curriculum Training succeeded on UNSW-NB15 but failed on CICIDS2017. We expected that this was due to a "feature-selection loophole" (whack-a-mole strategy) inherent to the methods' mathematical design.

## 📉 What actually happened

An intensive confound audit revealed that the collapse was NOT a property of the data or the method logic, but an artifact of massive engineering confounds in the legacy training scripts:

1. **The $\alpha_{cat}$ Bug:** The Standard and RSC models were trained using $\alpha=0.01$ for categorical gradients. Because $0.01 \times 10 \text{ steps} = 0.1$, the maximum continuous delta was mathematically incapable of flipping a one-hot category using `argmax` (which requires pushing past a 0.5 threshold). Consequently, **Standard and RSC models were never trained against categorical perturbations.** Only Curriculum models correctly ramped $\alpha_{cat}$ up to 1.0.
2. **Epsilon Discrepancy:** Curriculum models were evaluated against an $L_\infty$ bound of $\epsilon=0.15$, while Standard/RSC were trained at $\epsilon=0.1$.
3. **Legacy Provenance:** The NSL-KDD Hardened model was an outdated FGSM checkpoint, breaking comparability with the PGD models.
4. **Dataset Size Imbalance:** The training sets varied massively (CICIDS2017: 2.5M, UNSW-NB15: 2M, NSL-KDD: 125k).

## 🧠 What we learned

**Engineering confounds can perfectly mimic fundamental theoretical failures.**

We learned that:
- **Sandbox Retraining is Required:** To properly compare methods, we must execute them in a strictly unified training pipeline (`train_unified.py`).
- **The Methods Do Generalize:** When properly forced to defend categorical permutations with $\alpha_{cat}=1.0$, RSC achieves an incredible 96.17% robust accuracy on UNSW-NB15 (up from 0.00%). Curriculum achieves 24.47% on NSL-KDD (up from 0.00%).
- **Topology Dictates the Defense:** The optimal defense depends on the dimensionality of the categorical threat space. For simple topologies ($|G|=1$), Standard adversarial training is sufficient. For high-dimensionality topologies ($|G|>1$), RSC acts as a powerful regularizer to prevent overfitting.

**Caveats on reported figures:**
- **Checkpoint provenance:** The legacy `model_adv.pth` was inherited from the original AdvGuard monolith and underwent an additional 50 epochs of fine-tuning beyond the published weights (baseline report §1). Unified pipeline checkpoints are retrained from scratch and are not comparable.
- **PGD nondeterminism:** `unified_pgd.py` uses random epsilon-ball initialization (line 35). Robust-accuracy figures are reproducible in distribution (±~1pp at n=500) rather than bit-for-bit deterministic.
- **Faithful full-scale result:** Under the verified faithful attack with corrected defaults, the legacy hardened model achieves **40.36%** at ε=0.15 on the full test set (n=22,543). The 77.28% figure (K=0 continuous-only) and 29.10% figure (unconstrained, categorical disabled) are not comparable K=1 results.

_See the corresponding ADR: [ADR-002: Unified Adversarial Training Framework](../01-documentation/adrs/002-unified-adversarial-training.md)_
