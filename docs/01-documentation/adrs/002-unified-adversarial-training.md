# 📜 ADR-002: Unified Adversarial Training Framework

> **Status:** `Decided`
> **Date:** August, 2026

---

## 🌎 Context

During Phase 2 of evaluating adversarial defenses (Curriculum Training vs. Randomized Subset Constraints vs. Standard Hardened Training), we observed a catastrophic collapse of robust accuracy on multi-group datasets (NSL-KDD, UNSW-NB15). Initial theories blamed a "feature-selection loophole" in the methods themselves.

However, an audit revealed massive engineering confounds across the models:
1. **The $\alpha_{cat}$ Bug:** Standard and RSC models used an $\alpha=0.01$ step size for categorical gradients, which was mathematically insufficient (max delta 0.1 over 10 steps) to flip a one-hot `argmax` boundary. These models were effectively never trained against categorical perturbations. Curriculum properly ramped $\alpha_{cat}$ to 1.0.
2. **Epsilon Mismatch:** Curriculum was evaluated/trained against an $L_\infty$ bound of $\epsilon=0.15$, while Standard and RSC used $\epsilon=0.10$.
3. **Dataset Size Imbalance:** The training sets varied massively (CICIDS2017: 2.5M, UNSW-NB15: 2M, NSL-KDD: 125k), introducing another uncontrolled variable when comparing cross-dataset topological generalization.

## 🛤️ Options Considered

1. **Patch existing scripts** - Prone to leaving hidden artifacts or drift between the multiple legacy training files (`trainer.py` vs `train_pgd_robust.py`).
2. **Unified Retraining Pipeline** - Build a strictly unified sandboxed training script (`train_unified.py`) that applies the exact same attack logic, identical hyperparameters, and normalized dataset sizes to all 9 models.

---

## 🎯 Decision

> [!IMPORTANT]  
> **We will use a Unified Retraining Pipeline to completely eliminate engineering confounds.**

## 🧠 Reasoning

To definitively prove whether failure mechanisms were dataset properties or engineering artifacts, the training environment had to be absolutely sterile. By enforcing identical PGD logic ($\alpha_{cat}=1.0$, $\epsilon=0.15$, 10 steps) and subsampling the massive datasets down to 125,000 rows to perfectly match NSL-KDD, we reduced the number of variables to one: the dataset topology itself. This decision flipped the paper's conclusion, proving that the methods *do* generalize when properly implemented.

**Caveats and limitations:**

- **PGD nondeterminism:** `unified_pgd.py` uses `torch.empty_like(...).uniform_(-epsilon, epsilon)` for random epsilon-ball initialization (line 35). This is an inherent source of per-sample stochasticity independent of model initialization. Reported robust-accuracy figures are reproducible in distribution (empirically ±~1pp at n=500, substantially smaller at full test-set scale) rather than bit-for-bit deterministic.
- **Checkpoint provenance:** The legacy `model_adv.pth` (SHA: `f07d2e37...`) was inherited from the original AdvGuard monolith and underwent an additional 50 epochs of fine-tuning beyond the published weights, shifting robust accuracy from 29.10% to 27.69% under the same evaluation (baseline report §1). Unified pipeline checkpoints (`models/unified/model_adv_*_seed*.pth`) are retrained from scratch with identical hyperparameters and are not comparable to the legacy checkpoint.
- **Faithful full-scale result:** Under the verified faithful attack (original AdvGuard snap with corrected defaults), the legacy hardened model achieves **40.36%** at ε=0.15 on the full test set (n=22,543). This is the true DACM-constrained robustness — not 77.28% (K=0 continuous-only, snap inactive) or 29.10% (unconstrained, categorical disabled).
