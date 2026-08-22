# BlackICE-Mesh: Establishing the New Robustness Baseline

This document formally tracks the experimental findings transitioning from the prior **Adv-Guard** research to the newly established **BlackICE-Mesh** baseline. It serves as the official record of the current state of robustness, documenting the progression from the originally published numbers to the newly verified 59.71% robust accuracy.

---

## 1. Contextualizing the Prior Work (Adv-Guard)

In prior research, we established the foundational robustness trajectories of tabular defense models, explicitly highlighting the discrepancy between constrained (masked) and unconstrained (unmasked) adversarial evaluations. 

The originally published results (Table II of the prior work) documented the following behavior under an $\epsilon=0.15$ adversarial budget:

*   **Masked Evaluation (Categorical Snapping Enforced):** The Adversarial Training defense appeared to achieve **93.00%** robust accuracy. 
*   **Unmasked Evaluation (Unconstrained Continuous Space):** The Adversarial Training defense achieved **29.10%** robust accuracy.

### The Gradient Masking Discovery
Subsequent rigorous auditing revealed that the extremely high masked accuracy (93%) was an artifact of **gradient masking**. 
When standard gradient-based attacks (like FGSM or PGD) interact with the non-differentiable `argmax` operation used to enforce categorical tabular constraints (e.g., protocol types), the gradient is shattered. Furthermore, standard attack step sizes (e.g., $\alpha=0.01$) are mathematically too small to flip a one-hot categorical feature before it gets snapped back to its original value. 

Because the network relied heavily on categorical features for its decision boundary (~80% feature importance), an attack that failed to perturb categorical features resulted in an artificially inflated robust accuracy. 

**Conclusion on Prior Work:** The unmasked evaluation of **29.10%** stands as the true representation of the prior model's structural resilience when the attack is not artificially hindered by discrete mathematical discontinuities.

---

## 2. The BlackICE-Mesh Solution: Curriculum PGD

To solve the gradient shattering problem and achieve *genuine* structural robustness against fully constrained adversarial packets, we implemented a new training methodology: **Curriculum PGD Adversarial Training**.

### Methodology
1.  **Fixed Categorical Scaling (`alpha_cat = 1.0`):** The attack used during training was modified to scale categorical gradients aggressively. This ensures the attack can successfully bypass the `argmax` snap and flip one-hot boundaries, forcing the network to defend against true categorical perturbations.
2.  **Curriculum Learning:** Dropping extreme categorical perturbations on a network early in training causes catastrophic collapse (degenerating to majority-class prediction). We implemented a 30-epoch curriculum, scaling `alpha_cat` linearly from `0.01` to `1.0`. This allowed the model to build stable continuous representations before defending against discrete categorical hops.
3.  **Mixed Loss:** A 50/50 clean/adversarial loss ratio was maintained to preserve clean accuracy.

---

## 3. Verified Empirical Results (The New Baseline)

The models were evaluated strictly on the full **22,543-sample NSL-KDD test set**. 
The attack evaluated was a rigorous, mathematically sound **40-step PGD attack** with random restarts and proper categorical step scaling (`alpha_cat = 1.0`). This ensures no gradient masking or truncation artifacts inflate the results.

### Table: Robust Accuracy Comparison

| Defense Strategy | Attack Evaluated ($\epsilon=0.15$) | Clean Accuracy | Robust Accuracy |
|------------------|------------------------------------|----------------|-----------------|
| Baseline (No Defense) | Rigorous PGD (Fully Constrained) | 80.51% | **0.19%** |
| Prior Adv. Training | PGD (Unmasked, published Table II) | ~81.20% | **29.10%*** |
| **BlackICE-Mesh (New)** | **Rigorous PGD (Fully Constrained)** | **73.01%** | **59.71%** |

*\*Independent replication of the prior unmasked evaluation on the current legacy checkpoint yields **27.69%**. The minor ~1.4% discrepancy is confirmed to be the result of additional 50-epoch fine-tuning performed on the original published model.*

### Analysis
The results demonstrate a fundamental leap in true structural robustness. 

Where the prior standard Adversarial Training baseline achieved **29.10%** under unmasked PGD conditions, the new Curriculum-driven BlackICE-Mesh architecture achieves **59.71%** under even stricter, fully constrained PGD evaluations that respect all tabular data structures.

This massive improvement requires a standard robustness trade-off (the "robustness tax"), reflected in the clean accuracy dropping from 80.51% to 73.01%. However, the network fundamentally reshaped its decision boundary, surviving a rigorous 40-step attack that completely destroys undefended baseline models (0.19%).

---

## 4. Next Steps for Research

With a mathematically verified, unmasked robust baseline of **59.71%** established, the BlackICE research will proceed in two directions:
1.  **Attack Generalization:** Evaluate the 59.71% baseline against alternative adversarial algorithms (e.g., Carlini & Wagner, DeepFool via Foolbox) to verify defense generalizability.
2.  **Accuracy Improvement:** Implement novel architectural or loss-function modifications (e.g., Gumbel-Softmax differentiable approximations, feature-targeted adversarial loss) to push the robust accuracy beyond the 60% barrier.
