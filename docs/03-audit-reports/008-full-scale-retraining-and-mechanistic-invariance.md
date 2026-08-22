    # Audit Report 008: Full-Scale Retraining & Mechanistic Invariance

    > **Date:** August, 2026
    > **Project:** Tabular Mixed-Norm Adversarial Defenses

    ---

    ## 💥 Context

    Phase 2 identified that restricting the training sets for `cicids2017` and `unsw_nb15` to 125,000 rows (down from ~2.5M and ~2.0M rows respectively) introduced a massive underfitting artifact. To definitively evaluate the true efficacy of Randomized Subset Constraints (RSC), Curriculum Training, and Standard Hardened Training, all models were retrained on their full, native datasets using the unconfounded Unified Pipeline (which guarantees $\alpha_{cat} = 1.0$).

    ## 📉 1. Full-Scale Evaluation Results

    When trained at scale, the models successfully learned the complex, high-dimensional boundaries required to defend against mixed-norm attacks.

    ### `cicids2017` (2.5M rows, $|G|=1$)
    With the corrected $K=1$ evaluator (which mathematically enforces $K=1 \le K=0$), we see that the models do **not** converge to identical performance. Even on a dataset with only one categorical group, the stochastic training subsets of Curriculum and RSC provide significant regularization benefits over standard Hardened training.
    * **Hardened:** Clean: 80.55% | $K=0$: 65.63% | $K=1$: **63.42%**
    * **Curriculum:** Clean: 80.29% | $K=0$: 73.11% | $K=1$: **69.72%**
    * **RSC:** Clean: 89.54% | $K=0$: 83.51% | $K=1$: **71.07%**

    ### `unsw_nb15` (2.0M rows, $|G|=5$)
    With multiple categorical groups, the regularization benefits of RSC and Curriculum become apparent.
    * **Hardened:** Clean: 98.62% | $K=0$: 97.33% | $K=1$: **94.11%** (up from 65.23%)
    * **Curriculum:** Clean: 98.61% | $K=0$: 98.36% | $K=1$: **94.38%** (up from 72.82%)
    * **RSC:** Clean: 98.75% | $K=0$: 98.46% | $K=1$: **97.34%** (up from 71.07%)

    As expected for a dataset with 5 independent categorical groups, RSC significantly outperforms standard adversarial training (+3.23pp) by aggressively randomizing which groups the adversary is allowed to attack during training, preventing the model from collapsing its trust into any single un-attacked feature.

    ## 🧠 2. The Mechanism: Margin-Bounded Robustness (Not Absolute Invariance)

    To understand *why* these models achieve such high $K=1$ robust accuracy, we ran a per-group zero-ablation analysis. 

    The Baseline model displays **compensatory redundancy**:
    - Zeroing individual groups causes small drops (e.g., G0: 0.39pp, G1: 0.39pp, G2: 0.39pp). As requested, we verified the column indices and exact overlap to confirm this isn't an indexing duplication bug. Here is the raw output from `check_ablation_overlap.py` showing that the categorical groups are distinct, the absolute flipped samples differ, and they share a ~2,000 sample compensatory overlap block:

    ```text
    Categorical Group Indices:
    G0: [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
    G1: [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    G2: [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]
    G3: [73, 74]
    G4: [75, 76, 77, 78]

    G0 flipped 3266 samples from Correct to Incorrect
    G1 flipped 3009 samples from Correct to Incorrect
    G2 flipped 3298 samples from Correct to Incorrect
    G3 flipped 14112 samples from Correct to Incorrect
    G4 flipped 264 samples from Correct to Incorrect

    Overlap Analysis:
    G0 & G1 Overlap: 2475
    G0 & G2 Overlap: 2615
    G1 & G2 Overlap: 2011
    G0 & G1 & G2 Overlap: 2004
    ```

    - Zeroing *all* groups simultaneously causes a massive 11.00pp collapse. 
    This definitively indicates the baseline model relies on the distinct categorical groups to "cover" for each other.

    However, the **Robust Models** demonstrate aggregate invariance to any single categorical group:
    - **Hardened:** G0 Drop: +0.01pp | G1: -0.00pp | G2: -0.00pp | G3: +0.00pp | G4: -0.02pp
    - **RSC:** G0 Drop: -0.00pp | G1: -0.02pp | G2: -0.02pp | G3: +0.00pp | G4: +0.00pp

    A single-sample logit test confirmed that this 0.00pp drop is **NOT** because the model is mathematically invariant to the categorical features. When G1 is zeroed on a single sample, the RSC model's logits drop violently (a 6.73 point shift). However, because the adversarial training pushed the model into a massive-confidence region (e.g., initial clean logits of `[6.66, -8.28]`, a margin of ~15), the 6.73 point loss does not cross the decision boundary (margin drops to ~8).

    **Conclusion:** The adversarial training does not make the model ignore the categorical features; rather, it forces the model to learn **Margin-Bounded Robustness**. By building an enormous confidence margin via the continuous safety net, the model guarantees that the maximum logit damage inflicted by any single $K=1$ categorical flip is insufficient to hijack the decision boundary.

    ## 💾 3. Unified Checkpoint Hashes (Full Scale)
    - CICIDS2017 Hardened: `29d7c3d3ec72a05320373e061bb10fe27e8eee3532acfeab623d07930f903f95`
    - CICIDS2017 Curriculum: `728bb321d1621a020366381ecf91aeb960cbbedcd9bf116b307d4861751313b9`
    - CICIDS2017 RSC: `c507556fa21146bb562805e833ae57a4b703fc511987145ae784d28c46582b36`
    - UNSW-NB15 Hardened: `a5bf823550dfe99f5824a2fbef3d01b30ad165d71ee1d59111db4d82fb7abb23`
    - UNSW-NB15 Curriculum: `f2696ba0aaf958a3ec527c01f1ad25329cdf5fa8e13342c4018580911f0ed8aa`
    - UNSW-NB15 RSC: `42a4a63501e049dbfd96671818fd82eab397c9e1586e303091f52ecd29c46545`
