# Independent Verification Report — BlackICE-Mesh DACM Results

**Date:** 2026-08-13  
**Method:** All code was re-implemented from scratch (no imports from `app.ml.attacks`) and run independently against the same data and model weights.  
**Verdict:** Two bugs confirmed outright, one confirmed with nuance, and some prior claims are partially wrong.

---

## Verification Environment

| Property | Value |
|----------|-------|
| Device | CUDA (PyTorch 2.9.1+cu128) |
| Test set | NSL-KDD, 22,543 samples |
| Features | 4 continuous + 3 protocol one-hot + 11 service one-hot = 18 |
| Label split | class 0: 9,710 (43.1%), class 1: 12,833 (56.9%) |
| Baseline weights | [model.pth](ml-optimizer/app/ml/model.pth) |
| Hardened weights | [model_adv.pth](ml-optimizer/app/ml/model_adv.pth) |

---

## Data Integrity ✅

| Check | Result |
|-------|--------|
| CSV shape | (22543, 19) — 18 features + 1 label |
| Protocol one-hot sums = 1.0 | max deviation = 0.000000 |
| Service one-hot sums = 1.0 | max deviation = 0.000000 |
| Continuous range [0, 1] | min=0.0000, max=1.0000 |

---

## Clean Accuracy — Independently Verified ✅

| Model | Independent Result | Prior Claim | Match? |
|-------|-------------------|-------------|--------|
| Baseline | **80.51%** (18150/22543) | ~80.53% | ✅ Within rounding |
| FGSM-Hardened | **80.57%** (18162/22543) | ~78.44% | ⚠️ **Disagrees** |

> [!WARNING]
> The prior conversation claimed the FGSM-hardened model had 78.44% clean accuracy. My independent measurement shows **80.57%**. Both models have nearly identical clean performance (~80.5%), and neither is degraded. The 78.44% number reported earlier appears to be from a different evaluation path or a different model checkpoint.

### Detailed Prediction Analysis (No Degenerate Classifiers)

| Metric | Baseline | FGSM-Hardened |
|--------|----------|---------------|
| Predicts class 0 | 11,941 (53.0%) | 11,915 (52.9%) |
| Predicts class 1 | 10,602 (47.0%) | 10,628 (47.1%) |
| Sensitivity (TPR) | 74.2% | 74.3% |
| Specificity (TNR) | 88.9% | 88.8% |

Both models are healthy classifiers, not degenerate.

---

## Bug 1: Categorical Step Size — CONFIRMED ✅

> **Claim:** `alpha=0.01` is too small to ever flip a one-hot component during PGD attack.

### Mathematical Proof

For a 3-way one-hot `[1, 0, 0]` with worst-case gradient (hot component gets -1, cold gets +1):

| alpha | Hot component | Cold components | argmax | Flips? |
|-------|--------------|----------------|--------|--------|
| 0.01 | 0.99 | 0.01 | Original | ❌ No |
| 0.05 | 0.95 | 0.05 | Original | ❌ No |
| 0.10 | 0.90 | 0.10 | Original | ❌ No |
| 0.50 | 0.50 | 0.50 | **Tied/Flipped** | ✅ Yes |
| 1.00 | 0.00 | 1.00 | **Flipped** | ✅ Yes |

### Empirical Verification (1,000 samples, 40 steps)

| alpha_cat | Mean flip rate/step | Robust acc (subset) |
|-----------|-------------------|-------------------|
| 0.01 | **0.0000%** | 77.90% |
| 0.10 | 0.0000% | 77.90% |
| 0.50 | 0.5787% | 70.40% |
| 1.00 | 7.2950% | 23.80% |

> [!IMPORTANT]
> **Confirmed.** With `alpha_cat=0.01`, the categorical flip rate is **exactly 0.0000%** — not a single category ever changes across 1,000 samples × 40 steps = 40,000 update opportunities. The PGD attack, as implemented, only ever attacks the 4 continuous features (22% of the input space) and leaves the 14 categorical features completely untouched.

### Impact on Full Test Set Robust Accuracy

| Model | alpha_cat=0.01 (buggy) | alpha_cat=1.0 (fixed) | Delta |
|-------|----------------------|---------------------|-------|
| Baseline | **14.83%** | **0.21%** | −14.62 pp |
| FGSM-Hardened | **77.28%** | **22.93%** | −54.35 pp |

The "robust accuracy" numbers with alpha_cat=0.01 are **dramatically inflated** because the attack is only using 4 of 18 features.

---

## Bug 2: Phantom Packets — CONFIRMED ✅

> **Claim:** Unconstrained PGD produces invalid one-hot sums (physically impossible network packets).

### Empirical Verification (1,000 samples)

After unconstrained PGD (ε=0.15, α=0.01, steps=40):

| Feature Group | One-hot sum range | Mean sum | Samples invalid | Non-binary values |
|---------------|------------------|----------|----------------|-------------------|
| Protocol (3-way) | [0.86, 1.25] | 1.0352 | 800/1000 (80%) | 1,629 |
| Service (11-way) | [0.85, 2.35] | 1.6461 | 1000/1000 (100%) | 6,859 |

Example adversarial protocol features:
```
Sample 0: [0.85, 0.15, 0.14]  sum=1.14  ← INVALID
Sample 1: [0.85, 0.15, 0.14]  sum=1.14  ← INVALID
Sample 2: [0.98, 0.09, 0.00]  sum=1.07  ← INVALID
```

> [!IMPORTANT]
> **Confirmed.** 80-100% of samples have invalid one-hot encodings after unconstrained PGD. Service features are particularly bad (mean sum = 1.65, max = 2.35). These represent network packets that simultaneously claim to be multiple protocols/services, which is physically impossible.

---

## Bug 3: Dead Code in dacm_replication_test.py — CONFIRMED ✅

> **New finding not in prior claims:** In [dacm_replication_test.py line 185](../../ml-optimizer/dacm_replication_test.py#L185) and [variance_audit.py line 120](../../ml-optimizer/variance_audit.py#L120), `dacm_snap_categorical(adv, ...)` is called instead of `dacm_snap_categorical(adv_cat_proj, ...)`.

### Code Evidence

```python
# Lines 180-188 of dacm_replication_test.py:
for group_idx, valid_states in zip(CATEGORICAL_GROUPS, bounds):
    adv_cat = adv[:, group_idx] + alpha * grad[:, group_idx].sign()          # ← Compute gradient step
    eta_cat = (adv_cat - orig[:, group_idx]).clamp(-epsilon, epsilon)         # ← L_inf project
    adv_cat_proj = (orig[:, group_idx] + eta_cat).clamp(MIN_VAL, MAX_VAL)    # ← Final projected value
    snapped = dacm_snap_categorical(adv, group_idx, valid_states)            # ← BUG: uses `adv` not `adv_cat_proj`
    adv.data[:, group_idx] = snapped
```

Lines 181-183 compute `adv_cat_proj` but it is **never used**. The snap receives the un-modified `adv` tensor, which is already a valid one-hot from the previous step. Snapping a valid one-hot returns itself — an identity operation.

### Empirical Verification

| Version | Total categorical changes (500 samples × 40 steps) | Robust accuracy |
|---------|---------------------------------------------------|--------------  |
| Buggy (`adv`) | **0** | 79.20% |
| Fixed (`adv_cat_proj`) | **0** | 79.20% |

> [!NOTE]
> Both versions produce identical results because Bug 1 (alpha=0.01 too small) and the L_inf epsilon projection (ε=0.15) together ensure that even the correctly-projected `adv_cat_proj` can never flip a one-hot component. The bugs are **compounding**: Bug 3 makes the gradient step dead code, and even if Bug 3 were fixed, Bug 1 would still prevent any category flips.

---

## Cross-Check: Original pgd.py vs Independent Implementation

The original [pgd.py](../../ml-optimizer/app/ml/attacks/pgd.py) and my independent implementation **do NOT match** (max diff = 0.30, prediction agreement = 65%). This is because:

1. `pgd.py` initializes with random noise on continuous features (line 21-22), my independent implementation does not
2. `pgd.py` reads categorical values from `images` (which accumulates perturbations), while dacm_replication_test.py reads from `adv` 

This confirms there are at least **3 different attack implementations** in the codebase (pgd.py, dacm_replication_test.py, variance_audit.py) with subtly different behavior, making it impossible to know which one produced any given reported number.

---

## Full Test Set Results — Independently Verified

| Model | Clean | Robust (α_cat=0.01, buggy) | Robust (α_cat=1.0, fixed) | Unconstrained |
|-------|-------|---------------------------|--------------------------|---------------|
| **Baseline** | 80.51% | 14.83% | 0.21% | 0.31% |
| **FGSM-Hardened** | 80.57% | 77.28% | 22.93% | 27.69% |

---

## Corrections to Prior Claims

### ❌ Claim: "Robust accuracy collapses to ~0% with fixed alpha_cat"

**Partially wrong.** The prior conversation claimed all models collapse to ~0% with alpha_cat=1.0. My independent verification shows:

- Baseline: **0.21%** — confirmed, effectively zero
- FGSM-Hardened: **22.93%** — **not** zero! The FGSM-hardened model retains meaningful (though much lower) robustness

The prior claim of 0.04% for the FGSM-hardened model appears to be from a different evaluation (perhaps fewer samples, different seed, or different attack parameters).

### ❌ Claim: "FGSM-hardened clean accuracy is 78.44%"

**Wrong.** Independent measurement: **80.57%**. Nearly identical to the baseline (80.51%).

### ✅ Claim: "77.28% was the constrained robust accuracy with buggy attack"

**Confirmed exactly.** My independent implementation reproduces 77.28% (17,421/22,543) for the FGSM-hardened model with alpha_cat=0.01.

### ✅ Claim: "27.69% unconstrained robust accuracy for FGSM-hardened"

**Confirmed exactly.** (6,242/22,543)

### ⚠️ Claim: "Two bugs explain the inflated robustness numbers"

**Three bugs**, actually. In addition to Bugs 1 and 2, the dacm_replication_test.py has dead code for categorical gradient steps (Bug 3), though its effect is masked by Bug 1.

---

## Summary of Verified Facts

> [!CAUTION]
> **The 77.28% robust accuracy number in the paper is not real robustness.** It measures the model's resistance to an attack that only perturbs 4 of 18 features (22% of the input space). When the attack can actually explore the categorical feature space (alpha_cat=1.0), robust accuracy drops to 22.93% — a 54 percentage point collapse.

1. **Bug 1 (alpha_cat=0.01):** Independently confirmed. Zero categorical flips. Attack only touches 4/18 features.
2. **Bug 2 (phantom packets):** Independently confirmed. 80-100% of unconstrained adversarial samples have invalid one-hot encodings.
3. **Bug 3 (dead code):** Independently confirmed. Gradient step for categoricals is computed but never used in dacm_replication_test.py.
4. **Clean accuracy:** Both models ~80.5%. The 78.44% claim for FGSM-hardened is wrong.
5. **77.28%:** Exactly reproduced, but this is a bogus number measuring resistance to a crippled attack.
6. **0% collapse:** Only true for baseline. FGSM-hardened retains 22.93% with fixed alpha_cat.

---

*Verification script: [independent_verification.py](../../ml-optimizer/independent_verification.py)*  
*Bug 3 script: [verify_bug3.py](../../ml-optimizer/verify_bug3.py)*
