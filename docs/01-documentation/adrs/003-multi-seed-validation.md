# 📜 ADR-003: Multi-Seed Statistical Validation Before Ranking Claims

> **Status:** `Decided`  
> **Date:** August, 2026

---

## 🌎 Context

After completing the unified retraining pipeline (ADR-002), initial single-run results showed a striking ranking on CICIDS2017:

| Method | K=1 Robust Accuracy |
|---|---|
| RSC | 70.05% |
| Hardened | 33.45% |
| Curriculum | 30.97% |

This appeared to be a clean, decisive win for RSC. However, the gaps between Hardened and Curriculum were suspiciously close, and RSC's dominance stood in contrast to its poor NSL-KDD performance (0.81%). Prior rounds of this project had already demonstrated twice that "clean-looking" result gaps turned out to hide silent training failures or evaluation artifacts rather than genuine findings.

A single seed provides zero variance information. A 39pp gap could be a reproducible signal or a lucky initialization.

## 🛤️ Options Considered

1. **Report single-seed results directly** — Fast, but risks publishing a variance artifact as a finding. Precedent in this project: the phantom robustness ceiling (Postmortem-001) was also a single-seed artifact.
2. **Multi-seed validation (n=3)** — Run seeds 42, 43, 44. Sufficient to detect gross variance; not sufficient for formal significance tests at high within-group variance.
3. **Full power-sized study (n≥10 per group)** — Statistically rigorous but compute-heavy; reserved for datasets where instability has been confirmed.

---

## 🎯 Decision

> [!IMPORTANT]  
> **Adopt a tiered validation protocol:**  
> - **n=3** as minimum before any ranking claim is made publicly.  
> - **n≥10** specifically for any dataset where n=3 shows standard deviation >10pp — used to determine whether apparent rankings are statistically separable.

## 🧠 Reasoning

The n=3 run revealed the following on CICIDS2017:

| Method | Mean ± Std (K=1) |
|---|---|
| Hardened | 49.5% ± 15.5pp |
| Curriculum | 41.6% ± 27.5pp |
| RSC | 62.1% ± 13.7pp |

The RSC "breakthrough" from the single-seed run dissolved entirely when variance was measured. Curriculum's accuracy ranged from 20.97% to 72.72% across three seeds — a 51.75pp swing from identical hyperparameters. The only variable was random initialization.

This confirmed that the single-seed CICIDS2017 ranking was a variance artifact. The paper's headline finding changed from "RSC significantly outperforms Curriculum" to "all three methods exhibit high initialization variance on CICIDS2017; no method is a robust winner." This is a more scientifically honest and more interesting finding.

**The n=10 extension was triggered** because CICIDS2017's ±22-27pp variance makes it the dataset most at risk of publishing a false ranking. The ANOVA with full data will either confirm "no significant difference" or identify a genuine winner — either outcome is publishable.

_See also: Postmortem-003 (the timing contradiction investigation that preceded the n=3 decision)._
