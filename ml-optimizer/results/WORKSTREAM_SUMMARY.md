# Research Workstreams — Execution Summary

**Date**: 2026-08-26  
**Branch**: `refactor/directory-structure`  
**Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM), 10-core x86_64, 11GB RAM  
**Protocol**: PGD-40, ε=0.15, α_cont=0.01, α_cat=0.0

---

## Workstream 1: Threat-Feasibility Audit ✅

**Status**: Complete  
**Deliverable**: Feature dictionaries, classification tables, proposed Section IV wording

| Feature | Source Paper | Our Pipeline | Attacker Settable? | Threat Model |
|---------|-------------|-------------|-------------------|-------------|
| proto | `is_ftp_login`-free, no protocol features | 11-dim one-hot (UNSW-NB15) | YES — TCP header | Conservative upper bound |
| service | 13-dim one-hot | YES — port/behavior | Conservative upper bound |
| state | 11-dim one-hot | INDIRECTLY — not fully attacker-settable | Conservative upper bound |
| is_sm_ips_ports | 2-dim one-hot | INDIRECTLY — observable side-channel | Conservative upper bound |
| is_ftp_login | 4-dim one-hot | INDIRECTLY — requires FTP session | Conservative upper bound |

**Decision**: Keep all groups as conservative upper bound. Paper needs disclosure sentence (drafted in Workstream 1).

**Files**: Feature dictionaries and classification tables in conversation history (Workstream 1 deliverable).

---

## Workstream 2: Evaluator Scalability & Complexity Study ✅

**Status**: Complete  
**Script**: `scripts/eval_scalability.py`  
**Summary**: `results/scalability_unsw_nb15_summary.json`

### Results (UNSW-NB15, RSC seed42, 500 samples, GPU)

| Config | Candidates | Time (s) | s/sample | K0% | KK% |
|--------|-----------|----------|----------|-----|-----|
| K=1, G=[service] | 14 | 1.4 | 0.0029 | 99.8 | 98.4 |
| K=1, G=[service, proto] | 25 | 4.2 | 0.0084 | 99.8 | 97.8 |
| K=1, G=[service, proto, state] | 36 | 7.0 | 0.0140 | 99.8 | 87.6 |
| K=1, G=[service, proto, state, is_sm_ips_ports] | 38 | 7.2 | 0.0145 | 99.8 | 87.0 |
| K=1, G=[all 5 groups] | 42 | 7.8 | 0.0155 | 99.8 | 86.2 |
| **K=2, G=[all 5 groups]** | **667** | **59.0** | **0.1179** | **99.8** | **61.6** |

### Key Findings
1. K=1 scaling is near-linear (R² ≈ 1.0)
2. K=2: 15.9x more candidates → 7.6x wall-clock (GPU batching efficiency)
3. Per-candidate time decreases at scale (0.38ms → 0.18ms) due to amortized kernel launch
4. K=1 robustness: 86.2%; K=2 robustness: 61.6%
5. State group (|G|=11) is the dominant cost driver in K=1
6. Checkpoint: `6a42ed41d1ec760d3f1eb26c77f49938cd8e3db89451c67f9a777b1d600e434c`

### Complexity Formula
```
T(N, K, G) = N × C(K, |G|) × t_pgd
C(1, G) = 1 + Σ|g_i|
C(2, G) = 1 + Σ|g_i| + Σ_{i<j}|g_i|×|g_j|
```

**Device note**: Wall-clock times use GPU; production evaluator (`eval_deepfool_k1.py`) hardcodes CPU. Reported quantity is relative scaling trend, not absolute cost.

---

## Workstream 3: JSMA vs Exhaustive Comparison ✅

**Status**: Complete  
**Script**: `scripts/eval_jsma_vs_exhaustive.py`  
**Output**: `results/jsma_vs_exhaustive_unsw_nb15_K1.json`

### Methodology
- **Exhaustive**: Enumerates ALL 42 candidate states, applies PGD-40 to each, takes AND survival
- **JSMA**: Greedily selects most salient group (by gradient |max|), applies PGD-40 to that single candidate
- Both use identical PGD-40 evaluation backend — the only difference is selection strategy

### Results (UNSW-NB15, K=1, 500 samples)

| Metric | Exhaustive | JSMA |
|--------|-----------|------|
| Robust count | 437/500 | 497/500 |
| Robust % | 87.4% | 99.4% |
| Wall-clock | 3.37s | 39.46s |
| s/sample | 0.0067 | 0.0789 |

### Divergence Analysis
| Category | Count |
|----------|-------|
| Both robust | 436 |
| Both vulnerable | 2 |
| **JSMA overestimates robustness** | **61** (false negatives) |
| JSMA underestimates robustness | 1 (false positive) |

### Key Findings
1. **JSMA systematically overestimates robustness** by 12.0% (61/500 false negatives)
2. **JSMA is 10x slower** due to per-sample evaluation (no batching)
3. JSMA's gradient-based saliency is unreliable for one-hot categorical groups: gradient direction ≠ optimal discrete flip
4. The single case where JSMA underestimates robustness (1/500) confirms JSMA's selection is noisy
5. Greedy 1-step selection misses adversarial flips that exhaustive enumeration finds

**Why JSMA fails**: Gradient saliency computes ∂logit/∂x in continuous space, but the actual flip is discrete (one-hot column swap). The most salient feature (by gradient) is not the most adversarial feature (by actual discrete flip).

---

## Checkpoint Verification
All results use verified checkpoint: `sha256:6a42ed41d1ec760d3f1eb26c77f49938cd8e3db89451c67f9a777b1d600e434c`  
(Located at `models/unified/model_adv_rsc_unsw_nb15_seed42.pth`)

## Artifacts Produced
- `scripts/eval_scalability.py` — Scaling study script (K=1/K=2, GPU-batched)
- `scripts/eval_jsma_vs_exhaustive.py` — JSMA vs exhaustive comparison
- `scripts/trace_logit_reversal.py` — Logit trace for Section IV-C fix
- `results/scalability_unsw_nb15_summary.json` — Consolidated timing results
- `results/scalability_unsw_nb15_K1_g*.json` — Per-config timing (5 files)
- `results/scalability_unsw_nb15_K2_g*.json` — K=2 timing
- `results/jsma_vs_exhaustive_unsw_nb15_K1.json` — JSMA comparison
- `results/logit_trace_nsl_kdd.json` — Logit trace output (NSL-KDD)
