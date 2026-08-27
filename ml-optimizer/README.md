# ML-Optimizer

> PyTorch core for mixed-norm ($L_\infty$ / $L_0$) adversarial training and evaluation on tabular network intrusion detection datasets.

## Evaluation Methodology

### Canonical Constants

```python
EVAL_EPSILON = 0.15        # L_inf perturbation budget
EVAL_ALPHA_CONT = 0.01     # Continuous step size
EVAL_PGD_STEPS = 40        # PGD iterations
```

Defined in `app/ml/attacks/eval_protocol.py`.

### Three Evaluation Conventions

| Convention | Categorical handling | α_cat | Denominator | Hardened @ ε=0.15 (NSL-KDD) |
|---|---|---|---|---|
| **Legacy** (AdvGuard original) | argmax→one-hot, alpha_cat=0.01 (dead code) | 0.01 | correct/total | 29.10% (external) / 27.69% (this repo) |
| **SNAP** (gradient-snapped K=1) | Single best-by-gradient flip, applied once at end | 1.0 | full test set | 40.36% (faithful, full-scale) |
| **EXH** (exhaustive K=1) | Enumerate all |G| one-hot states, continuous PGD on each, pick worst | 0.0 | clean-correct | 0.00% |

**Canonical result for Section III:** 40.36% (SNAP, full test set, n=22,543).

### Verified Snap Logic

The original AdvGuard snap was ported from git history (`app/ml/attacks/pgd.py`) and confirmed to produce 40.36% at full scale:

```python
# Faithful snap: argmax → one_hot → float (from section3_faithful_diagnostic.py)
adv_cat = torch.argmax(adv[:, g], dim=1)
adv[:, g] = F.one_hot(adv_cat, num_classes=group_size).float()
```

Key property: categorical features are snapped once at the end of each PGD step (not continuously), making the attack K=1 by construction.

### Checkpoint Provenance

- **`models/model.pth`** (SHA: `8cbcb9d5...`) — Baseline NSL-KDD
- **`models/model_adv.pth`** (SHA: `f07d2e37...`) — Legacy hardened NSL-KDD (fine-tuned 50 epochs beyond published AdvGuard weights; baseline report §1)

Legacy checkpoints are not comparable to unified pipeline checkpoints (`models/unified/model_adv_*_seed*.pth`), which are retrained from scratch with identical hyperparameters.

### PGD Nondeterminism

`unified_pgd.py` uses random epsilon-ball initialization (line 35). Robust-accuracy figures are reproducible in distribution (±~1pp at n=500) rather than bit-for-bit deterministic.

### Small-Sample Fragility

At n=100, the faithful evaluation yields 48.00% for the legacy hardened model; at n=22,543, it drops to 40.36% — a 7.64pp overestimate at small scale.

## Diagnostic Scripts

| Script | Purpose |
|---|---|
| `scripts/consolidated_canonical_table.py` | EXH vs SNAP side-by-side aggregation across all datasets |
| `scripts/section3_faithful_diagnostic.py` | Ported original AdvGuard snap logic with corrected defaults |
| `scripts/eval_scalability.py` | Scalability study (batched, GPU) — near-linear scaling confirmed |
| `scripts/eval_jsma_vs_exhaustive.py` | JSMA vs exhaustive comparison (12.0% false negatives, 10x slower) |
| `scripts/trace_logit_reversal.py` | Logit trace for Section IV-C (margin-bounded robustness) |
| `scripts/eval_deepfool_k1.py` | Canonical exhaustive K=1 evaluator (α_cat=0.0, clean-correct) |
| `scripts/eval_unified.py` | Gradient-snapped K=1 evaluator (α_cat=1.0, full test set) |

## Results

Results are stored in `results/consolidated/` and `results/section3/`. The workstream summary is at `results/WORKSTREAM_SUMMARY.md`.

## Key ADRs and Postmortems

- `docs/01-documentation/adrs/001-canonical-exhaustive-evaluation.md` — Three conventions, full framework
- `docs/02-postmortems/001-phantom-robustness-ceiling.md` — 77.28% re-characterization, three-step staircase
- `docs/02-postmortems/002-dataset-dependent-failure-confounds.md` — Engineering confounds in training
