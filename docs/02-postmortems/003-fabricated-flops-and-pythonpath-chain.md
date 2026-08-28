# Postmortem: Fabricated FLOPs Estimate and the PYTHONPATH Silent-Failure Chain

> **Date:** August, 2026  
> **Project:** Tabular Mixed-Norm Adversarial Defenses — Multi-Seed Statistical Validation Phase

---

## 💥 What happened

Two distinct failures occurred in close sequence during the extended seed validation phase. They are documented together because the second was caused by misplaced trust from the first.

### Failure 1: The "157x Forward Pass" Fabricated FLOPs Claim

During a status update explaining why the RSC UNSW-NB15 training was taking longer than expected, a FLOPs estimate was generated that described the RSC training loop as performing **157 exhaustive forward passes per PGD step** — one per valid categorical state. This was presented with specific mathematical language (batch counts, epoch counts, FLOPs/second), giving it the appearance of a derived result.

**It was fabricated.** No calculation was performed. The actual RSC implementation in `unified_pgd.py` (lines 21–57) uses stochastic binary masking:

```python
# Drawn ONCE before the PGD loop — not inside it
k_samples = torch.randint(0, num_groups + 1, ...)
rsc_mask = ranks < k_samples.unsqueeze(1)

# Applied ONCE per step via torch.where — not 157 times
images.data[:, cat_group] = torch.where(active_mask, snapped_tensor, ori_images[:, cat_group])
```

RSC costs exactly **one forward pass and one backward pass per PGD step** — identical to Hardened. When the user challenged the timing discrepancy that this claim created, investigation of the actual source code resolved it immediately.

**UNSW-NB15 also has 5 categorical groups, not 157.** The confusion arose from conflating the total number of one-hot encoded columns (41 across 5 groups) with the number of groups the RSC mask operates over.

### Failure 2: The PYTHONPATH Silent Failure Chain

When the first extended seed run (`cicids2017_extended_seeds.sh`) was written, `PYTHONPATH=.` was omitted from the `python` invocations. The original `resume_multiseed_sequential.sh` always used `PYTHONPATH=.` because `app.*` imports require the ml-optimizer root to be on the Python path.

The extended script ran to completion with exit code 0. The shell reported "Done: cicids2017 | hardened | Seed 45" for all 30 training runs. No error was surfaced. **All 30 runs had silently failed with `ModuleNotFoundError: No module named 'app'`** — because `set -e` was removed earlier to prevent the script from aborting on non-zero exits, and the training script's non-zero exit was swallowed silently by the redirect to a log file.

The failure was only discovered when the aggregation script reported n=3 after the "complete" 10-seed run — the new seed model files simply didn't exist on disk.

---

## 🧠 What we learned

**On fabricated reasoning:**
- Timing estimates stated with mathematical specificity that weren't derived from the code should be treated as red flags, not evidence. The correct response when asked "why is this taking so long?" is to read the training loop, not to generate a plausible-sounding FLOPs narrative.
- When a timing prediction and an actual outcome contradict each other, the hypothesis to investigate first is "my premise was wrong," not "PyTorch optimized it away."

**On silent failures in shell scripts:**
- `set -e` removal must be consciously scoped. Removing it globally to handle one edge case (non-zero exits from expected conditions) creates a silent failure mode for all other commands.
- After any "run completed" message from a script, **verify on-disk artifacts exist** before treating the run as successful. A script that prints "Done" and a script that writes model files are not the same thing.
- New scripts that invoke existing training/eval code should be spot-checked against the original invocation to catch missing env vars (`PYTHONPATH`, `CUDA_VISIBLE_DEVICES`, etc.).

**On the compound effect:**
The PYTHONPATH failure happened immediately after the FLOPs investigation. Attention was focused on the correctness of the RSC mechanism, and the new script was launched quickly to "make up for lost time." The urgency created by the first failure contributed to the second. Slow down after catching one bug before launching the next step.

---

## 🔧 Fixes Applied

1. `unified_pgd.py` verified correct — no code change needed; the implementation was always right.
2. `cicids2017_extended_seeds.sh` — added `PYTHONPATH=.` to both `python` invocations.
3. `eval_unified.py` — added `--datasets` CLI flag to allow filtering to a subset of datasets, enabling the extended script to run CICIDS2017 evaluation only without touching NSL-KDD or UNSW-NB15 (which already have results for seeds 42-44).
4. Post-run verification step added to protocol: after any batch script completes, run `ls models/unified/model_adv_*_seed{N}.pth` before reporting success.

**Checkpoint provenance note:** All `models/unified/model_adv_*_seed*.pth` checkpoints are retrained from scratch via the unified pipeline (ADR-002) with identical hyperparameters. They are not the same weights as the legacy `model_adv.pth` (SHA: `f07d2e37...`), which was inherited from the original AdvGuard and underwent additional fine-tuning. The fabricated FLOPs claim (157x) about RSC's computational cost has been empirically contradicted by the scalability study: the exhaustive evaluator scales near-linearly, with K=2 (667 candidates) completing in 59.0s on an RTX 4050 Laptop.
