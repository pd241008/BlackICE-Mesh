#!/bin/bash
# Exhaustive-state PGD-40 K=1: NSL-KDD seeds 42-44 x 3 methods (n=3/method).
# 15 enumerated states/sample; test set 22,543 -> ~20s/model. Serial by design.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
for seed in 42 43 44; do
  for method in hardened curriculum rsc; do
    echo "START $method seed$seed $(date +%H:%M:%S)"
    OMP_NUM_THREADS=8 python scripts/eval_deepfool_k1.py --attack pgd40 \
      --dataset nsl-kdd --method "$method" --seed "$seed" > "logs/exh_${m:-$method}_nsl_seed${seed}.log" 2>&1
    grep -q '"partial": false' "results/foolbox/exh_k1_pgd40_nsl_kdd_${method}_seed${seed}.json" \
      && echo "OK    $method seed$seed" || echo "FAIL  $method seed$seed"
  done
done
echo "NSL EXH SWEEP COMPLETE $(date +%H:%M:%S)"
