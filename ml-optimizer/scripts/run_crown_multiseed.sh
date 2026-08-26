#!/bin/bash
# CROWN vs IBP comparison across all checkpoints: 3 datasets × 3 methods × 3 seeds.
# CROWN is O(N * d * hidden^2) — fast for TabularMLP. Each checkpoint ~2-5 min.
set -e
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
mkdir -p logs

for dataset in nsl-kdd cicids2017 unsw_nb15; do
  safe_name="${dataset//-/_}"
  for method in hardened curriculum rsc; do
    for seed in 42 43 44; do
      ckpt="models/unified/model_adv_${method}_${safe_name}_seed${seed}.pth"
      [ ! -f "$ckpt" ] && echo "SKIP $method/$dataset/seed$seed (no checkpoint)" && continue
      tag="${method}_seed${seed}"
      out="results/certificates/crown_vs_ibp_${dataset}_${tag}.json"
      [ -f "$out" ] && echo "SKIP $method/$dataset/seed$seed (already done)" && continue
      echo "RUN  $method/$dataset/seed$seed $(date +%H:%M:%S)"
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        python scripts/run_crown_vs_ibp.py \
          --dataset "$dataset" --method "$method" --seed "$seed" \
          > "logs/crown_${method}_${dataset}_seed${seed}.log" 2>&1
      if [ -f "$out" ]; then
        echo "OK   $method/$dataset/seed$seed $(date +%H:%M:%S)"
      else
        echo "FAIL $method/$dataset/seed$seed (see logs/crown_${method}_${dataset}_seed${seed}.log)"
      fi
    done
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
