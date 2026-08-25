#!/bin/bash
# Exhaustive-state PGD-40 K=1 multiseed sweep: UNSW-NB15, seeds {42,43,44,46..51} x 3 methods.
# 42 enumerated states/sample (base + every one-hot state across 5 categorical groups).
# 5 workers x 2 threads; flock queue; resumable (skip-if-complete via partial flag).
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
QUEUE=/tmp/opencode/unsw_exh_queue.txt
LOCK=/tmp/opencode/unsw_exh_queue.lock
: > "$QUEUE"
for seed in 42 43 44 46 47 48 49 50 51; do
  for method in hardened curriculum rsc; do
    echo "$method $seed" >> "$QUEUE"
  done
done

worker() {
  local wid=$1
  while :; do
    local job
    job=$(flock "$LOCK" bash -c "head -n1 '$QUEUE'; sed -i '1d' '$QUEUE'")
    [ -z "$job" ] && break
    set -- $job
    local m=$1 s=$2
    echo "[w$wid] START $m seed$s $(date +%H:%M:%S)" >> logs/unsw_exh_sweep_status.log
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python scripts/eval_deepfool_k1.py --attack pgd40 --dataset unsw_nb15 \
        --method "$m" --seed "$s" \
      > "logs/unsw_exh_${m}_seed${s}.log" 2>&1
    if grep -q '"partial": false' "results/foolbox/exh_k1_pgd40_unsw_nb15_${m}_seed${s}.json" 2>/dev/null; then
      echo "[w$wid] OK    $m seed$s $(date +%H:%M:%S)" >> logs/unsw_exh_sweep_status.log
    else
      echo "[w$wid] FAIL  $m seed$s (see logs/unsw_exh_${m}_seed${s}.log) $(date +%H:%M:%S)" >> logs/unsw_exh_sweep_status.log
    fi
  done
  echo "[w$wid] exiting" >> logs/unsw_exh_sweep_status.log
}

rm -f logs/unsw_exh_sweep_status.log
for w in 1 2 3 4 5; do worker "$w" & done
wait
echo "ALL WORKERS DONE $(date +%H:%M:%S)" >> logs/unsw_exh_sweep_status.log
