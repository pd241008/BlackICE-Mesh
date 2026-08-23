#!/bin/bash
# Exhaustive-state PGD-40 K=1 multiseed sweep: CICIDS2017, seeds 42-54 x 3 methods.
# 5 workers x 2 threads; jobs pulled via flock from a shared queue file.
# Resumable: eval_deepfool_k1.py skips/continues from its own artifacts.

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
QUEUE=/tmp/opencode/exh_queue.txt
LOCK=/tmp/opencode/exh_queue.lock
: > "$QUEUE"
for seed in $(seq 42 54); do
  for method in hardened curriculum rsc; do
    [ "$seed" = 53 ] && continue  # already done & committed
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
    echo "[w$wid] START $m seed$s $(date +%H:%M:%S)" >> logs/exh_sweep_status.log
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python scripts/eval_deepfool_k1.py --attack pgd40 --method "$m" --seed "$s" \
      > "logs/exh_${m}_seed${s}.log" 2>&1
    if grep -q '"partial": false' "results/foolbox/exh_k1_pgd40_cicids2017_${m}_seed${s}.json" 2>/dev/null; then
      echo "[w$wid] OK    $m seed$s $(date +%H:%M:%S)" >> logs/exh_sweep_status.log
    else
      echo "[w$wid] FAIL  $m seed$s (see logs/exh_${m}_seed${s}.log) $(date +%H:%M:%S)" >> logs/exh_sweep_status.log
    fi
  done
  echo "[w$wid] exiting" >> logs/exh_sweep_status.log
}

rm -f logs/exh_sweep_status.log
for w in 1 2 3 4 5; do worker "$w" & done
wait
echo "ALL WORKERS DONE $(date +%H:%M:%S)" >> logs/exh_sweep_status.log
