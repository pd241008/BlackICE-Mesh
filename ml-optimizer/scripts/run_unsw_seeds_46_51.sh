#!/bin/bash
# UNSW-NB15 seed extension: seeds 46-51 x {hardened, curriculum, rsc} (n=3 -> n=9).
# Triggered by gap analysis (results/unified/gap_analysis_n3.txt): UNSW K=1
# Hardened-vs-Curriculum and Curriculum-vs-RSC have wide CIs spanning zero.
# Sequential execution + skip-if-exists resume (ADR-004). Logs under logs/unsw_ext/.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs models/unified results/unified
export PYTHONUNBUFFERED=1

# Hard requirement: never fall back to CPU training (hours per model).
if ! python -c "import sys; import torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[UNSW-EXT] $(date '+%F %T') NO CUDA AVAILABLE — aborting (GPU lost?)" >> logs/sweep_status_unsw_ext.log
    exit 2
fi

SEEDS=(46 47 48 49 50 51)
METHODS=(hardened curriculum rsc)
DATASET="unsw_nb15"
LOGDIR="logs/unsw_ext"

mkdir -p "$LOGDIR"
echo "[UNSW-EXT] Started $(date)" >> logs/sweep_status_unsw_ext.log

for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        ckpt="models/unified/model_adv_${method}_${DATASET}_seed${seed}.pth"
        if [ -f "$ckpt" ]; then
            echo "[UNSW-EXT] $(date '+%F %T') SKIP ${method} seed ${seed} (checkpoint exists)" >> logs/sweep_status_unsw_ext.log
            continue
        fi
        echo "[UNSW-EXT] $(date '+%F %T') TRAIN ${method} seed ${seed}" >> logs/sweep_status_unsw_ext.log
        PYTHONPATH=. python scripts/train_unified.py \
            --dataset "$DATASET" --method "$method" --seed "$seed" \
            > "${LOGDIR}/train_${method}_seed${seed}.log" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "[UNSW-EXT] $(date '+%F %T') FAIL ${method} seed ${seed} (exit ${rc})" >> logs/sweep_status_unsw_ext.log
            exit $rc
        fi
        echo "[UNSW-EXT] $(date '+%F %T') DONE ${method} seed ${seed}" >> logs/sweep_status_unsw_ext.log
    done
    # Evaluate this seed's three models as soon as they exist.
    echo "[UNSW-EXT] $(date '+%F %T') EVAL seed ${seed}" >> logs/sweep_status_unsw_ext.log
    PYTHONPATH=. python scripts/eval_unified.py --seed "$seed" --datasets unsw_nb15 \
        > "${LOGDIR}/eval_seed${seed}.log" 2>&1 || \
        echo "[UNSW-EXT] $(date '+%F %T') EVAL-FAIL seed ${seed}" >> logs/sweep_status_unsw_ext.log
done

echo "[UNSW-EXT] $(date '+%F %T') ALL COMPLETE" >> logs/sweep_status_unsw_ext.log
