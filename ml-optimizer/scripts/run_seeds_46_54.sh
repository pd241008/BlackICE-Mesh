#!/bin/bash
# Extended CICIDS2017 seed sweep: seeds 46-54 x {hardened, curriculum, rsc}.
# Sequential execution + skip-if-exists resume (ADR-004). Logs under logs/.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs models/unified results/unified
export PYTHONUNBUFFERED=1

# Hard requirement: never fall back to CPU training (hours per model).
if ! python -c "import sys; import torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[SWEEP] $(date '+%F %T') NO CUDA AVAILABLE — aborting (GPU lost?)" >> logs/sweep_status.log
    exit 2
fi

SEEDS=(46 47 48 49 50 51 52 53 54)
METHODS=(hardened curriculum rsc)
DATASET="cicids2017"

echo "[SWEEP] Started $(date)" >> logs/sweep_status.log

for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        ckpt="models/unified/model_adv_${method}_${DATASET}_seed${seed}.pth"
        if [ -f "$ckpt" ]; then
            echo "[SWEEP] $(date '+%F %T') SKIP ${method} seed ${seed} (checkpoint exists)" >> logs/sweep_status.log
            continue
        fi
        echo "[SWEEP] $(date '+%F %T') TRAIN ${method} seed ${seed}" >> logs/sweep_status.log
        PYTHONPATH=. python scripts/train_unified.py \
            --dataset "$DATASET" --method "$method" --seed "$seed" \
            > "logs/train_${method}_seed${seed}.log" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "[SWEEP] $(date '+%F %T') FAIL ${method} seed ${seed} (exit ${rc})" >> logs/sweep_status.log
            exit $rc
        fi
        echo "[SWEEP] $(date '+%F %T') DONE ${method} seed ${seed}" >> logs/sweep_status.log
    done
    # Evaluate this seed's three models as soon as they exist.
    echo "[SWEEP] $(date '+%F %T') EVAL seed ${seed}" >> logs/sweep_status.log
    PYTHONPATH=. python scripts/eval_unified.py --seed "$seed" --datasets cicids2017 \
        > "logs/eval_seed${seed}.log" 2>&1 || \
        echo "[SWEEP] $(date '+%F %T') EVAL-FAIL seed ${seed}" >> logs/sweep_status.log
done

echo "[SWEEP] $(date '+%F %T') ALL COMPLETE" >> logs/sweep_status.log
