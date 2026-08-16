#!/bin/bash
set -e
echo "Training CICIDS Baseline and Hardened"
PYTHONPATH=. python app/ml/training/trainer.py --dataset cicids2017
echo "Training UNSW Baseline and Hardened"
PYTHONPATH=. python app/ml/training/trainer.py --dataset unsw_nb15
echo "Training CICIDS Curriculum"
PYTHONPATH=. python scripts/train_pgd_robust.py --dataset cicids2017
echo "Training UNSW Curriculum"
PYTHONPATH=. python scripts/train_pgd_robust.py --dataset unsw_nb15
echo "All training completed!"
