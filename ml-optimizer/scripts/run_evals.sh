#!/bin/bash
set -e

echo "Evaluating CICIDS2017 models"
PYTHONPATH=. python scripts/eval_mixed_norm.py --dataset cicids2017

echo "Evaluating UNSW-NB15 models"
PYTHONPATH=. python scripts/eval_mixed_norm.py --dataset unsw_nb15

echo "All evaluations completed!"
