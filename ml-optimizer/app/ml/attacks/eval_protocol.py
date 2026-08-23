"""Single source of truth for EVALUATION-time attack budgets.

Every evaluator and mirror/replica implementation MUST import these instead of
redefining them. This module exists because hardcoded eval defaults have twice
been root causes: the alpha_cat training/eval drift, and a PGD mirror silently
using the training-time steps=10 instead of the canonical eval-time 40.

TRAINING-time PGD (steps=10) lives in scripts/train_unified.py — do not mix.
"""

EVAL_EPSILON = 0.15          # L-inf budget on continuous columns
EVAL_ALPHA_CONT = 0.01       # continuous PGD step size
EVAL_PGD_STEPS = 40          # canonical eval-time PGD budget
