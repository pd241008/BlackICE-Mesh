"""Set-level convergence probe: LinfDeepFool steps=50 vs steps=150, K=0 AND K=1.

Mirrors eval_deepfool_k1.py semantics exactly by importing the same wrapper and
column-split helpers from scripts.eval_foolbox: model.eval(), cat block frozen per
enumerated state (base + every one-hot state per group), eps-projection clamp with
assert, post-hoc misclassification scoring. Compares survivor SETS (indices), not
just counts, between the two step budgets.

Motivates/verifies: 97.3% DF K=0 baseline used in cross-optimizer comparisons.
"""
import hashlib
import json
import os
import sys

import torch
import foolbox as fb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ml.attacks.eval_protocol import EVAL_EPSILON
from app.ml.data.loader import get_test_loader, get_config
from app.ml.models.architecture import TabularMLP
from app.ml.utils.checkpoint import load_model_checkpoint
from scripts.eval_foolbox import ContinuousColumnWrapper, split_columns

METHOD, SEED, DATASET, BATCHES = "rsc", 53, "cicids2017", 30


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


config = get_config(DATASET)
safe = DATASET.replace('-', '_')
ckpt = f"models/unified/model_adv_{METHOD}_{safe}_seed{SEED}.pth"
sha = sha256_file(ckpt)
model = TabularMLP(input_dim=config.FEATURE_DIM)
load_model_checkpoint(model, ckpt, device='cpu')
model.eval()
cont_cols, cat_cols = split_columns(config)
groups = [list(g) for g in config.CATEGORICAL_GROUPS]
wrapper = ContinuousColumnWrapper(model, cont_cols, cat_cols)
EPS = EVAL_EPSILON

results = {}
for steps in (50, 150):
    fb_model = fb.PyTorchModel(wrapper, bounds=(0.0, 1.0), device='cpu')
    attack = fb.attacks.LinfDeepFoolAttack(steps=steps)
    sets_k0, sets_k1 = [], []
    n_att = 0
    for bi, (bx, by) in enumerate(get_test_loader(DATASET, batch_size=512)):
        if bi >= BATCHES:
            break
        with torch.no_grad():
            ok = model(bx).argmax(1) == by
        data, target = bx[ok], by[ok]
        n_att += data.size(0)

        def surv(state):
            wrapper.cat_fixed = state[:, cat_cols].detach()
            _, adv_fb, _ = attack(fb_model, data[:, cont_cols].detach(),
                                  target, epsilons=None)
            delta = (adv_fb - data[:, cont_cols]).clamp(-EPS, EPS)
            assert delta.abs().max().item() <= EPS + 1e-5, "eps projection failed"
            adv = state.clone()
            adv[:, cont_cols] = (data[:, cont_cols] + delta).clamp(0.0, 1.0)
            with torch.no_grad():
                return model(adv).argmax(1) == target

        k0 = surv(data)
        k1 = k0.clone()
        for group in groups:
            for j in group:
                st = data.clone()
                st[:, group] = 0.0
                st[:, j] = 1.0
                k1 &= surv(st)
        sets_k0.append(k0.cpu())
        sets_k1.append(k1.cpu())
    results[steps] = {"k0": sets_k0, "k1": sets_k1, "n_attacked": n_att}

out = {"checkpoint_sha256": sha, "batches": BATCHES,
       "n_attacked": results[50]["n_attacked"], "epsilon": EPS}
for lbl in ("k0", "k1"):
    s50 = torch.cat(results[50][lbl]); s150 = torch.cat(results[150][lbl])
    assert s50.numel() == s150.numel(), "budgets saw different sample counts"
    assert results[50]["n_attacked"] == results[150]["n_attacked"], \
        "budgets traversed different attacked populations"
    out[f"{lbl}_survivors_50"] = int(s50.sum())
    out[f"{lbl}_survivors_150"] = int(s150.sum())
    out[f"{lbl}_pct_of_attacked"] = round(100 * int(s50.sum()) / s50.numel(), 4)
    out[f"{lbl}_set_disagreements"] = int((s50 != s150).sum())
os.makedirs("results/foolbox", exist_ok=True)
json.dump(out, open("results/foolbox/df_set_convergence_probe_rsc_seed53.json", 'w'), indent=1)
print(json.dumps(out, indent=1))
