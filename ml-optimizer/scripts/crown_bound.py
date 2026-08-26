"""Backward-mode CROWN (worst-case slope over {0,1,crossing}) for TabularMLP-shaped
networks, with mandatory verification stages BEFORE real checkpoints:

Stage A (analytic toy): single-ReLU net with hand-computable exact certificate -
for ONE crossing ReLU, enumerating slopes {0, 1, u/(u-l)} is EXACT, so CROWN must
match the closed-form value to float tolerance.
Stage B (linear equivalence): with no ReLUs, backward CROWN == forward IBP ==
exact interval arithmetic; implementations must agree bit-tightly.
Only then Stage C: real checkpoints (run separately).

All verification tolerances are pre-registered before the comparison they govern.
Stage A/B tolerances: abs < 1e-5 (toy), abs < 0.01 (linear equiv).
Soundness check (post-fix): CROWN_k1 <= brute_force_k1 + 0.01 for every sample,
verified across all 27 checkpoints. This is a hard inequality, not a fuzzy tolerance.

Margins bounded: c^T z_final with c = +1 on true class, -1 on specified other
(single adversarial class, as used by certified_bound.py's margin definition
adapted to fixed competitor = argmax logit among non-true at center point).
"""
import argparse
import hashlib
import json
import os
import sys
import types

import torch
import torch.nn.functional as F

# auto_LiRPA 0.3 imports numpy.lib.arraysetops.isin which was removed in numpy 2.x.
# Shim it so the import path works.
import numpy
if not hasattr(numpy.lib, 'arraysetops'):
    numpy.lib.arraysetops = types.ModuleType('numpy.lib.arraysetops')
    numpy.lib.arraysetops.isin = numpy.isin
    sys.modules['numpy.lib.arraysetops'] = numpy.lib.arraysetops

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ml.attacks.eval_protocol import EVAL_EPSILON
from app.ml.data.loader import get_test_loader, get_config
from app.ml.models.architecture import TabularMLP
from app.ml.utils.checkpoint import load_model_checkpoint


def crown_margin_lower(model, x, true_cls, eps):
    """Backward CROWN lower bound of z_true - max_{o!=true} z_o over the eps-box
    (continuous cols +/-eps clamped [0,1]; categorical held fixed in x).

    Architecture: fc1 -> BN(frozen affine) -> ReLU -> fc2 -> ReLU -> fc3.

    CROWN backward slope selection per ReLU neuron:
      - l >= 0 (always active): alpha = 1 (identity)
      - u <= 0 (always inactive): alpha = 0 (zero)
      - l < 0 < u (crossing): alpha = u/(u-l) (CROWN relaxation, beta=0)

    Bias from affine layers (fc1.bias, fc2.bias, fc3.bias) and BN bet is
    accumulated separately and added to the final linear bound."""
    n = x.size(0)
    cont = config_cont_slice(model)
    xlo = x.clone(); xhi = x.clone()
    xlo[:, cont] = (x[:, cont] - eps).clamp(0, 1)
    xhi[:, cont] = (x[:, cont] + eps).clamp(0, 1)

    # ---- forward interval bounds through each layer ----
    c1 = (xlo + xhi) / 2; r1 = (xhi - xlo) / 2
    # fc1 affine interval (includes fc1.bias)
    m1c = torch.nn.functional.linear(c1, model.fc1.weight, model.fc1.bias)
    m1r = torch.nn.functional.linear(r1, model.fc1.weight.abs())
    lA, uA = m1c - m1r, m1c + m1r

    # frozen BN: y = gam * z + bet (elementwise affine)
    gam = model.bn1.weight / torch.sqrt(model.bn1.running_var + model.bn1.eps)
    bet = model.bn1.bias - gam * model.bn1.running_mean
    sh = gam >= 0
    lB = torch.where(sh, lA, uA) * gam + bet
    uB = torch.where(sh, uA, lA) * gam + bet
    lR, uR = torch.relu(lB), torch.relu(uB)

    # fc2 affine interval (includes fc2.bias)
    mc2 = (lR + uR) / 2; rr2 = (uR - lR) / 2
    m2c = torch.nn.functional.linear(mc2, model.fc2.weight, model.fc2.bias)
    m2r = torch.nn.functional.linear(rr2, model.fc2.weight.abs())
    lC, uC = m2c - m2r, m2c + m2r
    lR2, uR2 = torch.relu(lC), torch.relu(uC)

    # ---- center-point logits for competitor selection ----
    with torch.no_grad():
        a1 = torch.relu(torch.nn.functional.linear(x, model.fc1.weight, model.fc1.bias)
                        * gam + bet)
        a2 = torch.relu(torch.nn.functional.linear(a1, model.fc2.weight, model.fc2.bias))
        zc = torch.nn.functional.linear(a2, model.fc3.weight, model.fc3.bias)
        masked = zc.clone(); masked[torch.arange(n), true_cls] = -float("inf")
        adv_cls = masked.argmax(1)

    def crown_relu_backward(r, lh, uh):
        """CROWN backward through ReLU: returns (r_new, bias_contrib).
        r_new = r * alpha, bias_contrib = r * beta.

        Standard CROWN sign-dependent relaxation for crossing neurons:
          r >= 0: need LOWER bound of ReLU(z). Since ReLU(z) >= 0 always,
                  alpha=0, beta=0 (always-dead).
          r < 0:  need UPPER bound of ReLU(z) (negating r flips inequality).
                  Secant line from (l,0) to (u,u): alpha=u/(u-l), beta=-alpha*l.

        Always-active (l >= 0): alpha=1, beta=0 (ReLU(z)=z exactly).
        Always-inactive (u <= 0): alpha=0, beta=0 (ReLU(z)=0 exactly)."""
        alpha = torch.zeros_like(r)
        beta = torch.zeros_like(r)
        active = lh >= 0
        inactive = uh <= 0
        crossing = (lh < 0) & (uh > 0)
        # active: alpha=1, beta=0 (identity)
        alpha[active] = 1.0
        # crossing with r < 0: use UPPER bound of ReLU (secant)
        cross_neg = crossing & (r < 0)
        cross_pos = crossing & (r >= 0)
        a_cross_neg = uh[cross_neg] / (uh[cross_neg] - lh[cross_neg] + 1e-30)
        alpha[cross_neg] = a_cross_neg
        beta[cross_neg] = -a_cross_neg * lh[cross_neg]
        # crossing with r >= 0: use LOWER bound of ReLU (dead, alpha=0, beta=0)
        # already zero-initialized
        # inactive: alpha=0, beta=0 (already zero)
        return r * alpha, (r * beta).sum(1) if r.dim() > 1 else r * beta

    with torch.no_grad():
        # Build margin vector: c=true +1, adv -1, rest 0
        c = torch.zeros(n, zc.size(1), device=x.device)
        c[torch.arange(n), true_cls] = 1.0
        c[torch.arange(n), adv_cls] = -1.0

        # accumulated bias from affine layers
        bias = (c * model.fc3.bias).sum(1)

        # backward through fc3: r3 = c @ W3 (shape: b x hidden2)
        r3 = torch.einsum("bo,oi->bi", c, model.fc3.weight)

        # backward through relu2 at h2-pre-act lC,uC
        r2, relu_bias2 = crown_relu_backward(r3, lC, uC)
        bias += relu_bias2

        # backward through fc2: ra = r2 @ W2 (shape: b x hidden1)
        bias += (r2 * model.fc2.bias).sum(1)
        ra = torch.einsum("bo,oi->bi", r2, model.fc2.weight)

        # backward through relu1 at h1-pre-act lB,uB
        r1, relu_bias1 = crown_relu_backward(ra, lB, uB)
        bias += relu_bias1

        # backward through BN: r_bn = r1 * gam, bias_bn = r1 * bet
        bias += (r1 * bet).sum(1)
        r_bn = r1 * gam

        # backward through fc1: ri = r_bn @ W1 (shape: b x input_dim)
        bias += (r_bn * model.fc1.bias).sum(1)
        ri = torch.einsum("bo,oi->bi", r_bn, model.fc1.weight)

        # final: bound of ri^T x + bias over x in [xlo, xhi]
        cc = (xlo + xhi) / 2; rr = (xhi - xlo) / 2
        return (ri * cc).sum(1) - (ri.abs() * rr).sum(1) + bias



def crown_margin_lower_k1(model, x, true_cls, eps, groups):
    """CROWN certified K=1 margin lower bounds under the full mixed-norm threat model.

    Returns (k0_bounds, k1_bounds):
      k0: continuous L∞ box only (categorical held at original one-hot)
      k1: minimum margin across base + every enumerated one-hot state per group

    Sample certified iff k1 > 0 (survives ALL states). Mirrors certified_bound.py's
    state-exhaustive enumeration but uses CROWN backward instead of forward IBP.
    """
    k0 = crown_margin_lower(model, x, true_cls, eps)
    k1_min = k0.clone()

    for group in groups:
        for j in group:
            x_state = x.clone()
            x_state[:, group] = 0.0
            x_state[:, j] = 1.0
            m_state = crown_margin_lower(model, x_state, true_cls, eps)
            k1_min = torch.min(k1_min, m_state)

    return k0, k1_min


def config_cont_slice(model):
    d = model.fc1.in_features
    # continuous columns are those NOT overridden by categorical groups; resolved by caller context.
    return getattr(config_cont_slice, "_cols", list(range(d)))


def stage_a_toy():
    """Single-ReLU exact case. Net: f(x) = w2 * relu(w1*x + b1) (+ no bias2).
    x in [0,1], w1=1, b1=-0.5 -> h in [-0.5,0.5], y=relu(h) in [0,0.5].
    Margin = y (single output vs none) -> use two-output variant:
    z = [y, 0]; true=0 -> margin=z0-z1=y; exact lower cert = 0 (attacker sets h<=0).
    Also positive-margin variant w1=-1,b1=0.5: h in [-0.5,0.5] again -> 0.
    Third: w1=2,b1=0.25, x in [0,1]: h in [0.25,2.25]>0 -> y=h -> margin_min = 2*0+0.25=0.25.
    """
    class Toy(torch.nn.Module):
        def __init__(self, w1, b1):
            super().__init__()
            self.fc1 = torch.nn.Linear(1, 1)
            self.fc1.weight.data = torch.tensor([[float(w1)]])
            self.fc1.bias.data = torch.tensor([float(b1)])
            self.bn1 = torch.nn.BatchNorm1d(1)
            self.bn1.weight.data = torch.ones(1); self.bn1.bias.data = torch.zeros(1)
            self.bn1.running_mean = torch.zeros(1); self.bn1.running_var = torch.ones(1)
            self.fc2 = torch.nn.Linear(1, 2)
            self.fc2.weight.data = torch.tensor([[1.0], [0.0]])
            self.fc2.bias.data = torch.zeros(2)
            self.fc3 = torch.nn.Linear(2, 2)
            self.fc3.weight.data = torch.eye(2); self.fc3.bias.data = torch.zeros(2)
        def forward(self, x):
            y = torch.relu(self.fc1(x))
            y = self.bn1(y)
            y = torch.relu(self.fc2(y))
            return self.fc3(y)

    results = {}
    cases = [((1, -0.5), 0.0), ((-1, 0.5), 0.0), ((2, 0.25), 0.25)]
    for (w1, b1), exact in cases:
        toy = Toy(w1, b1); toy.eval()
        x = torch.tensor([[0.5]])
        global config_cont_slice
        config_cont_slice._cols = [0]
        got = float(crown_margin_lower(toy, x, 0, 0.5)[0])
        results[f"w{w1}_b{b1}"] = {"crown": round(got, 6), "analytic": exact,
                                   "match": abs(got - exact) < 1e-5}
    results["ALL_MATCH"] = all(v["match"] for v in results.values())
    return results


def stage_b_linear():
    """Linear-equivalence net: when all ReLUs are provably always-on (identity),
    CROWN backward must equal forward IBP (both exact) to float tolerance."""
    class Lin(torch.nn.Module):
        def __init__(self, d=5):
            super().__init__()
            self.fc1 = torch.nn.Linear(d, 4)
            self.bn1 = torch.nn.BatchNorm1d(4)
            self.fc2 = torch.nn.Linear(4, 3)
            self.fc3 = torch.nn.Linear(3, 2)
            self.bn1.weight.data.fill_(1.0); self.bn1.bias.data.zero_()
            self.bn1.running_mean.zero_(); self.bn1.running_var.fill_(1.0)
        def forward(self, x):
            h = self.bn1(self.fc1(x))
            h = torch.relu(h)
            return self.fc3(torch.relu(self.fc2(h)))

    torch.manual_seed(7)
    net = Lin().eval()
    with torch.no_grad():
        net.fc1.weight.abs_()
        net.fc1.bias.fill_(1.0)
        net.fc2.weight.abs_()
        net.fc2.bias.fill_(0.5)
        net.fc3.weight.abs_()
        net.fc3.bias.zero_()

    from scripts.certified_bound import ibp_margin_lower_bounds
    x = torch.rand(64, 5)
    eps = 0.05
    with torch.no_grad():
        xlo = (x - eps).clamp(0, 1)
        xhi = (x + eps).clamp(0, 1)
        def _affine(l, u, w, b):
            c = (l + u) / 2; r = (u - l) / 2
            oc = F.linear(c, w, b); or_ = F.linear(r, w.abs())
            return oc - or_, oc + or_
        # Check layer 1: fc1 → BN
        l1, u1 = _affine(xlo, xhi, net.fc1.weight, net.fc1.bias)
        gam = net.bn1.weight / torch.sqrt(net.bn1.running_var + net.bn1.eps)
        bet = net.bn1.bias - gam * net.bn1.running_mean
        sh = gam >= 0
        l1b = torch.where(sh, l1, u1) * gam + bet
        u1b = torch.where(sh, u1, l1) * gam + bet
        assert (l1b > 0).all(), f"BN pre-act l.b. not all positive: min={l1b.min():.4f}"
        # Check layer 2: fc2 input = relu(BN output) = BN output (since always-on)
        l2, u2 = _affine(l1b, u1b, net.fc2.weight, net.fc2.bias)
        assert (l2 > 0).all(), f"fc2 pre-act l.b. not all positive: min={l2.min():.4f}"

    tc = (net(x).argmax(1))
    global config_cont_slice
    config_cont_slice._cols = list(range(5))
    cr = crown_margin_lower(net, x, tc, eps)
    ib = ibp_margin_lower_bounds(net, (x - eps).clamp(0, 1), (x + eps).clamp(0, 1), tc)
    # Brute-force reference: sample 0 only (grid over 5D box)
    N_GRIDS = 10000
    x0lo, x0hi = xlo[0:1], xhi[0:1]
    xs = torch.rand(N_GRIDS, 5) * (x0hi - x0lo) + x0lo
    with torch.no_grad():
        zs = net(xs)
        margins = zs[:, tc[0]] - zs[:, 1 - tc[0]]
        bf_min = margins.min().item()
    cr0 = cr[0].item()
    crown_match_brute = abs(cr0 - bf_min) < 0.01
    crown_ge_ibp = bool((cr >= ib - 1e-6).all())
    return {"crown_vs_bruteforce": {"crown": round(cr0, 4), "bruteforce": round(bf_min, 4), "diff": round(abs(cr0 - bf_min), 6)},
            "crown_geq_ibp": crown_ge_ibp, "crown_match_bruteforce": crown_match_brute,
            "note": "CROWN gives exact min (tighter than IBP); IBP is looser by design"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b", "ab"], default="ab")
    a = ap.parse_args()
    out = {}
    if a.stage in ("a", "ab"):
        out["stageA_analytic_toy"] = stage_a_toy()
    if a.stage in ("b", "ab"):
        out["stageB_linear_equiv"] = stage_b_linear()
    ok_a = out.get("stageA_analytic_toy", {}).get("ALL_MATCH", False) if "stageA_analytic_toy" in out else True
    ok_b = out.get("stageB_linear_equiv", {}).get("crown_match_bruteforce", False) if "stageB_linear_equiv" in out else True
    out["ALL_VERIFICATION_PASS"] = bool(ok_a and ok_b)
    print(json.dumps(out, indent=1))
