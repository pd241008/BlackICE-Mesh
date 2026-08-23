"""Carlini & Wagner (C&W) attack adapted to the BlackIce mixed-norm threat model.

Reference: Carlini & Wagner, "Towards Evaluating the Robustness of Neural
Networks" (IEEE S&P 2017). The original attack minimizes ||delta||_2^2 +
c * f(x + delta) with the margin loss f6, Adam updates, and a per-sample
binary search over the confidence constant c.

Adaptation for this repo's tabular setting:

  - Perturbations live only on CONTINUOUS_COLS and are parameterized as
    delta = epsilon * tanh(p). Unlike the original tanh-space box trick,
    this keeps every iterate strictly inside the L-inf epsilon ball used
    throughout the study (default eps=0.15) AND provides healthy gradients
    at initialization (tanh'(0)=1); saturating arctanh-space decodes would
    otherwise die inside the projection clamp on one-hot-saturated features.
  - Categorical groups are discrete one-hot blocks; gradients do not flow
    through them. Following eval_mixed_norm.canonical_mixed_norm_attack, we
    enumerate all states reachable under the L0 budget K (K=0: original only;
    K>=1: every one-hot reassignment per group), run the optimizer against
    each expanded state, and keep the worst-case state per sample.
"""

import torch
import torch.nn as nn

DEFAULT_EPSILON = 0.15


def _raw_margin(logits, labels):
    """Unclamped CW margin Z_y - max_{t != y} Z_t; negative => misclassified."""
    correct = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.unsqueeze(1), float("-inf"))
    best_wrong = masked.max(dim=1).values
    return correct - best_wrong


def _margin_loss(logits, labels, kappa):
    """CW f6 hinge objective: max(Z_y - max_{t != y} Z_t, -kappa).

    NOTE: this is clamped at -kappa (>= 0 when kappa == 0) purely as the
    minimization target; attack-success detection must use _raw_margin,
    since f < 0 is unsatisfiable whenever kappa == 0.
    """
    return torch.clamp(_raw_margin(logits, labels), min=-kappa)


def cw_mixed_norm_attack(
    model,
    images,
    labels,
    config,
    epsilon=DEFAULT_EPSILON,
    steps=300,
    lr=0.05,
    kappa=0.0,
    binary_search_steps=3,
    confidence_init=1e-2,
    confidence_factor=10.0,
    chunk_size=8192,
    K=0,
):
    """Optimization-based CW attack under the unified mixed-norm threat model.

    Args mirror canonical_mixed_norm_attack where applicable:
      images/labels: batch tensors on the model device.
      config: dataset config exposing FEATURE_DIM, CONTINUOUS_COLS and
        CATEGORICAL_GROUPS.
      epsilon: L-inf bound applied to continuous perturbations.
      steps: Adam iterations per binary-search constant.
      lr: Adam learning rate on tanh-space variables.
      kappa: CW confidence margin.
      binary_search_steps: number of constants c tried (geometric sweep).
      confidence_init / confidence_factor: initial c and geometric growth rate.
      chunk_size: row budget per forward pass when categorical state expansion
        blows up the effective batch size.

    Returns a tensor shaped like `images` holding the strongest adversarial
    example found per sample.
    """
    device = images.device
    batch_size = images.shape[0]

    if not config.CATEGORICAL_GROUPS:
        return _cw_continuous(
            model, images, labels, config, epsilon, steps, lr, kappa,
            binary_search_steps, confidence_init, confidence_factor, chunk_size,
        )

    cat_indices = [idx for group in config.CATEGORICAL_GROUPS for idx in group]
    cat_min, cat_max = min(cat_indices), max(cat_indices)

    orig_cat = images[:, cat_min:cat_max + 1]

    # Enumerate L0-reachable categorical states (same scheme as canonical PGD).
    states_per_sample = [orig_cat]
    if K >= 1:
        rel_groups = [[idx - cat_min for idx in group] for group in config.CATEGORICAL_GROUPS]
        for rel_group in rel_groups:
            for i in range(len(rel_group)):
                new_state = orig_cat.clone()
                new_state[:, rel_group] = 0.0
                new_state[:, rel_group[i]] = 1.0
                states_per_sample.append(new_state)

    all_states = torch.stack(states_per_sample, dim=0)
    num_states = all_states.shape[0]

    expanded_images = images.unsqueeze(0).expand(num_states, -1, -1).clone()
    expanded_images[:, :, cat_min:cat_max + 1] = all_states
    expanded_images = expanded_images.reshape(num_states * batch_size, -1).detach()

    expanded_labels = labels.unsqueeze(0).expand(num_states, -1).reshape(-1)

    adv_rows = _cw_continuous(
        model, expanded_images, expanded_labels, config, epsilon, steps, lr,
        kappa, binary_search_steps, confidence_init, confidence_factor,
        chunk_size,
    )

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        adv_losses = []
        for start_idx in range(0, adv_rows.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, adv_rows.shape[0])
            out = model(adv_rows[start_idx:end_idx])
            adv_losses.append(loss_fn(out, expanded_labels[start_idx:end_idx]))
        adv_losses = torch.cat(adv_losses, dim=0)

    losses = adv_losses.view(num_states, batch_size)
    adv = adv_rows.view(num_states, batch_size, -1)
    _, best_idx = losses.max(dim=0)
    return adv[best_idx, torch.arange(batch_size, device=device)]


def _cw_continuous(
    model,
    images,
    labels,
    config,
    epsilon,
    steps,
    lr,
    kappa,
    binary_search_steps,
    confidence_init,
    confidence_factor,
    chunk_size,
    restarts=1,
    lambda_l2=1.0,
):
    """CW-L2 core loop restricted to continuous columns, chunked for memory."""
    chunks = []
    for start_idx in range(0, images.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, images.shape[0])
        chunks.append(
            _cw_continuous_chunk(
                model, images[start_idx:end_idx], labels[start_idx:end_idx],
                config, epsilon, steps, lr, kappa, binary_search_steps,
                confidence_init, confidence_factor, restarts, lambda_l2,
            )
        )
    return torch.cat(chunks, dim=0)


def _cw_continuous_chunk(
    model,
    images,
    labels,
    config,
    epsilon,
    steps,
    lr,
    kappa,
    binary_search_steps,
    confidence_init,
    confidence_factor,
    restarts=1,
    lambda_l2=1.0,
):
    """Eps-bounded CW-L2 for one chunk; categorical columns frozen."""
    device = images.device
    n = images.shape[0]
    cont_cols = list(config.CONTINUOUS_COLS)

    if not cont_cols:
        # Nothing optimizable under a continuous-only perturbation budget.
        return images.detach().clone()

    orig = images.detach().clone()
    x_cont = orig[:, cont_cols]

    def decode(p):
        return (x_cont + epsilon * torch.tanh(p)).clamp(0.0, 1.0)

    def assemble(adv_cont):
        full = orig.clone()
        full[:, cont_cols] = adv_cont
        return full

    best_adv = orig.clone()
    best_l2 = torch.full((n,), float("inf"), device=device)

    for restart in range(restarts):
        c = torch.full((n,), float(confidence_init), device=device)
        c_low = torch.zeros(n, device=device)
        c_high = torch.full((n,), float("inf"), device=device)

        for _ in range(binary_search_steps):
            if restart == 0:
                p = torch.zeros_like(x_cont, requires_grad=True)
            else:
                p = (torch.rand_like(x_cont) * 2 - 1).mul_(0.1).requires_grad_(True)
            optimizer = torch.optim.Adam([p], lr=lr)

            for step_idx in range(steps):
                adv_cont = decode(p)
                logits = model(assemble(adv_cont))
                f = _margin_loss(logits, labels, kappa)
                l2_sq = ((adv_cont - x_cont) ** 2).sum(dim=1)
                loss = (lambda_l2 * l2_sq + c * f).sum()

                optimizer.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # Capture transient successes DURING optimization: the L2
                # term can re-attract a flipped sample before round end,
                # so end-of-round snapshots alone under-report attack
                # strength (verified against the PGD K=0 anchor).
                if (step_idx + 1) % 50 == 0:
                    with torch.no_grad():
                        adv_cont = decode(p)
                        logits = model(assemble(adv_cont))
                        raw = _raw_margin(logits, labels)
                        l2_sq_now = ((adv_cont - x_cont) ** 2).sum(dim=1)
                        ok = (raw < 0) & (l2_sq_now < best_l2)
                        if ok.any():
                            best_adv[ok] = assemble(adv_cont)[ok].to(best_adv.dtype)
                            best_l2[ok] = l2_sq_now[ok]

            with torch.no_grad():
                adv_cont = decode(p)
                logits = model(assemble(adv_cont))
                raw = _raw_margin(logits, labels)
                f = torch.clamp(raw, min=-kappa)
                l2_sq = ((adv_cont - x_cont) ** 2).sum(dim=1)

                # Success is judged on the RAW margin: the clamped hinge can
                # never go below -kappa (0 when kappa == 0).
                success = raw < 0
                better = success & (l2_sq < best_l2)
                if better.any():
                    rows = assemble(adv_cont)[better]
                    best_adv[better] = rows.to(best_adv.dtype)
                    best_l2[better] = l2_sq[better]

                # Per-sample geometric binary search on the confidence constant.
                newly_success = success & torch.isinf(c_high)
                c_high[newly_success] = c[newly_success]
                failed = (~success) & torch.isinf(c_high)
                c_low[failed] = c[failed]
                resolved = ~torch.isinf(c_high)
                c[resolved] = torch.sqrt(c_low[resolved] * c_high[resolved])
                unresolved_failure = (~success) & torch.isinf(c_high)
                c[unresolved_failure] *= confidence_factor

    return best_adv
