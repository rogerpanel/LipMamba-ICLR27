"""PAC-Bayesian training objective.

    L(θ, σ) = L̂_S^{adv}(θ; ε) + L_SSM(θ) · ε / 2
              + β · sqrt( (KL(Q‖P) + ln(2√n / δ)) / (2n) )

The adversarial term is the closed-form margin upper bound (preferred during
training because it is differentiable end-to-end) or the empirical PGD loss
when a stronger attacker is requested for ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..certificates.pac_bayes import (
    PACBayesConfig,
    flatten_constrained_parameters,
    pac_bayes_training_term,
)


def pac_bayes_total_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    empirical_adv_loss: torch.Tensor,
    l_net: torch.Tensor,
    prior_params: torch.Tensor,
    cfg: PACBayesConfig,
) -> dict[str, torch.Tensor]:
    """Return the full training objective and its components.

    Returns
    -------
    dict with keys:
      * ``loss``           — total scalar loss to backprop
      * ``adv_loss``       — empirical adversarial loss
      * ``lipschitz_term`` — L_SSM·ε/2 contribution
      * ``complexity``     — PAC-Bayes complexity term
      * ``kl``             — KL(Q‖P)
    """
    posterior = flatten_constrained_parameters(model)
    pieces = pac_bayes_training_term(
        posterior_params=posterior,
        prior_params=prior_params.to(posterior.device),
        l_ssm=l_net.to(posterior.device),
        cfg=cfg,
    )
    total = empirical_adv_loss + pieces["total"]
    return {
        "loss": total,
        "adv_loss": empirical_adv_loss.detach(),
        "lipschitz_term": pieces["lipschitz_term"].detach(),
        "complexity": pieces["complexity"].detach(),
        "kl": pieces["kl"].detach(),
    }
