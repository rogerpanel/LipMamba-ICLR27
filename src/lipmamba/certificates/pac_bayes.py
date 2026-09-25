"""PAC-Bayes bound on adversarial risk (Theorem 3 of the ICLR manuscript).

With ℓ(·,y) ∈ [0,1] L_ℓ-Lipschitz in the logits, Q = N(θ, σ²I), P a prior
independent of S, f_θ L(θ)-Lipschitz on the perturbation domain:

    E_Q[L_adv(θ;ε)] ≤ E_Q[L̂_S(θ)] + E_Q[L_ℓ L(θ)] ε + sqrt((KL(Q‖P) + ln(2√n/δ)) / (2n)).

Training objective (Eq. 5):

    L(θ,σ) = L̂_S^adv(θ) + ½ L_ℓ L_loc(θ) ε_train + β sqrt((KL + ln(2√n/δ)) / (2n)).

The ½ in the objective versus the 1 in the theorem is a design choice of the
manuscript (the adversarial empirical loss already absorbs part of the
gap); both are exposed.  With the *global* Theorem-1 constant the middle
term is vacuous (Remark 4); instantiate with L_loc.

L_ℓ for softmax cross-entropy restricted to [0,1] by clipping is ≤ √2 in the
logits (the log-softmax is 1-Lipschitz in ℓ∞→ℓ1 sense, √2 in ℓ2 for the
two-class margin); we default to L_ℓ = 1.0 and let it be configured.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PACBayesConfig:
    delta: float = 0.05
    sigma_post: float = 0.04       # manuscript instantiation: σ = 0.04
    sigma_prior: float = 0.10      # σ₀ = 0.10
    epsilon_train: float = 0.18
    beta: float = 1.0
    n_train: int = 1
    l_ell: float = 1.0             # Lipschitz constant of the loss in the logits
    objective_half: bool = True    # ½ L_ℓ L ε (Eq. 5) vs L_ℓ L ε (Thm 3)


def gaussian_kl_divergence(posterior_mean: torch.Tensor, prior_mean: torch.Tensor,
                           sigma_post: float, sigma_prior: float) -> torch.Tensor:
    if posterior_mean.shape != prior_mean.shape:
        raise ValueError("posterior and prior must share dimensionality")
    diff_sq = (posterior_mean - prior_mean).pow(2).sum()
    n = posterior_mean.numel()
    vp, vq = sigma_post**2, sigma_prior**2
    return 0.5 * (n * vp / vq + diff_sq / vq - n + 2.0 * n * math.log(sigma_prior / sigma_post))


def pac_bayes_complexity(kl: torch.Tensor, n: int, delta: float) -> torch.Tensor:
    if n <= 0:
        raise ValueError("n must be positive")
    return torch.sqrt((kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n))


def lipschitz_gap_term(l: torch.Tensor | float, cfg: PACBayesConfig, half: bool | None = None) -> torch.Tensor:
    l = torch.as_tensor(l, dtype=torch.float32)
    h = cfg.objective_half if half is None else half
    return (0.5 if h else 1.0) * cfg.l_ell * l * cfg.epsilon_train


def pac_bayes_training_term(posterior_params: torch.Tensor, prior_params: torch.Tensor,
                            l_ssm: torch.Tensor | float, cfg: PACBayesConfig) -> dict[str, torch.Tensor]:
    kl = gaussian_kl_divergence(posterior_params, prior_params, cfg.sigma_post, cfg.sigma_prior)
    complexity = pac_bayes_complexity(kl, cfg.n_train, cfg.delta)
    lip = lipschitz_gap_term(l_ssm, cfg)
    return {"kl": kl, "complexity": complexity, "lipschitz_term": lip,
            "total": lip + cfg.beta * complexity}


def pac_bayes_bound(empirical_clean_loss: torch.Tensor, posterior_params: torch.Tensor,
                    prior_params: torch.Tensor, l: torch.Tensor | float, cfg: PACBayesConfig) -> dict[str, float]:
    """Evaluate Theorem 3's right-hand side (coefficient 1 on the gap term)."""
    kl = gaussian_kl_divergence(posterior_params, prior_params, cfg.sigma_post, cfg.sigma_prior)
    comp = pac_bayes_complexity(kl, cfg.n_train, cfg.delta)
    gap = lipschitz_gap_term(l, cfg, half=False)
    return {"empirical": float(empirical_clean_loss), "gap": float(gap), "kl": float(kl),
            "complexity": float(comp), "bound": float(empirical_clean_loss + gap + comp),
            "vacuous": bool(float(empirical_clean_loss + gap + comp) >= 1.0)}


def flatten_constrained_parameters(model: nn.Module) -> torch.Tensor:
    """Constrained parameters (those that enter the Lipschitz bound)."""
    from ..models.eigen_reparam import EigenReparamA
    from ..models.spectral_norm import SpectralNormLinear
    chunks = []
    for m in model.modules():
        if isinstance(m, SpectralNormLinear):
            chunks.append(m.weight.reshape(-1))
        elif isinstance(m, EigenReparamA):
            chunks.append(m.alpha.reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(1)
