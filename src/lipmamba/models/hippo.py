"""HiPPO-style initialisation for the eigenvalue parameter.

The standard Mamba/S4 baseline uses HiPPO initialisation for ``A``.  Because
LipMamba reparameterises ``A`` through ``α`` (see :mod:`eigen_reparam`), we
initialise ``α`` so that the resulting eigenvalues match a HiPPO-LegS
spectrum clipped into ``[λ_min, λ_max]``.

Reference: Gu et al., 2020.  *HiPPO: Recurrent Memory with Optimal
Polynomial Projections.*
"""
from __future__ import annotations

import math

import torch


def hippo_legs_eigenvalues(state_dim: int) -> torch.Tensor:
    """HiPPO-LegS real eigenvalues |λᵢ| = (i + 0.5) for i in 0..N-1.

    Returns
    -------
    eigvals : (state_dim,) torch.Tensor of strictly positive floats.
    """
    return torch.arange(state_dim, dtype=torch.float32) + 0.5


def hippo_init(
    alpha: torch.Tensor,
    lambda_min: float,
    lambda_max: float,
    free: bool = False,
) -> None:
    """Fill ``α`` so that the reparameterised eigenvalues track HiPPO-LegS.

    The reparameterisation is ``λ = λ_min + (λ_max-λ_min)·σ(α)``.  We pick the
    target eigenvalues, normalise them into ``[λ_min, λ_max]``, and invert the
    sigmoid.

    Parameters
    ----------
    alpha : Parameter / Tensor of shape ``(D, N)`` (modified in-place).
    lambda_min, lambda_max : float
    """
    if alpha.dim() != 2:
        raise ValueError("alpha must be 2-D (n_channels, state_dim)")
    n_channels, state_dim = alpha.shape
    raw = hippo_legs_eigenvalues(state_dim)
    if free:
        # vanilla Mamba: A = -exp(a_log) with a_log = log(i + 1)
        with torch.no_grad():
            alpha.copy_(torch.log(raw + 0.5).unsqueeze(0).expand(n_channels, state_dim))
        return
    raw = raw / raw[-1]  # in (0, 1]
    # squeeze a touch inside (lambda_min, lambda_max) so logit is finite.
    eps = 1e-3
    target = lambda_min + (lambda_max - lambda_min) * (
        eps + (1.0 - 2 * eps) * raw
    )
    # invert λ = λ_min + (λ_max - λ_min) σ(α)  ⇒  α = logit((λ - λ_min)/(λ_max - λ_min))
    s = (target - lambda_min) / (lambda_max - lambda_min)
    alpha_row = torch.log(s / (1.0 - s))
    with torch.no_grad():
        alpha.copy_(alpha_row.unsqueeze(0).expand(n_channels, state_dim))


def s4d_lin_init(state_dim: int, dt: float = 1.0) -> torch.Tensor:
    """S4D-Lin alternative initialisation (real positive part only).

    Useful for ablations.  Returns positive eigenvalues spaced linearly.
    """
    return 0.5 + torch.arange(state_dim, dtype=torch.float32) * (1.0 / max(state_dim, 1))


def lipmamba_default_alpha(
    n_channels: int,
    state_dim: int,
    lambda_min: float = 0.05,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    """Convenience wrapper: build α tensor pre-filled with HiPPO init."""
    alpha = torch.zeros(n_channels, state_dim)
    hippo_init(alpha, lambda_min, lambda_max)
    return alpha


__all__ = [
    "hippo_init",
    "hippo_legs_eigenvalues",
    "lipmamba_default_alpha",
    "s4d_lin_init",
]
