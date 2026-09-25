"""Spectral normalisation primitives.

Implements one-step power iteration with running-average smoothing as
described in Section 3 of the LipMamba paper.  The scaling rule used during
the forward pass is

    W̄ = W · min(1, s / σ̂_max(W))

so that ``‖W̄‖₂ ≤ s`` while leaving the parameter free to take any norm at
initialisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def power_iteration_sigma(
    weight: torch.Tensor,
    u: torch.Tensor,
    n_iters: int = 1,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-step (or n-step) power iteration.

    Parameters
    ----------
    weight : (out, in)
        The weight matrix being normalised.
    u : (out,)
        The previous left singular vector estimate.
    n_iters : int
        Number of power-iteration sweeps.
    eps : float
        Numerical stabiliser.

    Returns
    -------
    sigma : scalar tensor
        The current estimate of ``σ_max(weight)``.
    u_new : (out,)
        Updated left singular vector estimate.
    """
    w_mat = weight.reshape(weight.size(0), -1)
    for _ in range(n_iters):
        v = F.normalize(w_mat.t() @ u, dim=0, eps=eps)
        u = F.normalize(w_mat @ v, dim=0, eps=eps)
    sigma = torch.dot(u, w_mat @ v)
    return sigma, u


class SpectralNormLinear(nn.Module):
    """Linear layer with spectrally-normalised weight ``‖W̄‖₂ ≤ s``.

    The unnormalised weight is a regular ``nn.Parameter``; the normalisation
    factor is recomputed every forward pass during training and frozen at the
    last value during evaluation, which matches the convention from
    Miyato et al. (2018) and the LipMamba manuscript.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 1.0,
        bias: bool = True,
        n_power_iters: int = 1,
        smoothing: float = 0.99,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.n_power_iters = int(n_power_iters)
        self.smoothing = float(smoothing)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # buffers used by power iteration / running estimate
        self.register_buffer("_u", F.normalize(torch.randn(out_features), dim=0))
        self.register_buffer("_sigma_ema", torch.tensor(1.0))

    def normalised_weight(self) -> torch.Tensor:
        """Return ``W · min(1, s / σ̂_max(W))`` with up-to-date σ estimate."""
        if self.training:
            sigma, u_new = power_iteration_sigma(
                self.weight, self._u, n_iters=self.n_power_iters
            )
            self._u.copy_(u_new)
            sigma_ema = self.smoothing * self._sigma_ema + (1.0 - self.smoothing) * sigma
            self._sigma_ema.copy_(sigma_ema.detach())
        else:
            sigma = self._sigma_ema
        scale = torch.clamp(self.s / (sigma + 1e-12), max=1.0)
        return self.weight * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.normalised_weight(), self.bias)

    @property
    def sigma(self) -> torch.Tensor:
        """Current estimate of ``σ_max(W̄)`` (≤ s by construction)."""
        return torch.minimum(self._sigma_ema, torch.tensor(self.s))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"s={self.s}, bias={self.bias is not None}"
        )
