"""Two-sided clamped discretisation step Δ_t (Eq. 2 of the ICLR manuscript).

    Δ_t = Δ_min + (Δ_max − Δ_min) · tanh( softplus(W̄_Δ x_t + τ) / (Δ_max − Δ_min) )

so that Δ_t ∈ [Δ_min, Δ_max).  Both ends matter:

* the *upper* clamp stops the HiSPA mechanism (Δ_t → ∞ ⇒ Ā_t → 0);
* the *lower* clamp Δ_min > 0 is what makes the state bounded
  (Lemma "Bounded state": ‖h_t‖ ≤ H = c / (1 − ρ_max) with
  ρ_max = exp(−Δ_min λ_min) < 1).  With Δ_min = 0 the geometric series
  diverges, which is the error-explosion regime of Qi et al. (NeurIPS 2024).

The map is smooth and 1-Lipschitz in its pre-activation (tanh′ ≤ 1,
softplus′ ≤ 1), which the Theorem-1 proof uses to get
|Δ_t − Δ′_t| ≤ s_Δ ‖x_t − x′_t‖.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_norm import SpectralNormLinear


class ClippedDelta(nn.Module):
    """Spectrally-bounded, two-sided clamped Δ_t projection."""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        delta_min: float = 1e-3,
        delta_max: float = 0.5,
        s_delta: float = 0.5,
        n_power_iters: int = 1,
        tau_init: float = 0.0,
    ) -> None:
        super().__init__()
        if not (0.0 < delta_min < delta_max):
            raise ValueError("require 0 < delta_min < delta_max")
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.s_delta = float(s_delta)
        self.proj = SpectralNormLinear(
            d_model, d_inner, s=s_delta, bias=True, n_power_iters=n_power_iters
        )
        nn.init.constant_(self.proj.bias, tau_init)

    @property
    def span(self) -> float:
        return self.delta_max - self.delta_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Δ_t ∈ [Δ_min, Δ_max) from ``x`` of shape ``(B, T, d_model)``."""
        z = self.proj(x)
        return self.delta_min + self.span * torch.tanh(F.softplus(z) / self.span)

    def extra_repr(self) -> str:
        return f"delta_min={self.delta_min}, delta_max={self.delta_max}, s_delta={self.s_delta}"


def vanilla_softplus_delta(z: torch.Tensor) -> torch.Tensor:
    """Unconstrained Mamba step (ablation "w/o step clamp")."""
    return F.softplus(z)
