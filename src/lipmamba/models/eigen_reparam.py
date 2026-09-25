"""Eigenvalue reparameterisation for the state matrix ``A``.

    A = -diag( λ_min + (λ_max − λ_min) · σ(α) )

so that λ_i(A) ∈ [-λ_max, -λ_min] throughout training.  Because A is real
diagonal, Ā_t = exp(Δ_t A) is diagonal with entries e^{Δ_t a_i} ∈ (0, 1), and
its spectral radius, largest and smallest singular values coincide with
max_i e^{Δ_t a_i} and min_i e^{Δ_t a_i} (Section 3 of the manuscript).  The
retention bound uses the *smallest* singular value; the Lipschitz bound the
largest.  The ``free`` flag reproduces the ablation "w/o eigenvalue
reparameterisation": A = -exp(a_log) as in vanilla Mamba, unbounded.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EigenReparamA(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_channels: int = 1,
        lambda_min: float = 0.05,
        lambda_max: float = 1.0,
        free: bool = False,
    ) -> None:
        super().__init__()
        if not (0.0 < lambda_min <= lambda_max):
            raise ValueError("require 0 < lambda_min <= lambda_max")
        self.state_dim = state_dim
        self.n_channels = n_channels
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.free = bool(free)
        self.alpha = nn.Parameter(torch.zeros(n_channels, state_dim))

    @property
    def lambdas(self) -> torch.Tensor:
        """|λ_i(A)| per channel and state."""
        if self.free:
            return torch.exp(self.alpha)          # vanilla Mamba: A = -exp(a_log)
        return self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(self.alpha)

    def forward(self) -> torch.Tensor:
        return -self.lambdas

    def discretise(self, delta: torch.Tensor) -> torch.Tensor:
        """Ā = exp(Δ ⊙ A): delta (B,T,D) → (B,T,D,N)."""
        a = self.forward()
        return torch.exp(delta.unsqueeze(-1) * a.unsqueeze(0).unsqueeze(0))

    def extra_repr(self) -> str:
        return (f"state_dim={self.state_dim}, n_channels={self.n_channels}, "
                f"lambda_min={self.lambda_min}, lambda_max={self.lambda_max}, free={self.free}")
