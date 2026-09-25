"""Bounded-input projection (Assumption 1 of the ICLR manuscript).

Theorem 1 and the retention bound hold on the *bounded-input domain*
‖x_t‖₂ ≤ X_max.  Rather than assume it, we enforce it by construction with
the Euclidean projection onto the X_max-ball,

    Π(x) = x · min(1, X_max / ‖x‖₂),

which is 1-Lipschitz (projection onto a convex set) and therefore does not
change any Lipschitz constant downstream.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InputNormClip(nn.Module):
    """Project every token vector onto the ball of radius ``x_max``."""

    def __init__(self, x_max: float = 1.0, eps: float = 1e-12) -> None:
        super().__init__()
        self.x_max = float(x_max)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.x_max / (norm + self.eps), max=1.0)
        return x * scale

    def extra_repr(self) -> str:
        return f"x_max={self.x_max}"
