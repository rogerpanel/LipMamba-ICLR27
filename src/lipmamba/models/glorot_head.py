"""GloRo-style certification head (Leino et al., ICML 2021).

    ε*(x) = ( z_ŷ − max_{k≠ŷ} z_k ) / ( √2 · L )
    z̃_K  = max_{k≠ŷ} z_k + √2 · L · ε_train          (training augmentation)

Which ``L`` is passed in is the subject of Remark 4 of the manuscript: the
*global* Theorem-1 product is vacuous at depth, so the reported radii use the
local estimate L_loc (certificates/local_lipschitz.py) and are labelled
"LL-Acc" (empirical local-Lipschitz radii, not deterministic certificates).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_norm import SpectralNormLinear


def _top2(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_hat, hat_idx = logits.max(dim=-1)
    masked = logits.clone()
    masked.scatter_(1, hat_idx.unsqueeze(-1), float("-inf"))
    z_runner, runner_idx = masked.max(dim=-1)
    return z_hat, z_runner, runner_idx


class GloroNetHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_classes: int,
        s_head: float = 1.0,
        epsilon_train: float = 0.18,
        spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.s_head = float(s_head)
        self.epsilon_train = float(epsilon_train)
        self.classifier = (
            SpectralNormLinear(d_model, n_classes, s=s_head) if spectral_norm
            else nn.Linear(d_model, n_classes)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    @staticmethod
    def certified_radius(logits: torch.Tensor, l: torch.Tensor | float) -> torch.Tensor:
        z_hat, z_runner, _ = _top2(logits)
        l = torch.as_tensor(l, dtype=logits.dtype, device=logits.device)
        return (z_hat - z_runner) / (math.sqrt(2.0) * (l + 1e-12))

    def margin_augmented(
        self, logits: torch.Tensor, l: torch.Tensor | float, epsilon: float | None = None
    ) -> torch.Tensor:
        """Append the GloRo "⊥" logit z̃_K = max_{k≠ŷ} z_k + √2 L ε as an extra class."""
        eps = self.epsilon_train if epsilon is None else float(epsilon)
        _, z_runner, _ = _top2(logits)
        l = torch.as_tensor(l, dtype=logits.dtype, device=logits.device)
        if l.dim() == 0:
            l = l.expand(logits.size(0))
        bump = math.sqrt(2.0) * l * eps
        return torch.cat([logits, (z_runner + bump).unsqueeze(-1)], dim=-1)

    def margin_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, l: torch.Tensor | float,
        epsilon: float | None = None,
    ) -> torch.Tensor:
        return F.cross_entropy(self.margin_augmented(logits, l, epsilon), targets)
