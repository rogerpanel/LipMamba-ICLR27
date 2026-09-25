"""Projected gradient descent on input embeddings.

Used both as an *evaluation* attack (compute empirical robust accuracy) and
as the *training* attack inside the adversarial PAC-Bayes objective.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PGDConfig:
    """Standard PGD hyper-parameters used in the paper."""

    epsilon: float = 0.18      # ℓ₂ ball radius
    step_size: float | None = None  # default ε/4
    n_steps: int = 20          # ablations span {10, 20, 40}


class PGDAttack:
    """ℓ₂ PGD on the embedding tensor of a classification head."""

    def __init__(self, model: nn.Module, cfg: PGDConfig) -> None:
        self.model = model
        self.cfg = cfg

    @torch.enable_grad()
    def attack(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return adversarial embeddings of the same shape as ``E[x]``."""
        cfg = self.cfg
        step = cfg.step_size if cfg.step_size is not None else cfg.epsilon / 4

        emb = self.model.embed_tokens(input_ids).detach()
        delta = torch.zeros_like(emb)
        delta.requires_grad_(True)
        for _ in range(cfg.n_steps):
            h = emb + delta
            for blk in self.model.blocks:
                h = blk(h)
            h = self.model.norm_f(h)
            logits = self.model.cls_head(h[:, -1]) if self.model.cls_head else self.model.lm_head(h[:, -1])
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            with torch.no_grad():
                grad = delta.grad
                grad_norm = grad.flatten(1).norm(dim=-1, keepdim=True).clamp_min(1e-12)
                delta.add_(step * grad / grad_norm.unsqueeze(-1))
                # project back to ε-ball
                d_norm = delta.flatten(1).norm(dim=-1, keepdim=True)
                scale = (cfg.epsilon / (d_norm + 1e-12)).clamp(max=1.0)
                delta.mul_(scale.unsqueeze(-1))
                delta.grad.zero_()
        return (emb + delta).detach()
