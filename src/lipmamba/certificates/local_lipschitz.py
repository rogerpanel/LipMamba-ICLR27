"""Local Lipschitz estimator (Appendix E of the ICLR manuscript).

    L_loc(x) = max_{k≠ŷ} max_{x' ∈ B(x, r)} ‖∇_{x'} (z_ŷ − z_k)(x')‖₂ / √2

found by 20 steps of projected gradient ascent from 8 random starts, r = 0.3
in embedding space.  It is a *lower* estimate of the true local constant,
so radii computed with it are *empirical local-Lipschitz radii* ("LL-Acc"),
not deterministic certificates (Remark 4).

The implementation maximises over the runner-up class jointly with the ball
(the runner-up may change inside the ball; we track the max over all k ≠ ŷ).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LocalLipschitzConfig:
    radius: float = 0.3
    n_steps: int = 20
    n_restarts: int = 8
    step_size: float | None = None     # default r/4
    all_runner_ups: bool = True        # max over every k≠ŷ (paper) vs. only the current runner-up


def local_lipschitz_estimate(
    model: nn.Module,
    emb: torch.Tensor,
    cfg: LocalLipschitzConfig = LocalLipschitzConfig(),
) -> torch.Tensor:
    """Return L_loc(x) for each sample in ``emb`` (B, T, d)."""
    model.eval()
    b = emb.size(0)
    device = emb.device
    step = cfg.step_size if cfg.step_size is not None else cfg.radius / 4
    best = torch.zeros(b, device=device)

    with torch.no_grad():
        y_hat = model.logits_from_embeddings(emb).argmax(dim=-1)

    for _ in range(cfg.n_restarts):
        delta = torch.randn_like(emb)
        delta = delta / delta.flatten(1).norm(dim=-1).view(-1, 1, 1) * cfg.radius * torch.rand(b, 1, 1, device=device)
        for _ in range(cfg.n_steps):
            delta.requires_grad_(True)
            x = emb + delta
            logits = model.logits_from_embeddings(x)
            k_classes = logits.size(-1)
            z_hat = logits.gather(1, y_hat.unsqueeze(-1)).squeeze(-1)
            if cfg.all_runner_ups:
                ks = [k for k in range(k_classes)]
            else:
                masked = logits.clone(); masked.scatter_(1, y_hat.unsqueeze(-1), float("-inf"))
                ks = [None]
            gnorm_max = torch.zeros(b, device=device)
            for k in ks:
                if k is None:
                    z_k = masked.max(dim=-1).values
                else:
                    z_k = logits[:, k]
                    valid = (y_hat != k)
                    if not valid.any():
                        continue
                g = (z_hat - z_k).sum()
                grad_x, = torch.autograd.grad(g, x, retain_graph=True, create_graph=True)
                gn = grad_x.flatten(1).norm(dim=-1)
                if k is not None:
                    gn = torch.where(valid, gn, torch.zeros_like(gn))
                gnorm_max = torch.maximum(gnorm_max, gn)
            with torch.no_grad():
                best = torch.maximum(best, gnorm_max.detach())
            ascent, = torch.autograd.grad(gnorm_max.sum(), delta)
            with torch.no_grad():
                delta = delta.detach() + step * ascent / (ascent.flatten(1).norm(dim=-1).view(-1, 1, 1) + 1e-12)
                dn = delta.flatten(1).norm(dim=-1).view(-1, 1, 1)
                delta = delta * torch.clamp(cfg.radius / (dn + 1e-12), max=1.0)
    return best.detach() / math.sqrt(2.0)


def ll_radius(logits: torch.Tensor, l_loc: torch.Tensor) -> torch.Tensor:
    """ε*(x) = margin / (√2 · L_loc(x)) — the GloRo radius with L := L_loc.

    Consistency check: the margin m = z_ŷ − z_k has Lipschitz constant
    max‖∇m‖ = √2·L_loc by the App.-E definition, and ε* = m / Lip(m)
    = m / (√2 L_loc), which is exactly the GloRo formula with L = L_loc."""
    z_hat, hat_idx = logits.max(dim=-1)
    masked = logits.clone(); masked.scatter_(1, hat_idx.unsqueeze(-1), float("-inf"))
    margin = z_hat - masked.max(dim=-1).values
    return margin / (math.sqrt(2.0) * l_loc + 1e-12)
