"""Lipschitz constants: worst-case (Theorem 1), data-dependent (Algorithm 1)
and empirical lower estimates (Figure 2).

Three quantities, deliberately kept apart because the audit found them
conflated:

1. **Worst-case analytical** — ``ConstraintSet.l_block`` / ``l_network``:
   a valid upper bound on the bounded-input domain, input-independent, and
   (Remark 4) astronomically large at depth.
2. **Data-dependent analytical** — Algorithm 1's D_t accumulator using the
   observed ‖Ā_t‖₂ and ‖h_{t-1}‖₂ on a specific input; still an upper bound
   *for that input*, usually orders of magnitude below (1).
3. **Empirical lower estimate** — the largest gradient norm found by attack
   (:func:`empirical_lipschitz_lower_bound`); a *lower* bound on the true
   constant.  It must lie below (1) and (2) by a wide margin; if it tracks
   the analytical curve "within 2%" the plot is wrong (TODO 5 of the
   manuscript).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .constants import L_SILU, ConstraintSet

__all__ = [
    "L_SILU", "ConstraintSet", "layer_lipschitz_bound", "network_lipschitz",
    "LipschitzTracker", "empirical_network_lipschitz", "worst_case_curve_vs_depth",
    "empirical_lipschitz_lower_bound", "operator_norm_product",
]


def layer_lipschitz_bound(**kw) -> float:
    """Theorem-1 constant from keyword constraint values (see ConstraintSet)."""
    return ConstraintSet(**kw).l_block


def network_lipschitz(block_bounds: list[float], head_factor: float = 1.0,
                      use_residual_inflation: bool = False) -> float:
    f = 1.0
    for l in block_bounds:
        f *= (1.0 + l) if use_residual_inflation else l
    return head_factor * f


class LipschitzTracker:
    """Streaming version of Algorithm 1's D_t recurrence for one channel."""

    def __init__(self, cs: ConstraintSet) -> None:
        self.cs = cs
        self.D = 0.0
        self.h_max = 0.0

    def reset(self) -> None:
        self.D = 0.0
        self.h_max = 0.0

    def update(self, a_bar_norm: float, delta_t: float, h_prev_norm: float, h_norm: float) -> float:
        cs = self.cs
        gamma_t = 2 * cs.s_b * delta_t * cs.x_max + cs.s_delta * (cs.lambda_max * h_prev_norm + cs.s_b * cs.x_max**2)
        self.D = a_bar_norm * self.D + gamma_t
        self.h_max = max(self.h_max, h_norm)
        return self.D

    def block_bound(self) -> float:
        cs = self.cs
        return cs.s_out * cs.l_silu * cs.s_c * (cs.x_max * self.D + self.h_max)


@torch.no_grad()
def empirical_network_lipschitz(model: nn.Module, **_ignored) -> float:
    """Worst-case Theorem-1 product for any module exposing ``constraints``."""
    if hasattr(model, "network_lipschitz_bound"):
        return float(model.network_lipschitz_bound().item())
    bounds = [float(m.block_lipschitz_bound().item()) for m in model.modules()
              if hasattr(m, "block_lipschitz_bound") and m is not model]
    return network_lipschitz(bounds)


def worst_case_curve_vs_depth(cs: ConstraintSet, depths=(4, 8, 16, 24), residual: bool = False) -> dict[int, float]:
    """log10 of the Theorem-1 product at each depth (Figure 2, analytical curve)."""
    per = math.log10((1.0 + cs.l_block) if residual else cs.l_block)
    return {d: d * per for d in depths}


@torch.no_grad()
def operator_norm_product(model: nn.Module) -> float:
    """Bare product of ‖W̄_•‖₂ over all spectrally-normalised layers (Figure 2, dotted curve)."""
    from ..models.spectral_norm import SpectralNormLinear
    logp = 0.0
    for m in model.modules():
        if isinstance(m, SpectralNormLinear):
            logp += math.log10(max(float(m.sigma), 1e-12))
    return 10.0**logp


def empirical_lipschitz_lower_bound(
    model: nn.Module,
    emb: torch.Tensor,
    n_steps: int = 20,
    n_restarts: int = 8,
    radius: float = 0.3,
    step_size: float | None = None,
    output: str = "margin",
) -> torch.Tensor:
    """Attack-based *lower* estimate of the local Lipschitz constant at ``emb``.

    Maximises ‖∇_{x'} g(x')‖₂ over the ball of radius ``radius`` around ``emb``
    by projected gradient ascent (Appendix E of the manuscript), where
    ``g`` is the top-1 margin (``output="margin"``) or the full logit vector
    Jacobian norm proxy (``output="logits"``: ‖J^T v‖ for a random unit v).

    Returns the per-sample estimate (B,).  Divides by √2 for the margin case
    so it is directly comparable to the GloRo radius denominator.
    """
    model.eval()
    device = emb.device
    b = emb.size(0)
    step = step_size if step_size is not None else radius / 4
    best = torch.zeros(b, device=device)

    for _ in range(n_restarts):
        delta = torch.randn_like(emb)
        delta = delta / delta.flatten(1).norm(dim=-1).view(-1, 1, 1) * radius * torch.rand(b, 1, 1, device=device)
        delta.requires_grad_(True)
        for _ in range(n_steps):
            x = emb + delta
            x.requires_grad_(True)
            logits = model.logits_from_embeddings(x)
            if output == "margin":
                z_hat, hat_idx = logits.max(dim=-1)
                masked = logits.clone(); masked.scatter_(1, hat_idx.unsqueeze(-1), float("-inf"))
                g = (z_hat - masked.max(dim=-1).values)
            else:
                v = torch.randn_like(logits); v = v / v.norm(dim=-1, keepdim=True)
                g = (logits * v).sum(-1)
            grad_x, = torch.autograd.grad(g.sum(), x, create_graph=True)
            gnorm = grad_x.flatten(1).norm(dim=-1)                 # ‖∇g‖ per sample
            with torch.no_grad():
                best = torch.maximum(best, gnorm.detach())
            # ascend on the gradient norm
            ggrad, = torch.autograd.grad(gnorm.sum(), delta)
            with torch.no_grad():
                delta += step * ggrad / (ggrad.flatten(1).norm(dim=-1).view(-1, 1, 1) + 1e-12)
                dn = delta.flatten(1).norm(dim=-1).view(-1, 1, 1)
                delta *= torch.clamp(radius / (dn + 1e-12), max=1.0)
    if output == "margin":
        best = best / math.sqrt(2.0)
    return best.detach()
