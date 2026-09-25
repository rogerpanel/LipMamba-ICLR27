"""Per-input certified radius and certified accuracy."""
from __future__ import annotations

import math

import torch


def certified_radius_batch(
    logits: torch.Tensor,
    l_net: torch.Tensor | float,
) -> torch.Tensor:
    """Per-sample GloroNet certified radius.

    ε*(x) = ( z_{ŷ} − max_{k≠ŷ} z_k ) / ( √2 · L_net ).
    """
    z_hat, hat_idx = logits.max(dim=-1)
    masked = logits.clone()
    masked.scatter_(1, hat_idx.unsqueeze(-1), float("-inf"))
    z_runner = masked.max(dim=-1).values
    margin = z_hat - z_runner
    l = torch.as_tensor(l_net, dtype=logits.dtype, device=logits.device)
    return margin / (math.sqrt(2.0) * (l + 1e-12))


def certified_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    l_net: torch.Tensor | float,
    radius: float,
) -> torch.Tensor:
    """Fraction of samples that are *both* correctly classified and have
    certified radius ≥ ``radius``.
    """
    pred = logits.argmax(dim=-1)
    correct = pred == targets
    eps = certified_radius_batch(logits, l_net)
    return (correct & (eps >= radius)).float().mean()


def certified_curve(
    logits: torch.Tensor,
    targets: torch.Tensor,
    l_net: torch.Tensor | float,
    radii: list[float],
) -> dict[float, float]:
    """Return certified-accuracy curve over a grid of radii."""
    return {r: float(certified_accuracy(logits, targets, l_net, r).item()) for r in radii}
