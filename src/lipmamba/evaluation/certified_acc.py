"""Certified accuracy + per-input certified-radius distribution."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..certificates.certified_radius import (
    certified_accuracy,
    certified_curve,
    certified_radius_batch,
)
from ..certificates.lipschitz import empirical_network_lipschitz


@torch.no_grad()
def certified_eval(
    model: nn.Module,
    loader: DataLoader,
    radii: list[float] | None = None,
) -> dict[str, object]:
    """Compute certified accuracy at every requested radius.

    Returns a dict with keys ``L_net``, ``radii_curve`` (mapping
    ``radius -> certified_acc``) and ``mean_radius``.
    """
    model.eval()
    radii = radii or [0.04, 0.08, 0.12, 0.18, 0.24]
    device = next(model.parameters()).device
    l_net = empirical_network_lipschitz(model)

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"])
        logits = out.get("cls_logits") or out["lm_logits"][:, -1]
        target = batch["labels"]
        if target.dim() == 2:
            target = target[:, -1]
        all_logits.append(logits)
        all_targets.append(target)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    eps = certified_radius_batch(logits, l_net)
    return {
        "L_net": float(l_net),
        "mean_radius": float(eps.mean().item()),
        "median_radius": float(eps.median().item()),
        "radii_curve": certified_curve(logits, targets, l_net, radii),
        "certified_at_018": float(certified_accuracy(logits, targets, l_net, 0.18).item()),
    }
