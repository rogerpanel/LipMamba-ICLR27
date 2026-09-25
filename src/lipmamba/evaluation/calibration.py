"""Expected calibration error (Guo et al., ICML 2017) with 15 equal-width bins."""
from __future__ import annotations

import torch


@torch.no_grad()
def expected_calibration_error(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    probs = logits.softmax(-1)
    conf, pred = probs.max(-1)
    correct = (pred == targets).float()
    edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return float(ece)
