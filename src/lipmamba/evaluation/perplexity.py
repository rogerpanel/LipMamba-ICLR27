"""Language-modelling perplexity."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def perplexity(model: nn.Module, loader: DataLoader) -> float:
    """Standard token-level perplexity."""
    model.eval()
    device = next(model.parameters()).device
    nll_sum = 0.0
    n_tok = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"])
        logits = out["lm_logits"]
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch["labels"].reshape(-1),
            reduction="sum",
        )
        nll_sum += float(nll.item())
        n_tok += int(batch["labels"].numel())
    return math.exp(nll_sum / max(1, n_tok))
