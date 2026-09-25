"""Clean accuracy on a held-out test set."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def clean_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """Return top-1 accuracy on the loader."""
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"])
        logits = out.get("cls_logits") or out["lm_logits"][:, -1]
        pred = logits.argmax(dim=-1)
        target = batch["labels"]
        if target.dim() == 2:
            target = target[:, -1]
        correct += int((pred == target).sum().item())
        total += int(target.size(0))
    return correct / max(1, total)
