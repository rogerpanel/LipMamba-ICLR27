"""Optimiser / scheduler factories.

Defaults match the paper:

* AdamW (lr=2e-4, weight_decay=0.1)
* Cosine annealing across ``max_steps`` after a short linear warm-up.
* Gradient clipping at norm 1.0 (handled by the Trainer).
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    lr: float = 2e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """Create an AdamW optimiser configured per the LipMamba paper."""
    return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


class CosineWithWarmup(LRScheduler):
    """Linear warm-up → cosine decay to a minimum lr."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = max(1, int(warmup_steps))
        self.max_steps = max(self.warmup_steps + 1, int(max_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        step = max(0, self.last_epoch)
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * (step + 1) / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / max(1, (self.max_steps - self.warmup_steps))
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                lr = base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)
            lrs.append(lr)
        return lrs


def build_scheduler(
    optimizer: Optimizer,
    warmup_steps: int = 1_000,
    max_steps: int = 100_000,
    min_lr_ratio: float = 0.1,
) -> CosineWithWarmup:
    return CosineWithWarmup(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr_ratio=min_lr_ratio,
    )
