"""Top-level benchmark runner — assembles the per-task evaluators."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .certified_acc import certified_eval
from .clean_acc import clean_accuracy
from .pacc import poisoning_attack_clean_correctness
from .perplexity import perplexity


@dataclass
class BenchmarkResult:
    name: str
    clean_acc: float | None = None
    perplexity: float | None = None
    certified: dict[str, Any] = field(default_factory=dict)
    pacc: float | None = None


class BenchmarkRunner:
    """Run any combination of {clean acc, PPL, certified acc, PACC} on a model.

    The runner is intentionally stateless except for the model reference; it
    orchestrates the per-task helpers and prints a summary table.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def run(
        self,
        *,
        name: str,
        cls_loader: DataLoader | None = None,
        lm_loader: DataLoader | None = None,
        radii: list[float] | None = None,
        pacc_inputs: tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]] | None = None,
    ) -> BenchmarkResult:
        out = BenchmarkResult(name=name)
        if cls_loader is not None:
            out.clean_acc = clean_accuracy(self.model, cls_loader)
            out.certified = certified_eval(self.model, cls_loader, radii=radii)
        if lm_loader is not None:
            out.perplexity = perplexity(self.model, lm_loader)
        if pacc_inputs is not None:
            prefixes, targets, triggers = pacc_inputs
            out.pacc = poisoning_attack_clean_correctness(
                self.model, prefixes, targets, triggers
            )
        return out
