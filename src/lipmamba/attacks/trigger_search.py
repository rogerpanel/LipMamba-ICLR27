"""Greedy discrete trigger search for HiSPA / GCG-style attacks.

A budget-constrained, batched, top-k coordinate descent over discrete tokens.
Each step picks the position whose substitution yields the largest drop in
``‖h_T‖₂``.  This implementation is deliberately compact: it favours
clarity over the optimised batched-evaluation tricks of GCG.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GreedySearchConfig:
    trigger_length: int = 16
    n_iters: int = 100
    candidates_per_step: int = 64
    top_k: int = 256


class GreedyDiscreteTriggerSearch:
    """Token-level greedy hidden-state poisoning."""

    def __init__(self, model: nn.Module, vocab_size: int, cfg: GreedySearchConfig) -> None:
        self.model = model
        self.vocab_size = vocab_size
        self.cfg = cfg

    @torch.no_grad()
    def _hidden_norm(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.model.encode(ids)
        return h[:, -1].norm(dim=-1)

    def search(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """Return ``(B, trigger_length)`` token ids minimising ``‖h_T‖₂``."""
        cfg = self.cfg
        device = prefix_ids.device
        b = prefix_ids.size(0)

        trigger = torch.randint(
            0, self.vocab_size, (b, cfg.trigger_length), device=device, dtype=prefix_ids.dtype
        )

        for _ in range(cfg.n_iters):
            for pos in range(cfg.trigger_length):
                base = torch.cat([prefix_ids, trigger], dim=1)
                base_norm = self._hidden_norm(base)
                # propose `candidates_per_step` random replacements per row
                candidates = torch.randint(
                    0, self.vocab_size, (b, cfg.candidates_per_step), device=device,
                    dtype=prefix_ids.dtype,
                )
                best = trigger[:, pos].clone()
                best_norm = base_norm.clone()
                for c in range(cfg.candidates_per_step):
                    swapped = trigger.clone()
                    swapped[:, pos] = candidates[:, c]
                    full = torch.cat([prefix_ids, swapped], dim=1)
                    h_norm = self._hidden_norm(full)
                    mask = h_norm < best_norm
                    best = torch.where(mask, candidates[:, c], best)
                    best_norm = torch.where(mask, h_norm, best_norm)
                trigger[:, pos] = best
        return trigger
