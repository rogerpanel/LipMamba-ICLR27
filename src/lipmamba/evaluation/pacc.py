"""PACC — Poisoning-Attack Clean Correctness.

Defined in Section 5.1 of the paper as the fraction of inputs that:
  (i)  are classified correctly on the clean prefix, AND
  (ii) remain correctly classified after the worst-case trigger from RoBench.

This is the *empirical* counterpart to the certified-accuracy metric.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from ..attacks.hispa import HiSPAAttack, HiSPAConfig


@torch.no_grad()
def _classify(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    out = model(ids)
    logits = out.get("cls_logits") or out["lm_logits"][:, -1]
    return logits.argmax(dim=-1)


def poisoning_attack_clean_correctness(
    model: nn.Module,
    prefixes: torch.Tensor,
    targets: torch.Tensor,
    triggers: Iterable[torch.Tensor],
) -> float:
    """Empirical PACC over a set of pre-computed triggers."""
    clean_pred = _classify(model, prefixes)
    correct_clean = clean_pred == targets

    best_correct = correct_clean.clone()
    for trig in triggers:
        if trig.dim() == 1:
            trig = trig.unsqueeze(0).expand(prefixes.size(0), -1)
        ids = torch.cat([prefixes, trig.to(prefixes.device)], dim=1)
        pred = _classify(model, ids)
        best_correct = best_correct & (pred == targets)
    return float(best_correct.float().mean().item())


def pacc_with_hispa(
    model: nn.Module,
    prefixes: torch.Tensor,
    targets: torch.Tensor,
    cfg: HiSPAConfig,
    n_attacks: int = 4,
) -> float:
    """Adaptive PACC: re-run HiSPA `n_attacks` times and aggregate worst-case."""
    attacker = HiSPAAttack(model, cfg)
    correct_under_attack = torch.ones_like(targets, dtype=torch.bool)
    for _ in range(n_attacks):
        delta, _info = attacker.attack(prefixes)
        embeds = torch.cat([model.embed_tokens(prefixes).detach(), delta], dim=1)
        h = embeds
        for blk in model.blocks:
            h = blk(h)
        h = model.norm_f(h)
        logits = (
            model.cls_head(h[:, -1]) if model.cls_head is not None else model.lm_head(h[:, -1])
        )
        pred = logits.argmax(dim=-1)
        correct_under_attack &= pred == targets
    return float(correct_under_attack.float().mean().item())
