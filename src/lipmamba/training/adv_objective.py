"""Empirical adversarial cross-entropy loss.

Used by :func:`pac_objective.pac_bayes_total_loss` and the trainer.  Two
flavours are exposed:

* ``adversarial_loss(model, batch, attack)`` — runs an attacker (PGD or
  HiSPA) and computes the loss on the perturbed inputs.
* ``margin_adversarial_loss(model, batch, l_net, epsilon)`` — uses the
  margin-augmented logits from :class:`models.glorot_head.GloroNetHead`,
  which provably upper-bounds the adversarial loss (Section 4.2).
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def adversarial_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    attack_fn: Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Cross-entropy on either clean or adversarial inputs.

    ``attack_fn`` is expected to return adversarial *embeddings* of the same
    shape as ``model.embed_tokens(input_ids)``.  When ``None`` we fall back
    to clean training.
    """
    input_ids = batch["input_ids"]
    targets = batch["labels"]

    if attack_fn is None:
        out = model(input_ids)
    else:
        adv_embed = attack_fn(model, batch)
        h = adv_embed
        for blk in model.blocks:
            h = blk(h)
        h = model.norm_f(h)
        if model.cls_head is not None:
            logits = model.cls_head(h[:, -1])
        else:
            logits = model.lm_head(h)
        out = {"cls_logits": logits} if logits.dim() == 2 else {"lm_logits": logits}

    if "cls_logits" in out:
        return F.cross_entropy(out["cls_logits"], targets)
    # Language-modelling: shift labels expected by caller.
    return F.cross_entropy(
        out["lm_logits"].reshape(-1, out["lm_logits"].size(-1)),
        targets.reshape(-1),
    )


def margin_adversarial_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    l_net: torch.Tensor | float,
    epsilon: float,
) -> torch.Tensor:
    """Closed-form upper bound on the adversarial loss via margin augmentation."""
    out = model(batch["input_ids"])
    if "cls_logits" not in out:
        raise RuntimeError("margin_adversarial_loss requires a classification head")
    logits = out["cls_logits"]
    if model.cls_head is None:
        raise RuntimeError("model has no GloroNet classification head")
    return model.cls_head.margin_loss(logits, batch["labels"], l=l_net, epsilon=epsilon)
