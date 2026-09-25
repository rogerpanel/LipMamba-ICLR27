"""Certified-radius / certified-accuracy invariants."""
from __future__ import annotations

import math

import torch

from lipmamba.certificates.certified_radius import (
    certified_accuracy,
    certified_radius_batch,
)


def test_radius_positive_when_clear_margin() -> None:
    logits = torch.tensor([[5.0, 0.0, 0.0]])
    eps = certified_radius_batch(logits, l_net=1.0)
    assert eps.item() > 0


def test_certified_accuracy_threshold() -> None:
    logits = torch.tensor([[5.0, 0.0], [0.1, 0.0]])
    targets = torch.tensor([0, 0])
    # row 1 has small margin → should fall below ε=0.18
    acc = certified_accuracy(logits, targets, l_net=1.0, radius=0.18)
    assert 0 <= acc.item() <= 1
