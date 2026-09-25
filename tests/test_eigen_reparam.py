"""Eigenvalue reparameterisation invariants."""
from __future__ import annotations

import torch

from lipmamba.models.eigen_reparam import EigenReparamA


def test_eigenvalues_within_bounds() -> None:
    torch.manual_seed(0)
    layer = EigenReparamA(state_dim=8, n_channels=4, lambda_min=0.05, lambda_max=1.0)
    a = layer()
    # all eigenvalues are negative
    assert (a < 0).all()
    abs_a = a.abs()
    assert (abs_a >= 0.05 - 1e-6).all()
    assert (abs_a <= 1.0 + 1e-6).all()


def test_discretise_shape_and_contraction() -> None:
    layer = EigenReparamA(state_dim=4, n_channels=3, lambda_min=0.05, lambda_max=1.0)
    delta = torch.full((2, 5, 3), 0.5)
    a_bar = layer.discretise(delta)
    assert a_bar.shape == (2, 5, 3, 4)
    # Each diagonal entry of Ā should be in (0, 1).
    assert (a_bar > 0).all()
    assert (a_bar < 1).all()
