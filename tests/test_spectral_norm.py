"""Spectral-norm Lipschitz invariant tests."""
from __future__ import annotations

import torch

from lipmamba.models.spectral_norm import SpectralNormLinear, power_iteration_sigma


def test_power_iteration_recovers_singular_value() -> None:
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    expected = torch.linalg.svdvals(w)[0]
    u = torch.nn.functional.normalize(torch.randn(32), dim=0)
    sigma, _ = power_iteration_sigma(w, u, n_iters=200)
    assert torch.isclose(sigma, expected, rtol=5e-3)


def test_spectral_norm_linear_bound() -> None:
    torch.manual_seed(0)
    layer = SpectralNormLinear(64, 32, s=2.0)
    layer.train()
    x = torch.randn(16, 64)
    layer(x)  # populate σ̂
    layer.eval()
    sigma = float(layer.sigma.item())
    assert sigma <= 2.0 + 1e-3
