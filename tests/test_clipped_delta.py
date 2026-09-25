"""Two-sided clamp invariants (Eq. 2)."""
from __future__ import annotations

import torch

from lipmamba.models.clipped_delta import ClippedDelta


def test_delta_within_two_sided_bounds() -> None:
    layer = ClippedDelta(d_model=16, d_inner=16, delta_min=1e-3, delta_max=0.5, s_delta=0.5).eval()
    x = torch.randn(4, 32, 16) * 50.0            # extreme inputs
    d = layer(x)
    assert (d >= 1e-3 - 1e-7).all(), "lower clamp violated"
    assert (d < 0.5 + 1e-6).all(), "upper clamp violated"


def test_delta_is_one_lipschitz_in_preactivation() -> None:
    dmin, dmax = 1e-3, 0.5
    z = torch.linspace(-10, 10, 20001, requires_grad=True)
    d = dmin + (dmax - dmin) * torch.tanh(torch.nn.functional.softplus(z) / (dmax - dmin))
    g, = torch.autograd.grad(d.sum(), z)
    assert float(g.abs().max()) <= 1.0 + 1e-5
