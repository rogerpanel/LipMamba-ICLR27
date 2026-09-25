"""Hidden-state poisoning lower-bound invariants."""
from __future__ import annotations

import math

from lipmamba.certificates.poisoning_immunity import (
    max_certified_trigger_length,
    poisoning_immunity_lower_bound,
)


def test_lower_bound_decays_with_length() -> None:
    bounds = [
        poisoning_immunity_lower_bound(rho_min=0.95, h0_norm=1.0, b_bar_max=0.1, x_max=1.0, ell=ell)
        for ell in (0, 4, 8, 16, 32)
    ]
    assert bounds[0] >= bounds[1] >= bounds[2] >= bounds[3] >= bounds[4]


def test_max_trigger_length_is_finite() -> None:
    ell_star = max_certified_trigger_length(
        rho_min=0.95, h0_norm=1.0, b_bar_max=0.1, x_max=1.0, alpha=0.05
    )
    assert isinstance(ell_star, int) and 0 <= ell_star <= 1024
