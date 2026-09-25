"""Lipschitz tracker / closed-form bound (corrected Theorem 1)."""
from __future__ import annotations

import math

from lipmamba.certificates.constants import ConstraintSet
from lipmamba.certificates.lipschitz import (LipschitzTracker, layer_lipschitz_bound,
                                             network_lipschitz, worst_case_curve_vs_depth)


def test_layer_bound_matches_constraint_set() -> None:
    kw = dict(s_b=1.0, s_c=1.0, s_delta=0.5, s_out=1.0, delta_min=1e-3, delta_max=0.5,
              lambda_min=0.05, lambda_max=1.0, x_max=1.0)
    assert math.isclose(layer_lipschitz_bound(**kw), ConstraintSet(**kw).l_block)


def test_network_product_vs_residual_inflation() -> None:
    b = 7.0
    assert math.isclose(network_lipschitz([b] * 4), b**4)
    assert math.isclose(network_lipschitz([b] * 4, use_residual_inflation=True), (1 + b) ** 4)


def test_depth_curve_is_linear_in_log10() -> None:
    cs = ConstraintSet(delta_min=0.5, delta_max=1.0, lambda_min=1.0)     # per-block ≈ 7.4
    curve = worst_case_curve_vs_depth(cs, depths=(4, 8, 16, 24))
    per = math.log10(cs.l_block)
    for d, v in curve.items():
        assert math.isclose(v, d * per)


def test_tracker_never_exceeds_worst_case() -> None:
    cs = ConstraintSet(delta_min=0.5, delta_max=1.0, lambda_min=1.0)
    tr = LipschitzTracker(cs)
    for _ in range(200):
        tr.update(a_bar_norm=cs.rho_max, delta_t=cs.delta_max, h_prev_norm=cs.H, h_norm=cs.H)
    assert tr.block_bound() <= cs.l_block * (1 + 1e-9)
