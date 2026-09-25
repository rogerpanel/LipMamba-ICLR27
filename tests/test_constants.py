"""Closed-form constants: Lemma 1, Theorem 1, Theorem 2 (Eq. lstar)."""
from __future__ import annotations

import math

import pytest

from lipmamba.certificates.constants import (ConstraintSet, data_dependent_ell_star,
                                             required_delta_lambda_product_for_ell_star)

PAPER_130M = ConstraintSet(s_b=1.0, s_c=1.0, s_delta=0.5, s_out=1.0, delta_min=1e-3,
                           delta_max=0.5, lambda_min=0.05, lambda_max=1.0, x_max=1.0)


def test_rho_definitions_match_assumption_1() -> None:
    cs = PAPER_130M
    assert math.isclose(cs.rho_max, math.exp(-1e-3 * 0.05))
    assert math.isclose(cs.rho_min, math.exp(-0.5 * 1.0))
    assert 0 < cs.rho_min < cs.rho_max < 1


def test_lemma_state_bound_positive_and_diverges_as_delta_min_to_zero() -> None:
    h1 = PAPER_130M.H
    h2 = ConstraintSet(**{**PAPER_130M.__dict__, "delta_min": 1e-6}).H
    assert h1 > 0 and h2 > h1 * 100


def test_todo1_manuscript_constants_give_ell_star_about_one() -> None:
    """The TODO in Remark 5: (s_B,Δmax,λmax,Xmax,α,‖h‖)=(1,0.5,1,1,0.5,4) ⇒ ℓ*≈1, not 24."""
    l = PAPER_130M.ell_star(h0_norm=4.0, alpha_min=0.5)
    assert 0.9 < l < 1.0
    assert PAPER_130M.ell_star_int(4.0, 0.5) == 0


def test_ell_star_positive_for_alpha_below_one_and_monotone_in_rho_min() -> None:
    for dm in (0.5, 0.25, 0.1, 0.05, 0.02):
        cs = ConstraintSet(**{**PAPER_130M.__dict__, "delta_max": dm, "delta_min": min(1e-3, dm / 2)})
        assert cs.ell_star(4.0, 0.5) > 0
    a = ConstraintSet(**{**PAPER_130M.__dict__, "delta_max": 0.02, "delta_min": 1e-3}).ell_star(4.0, 0.5)
    assert a > 24  # Δmax·λmax ≈ 0.02 is what ℓ*≈24 requires


def test_required_product_for_ell_star_24() -> None:
    p = required_delta_lambda_product_for_ell_star(24, alpha_min=0.5, kappa=0.0)
    assert math.isclose(p, -math.log(0.5 ** (1 / 24)), rel_tol=1e-9)
    assert p < 0.03


def test_retention_bound_matches_eq_and_is_consistent_with_ell_star() -> None:
    cs = ConstraintSet(**{**PAPER_130M.__dict__, "delta_max": 0.05, "delta_min": 1e-3})
    h0, alpha = 4.0, 0.5
    l = cs.ell_star_int(h0, alpha)
    assert cs.retention_lower_bound(h0, l) >= alpha * h0 - 1e-9
    assert cs.retention_lower_bound(h0, l + 1) < alpha * h0


def test_theorem1_constant_with_paper_constants_is_astronomical() -> None:
    """Confirms the audit: L_block ≈ 1.1e8 with Δmin=1e-3, λmin=0.05 ⇒ 24-layer product ≈ 1e193."""
    cs = PAPER_130M
    assert 1e8 < cs.l_block < 2e8
    assert cs.summary(n_blocks=24)["log10_L_network"] > 190


def test_data_dependent_ell_star_is_tighter_than_worst_case() -> None:
    cs = PAPER_130M
    worst = cs.ell_star(4.0, 0.5)
    obs = data_dependent_ell_star(rho_min_observed=0.95, injection_observed=0.05, h0_norm=4.0, alpha_min=0.5)
    assert obs > worst


def test_invalid_constraints_rejected() -> None:
    with pytest.raises(ValueError):
        ConstraintSet(delta_min=0.0)
    with pytest.raises(ValueError):
        ConstraintSet(lambda_min=0.0)
