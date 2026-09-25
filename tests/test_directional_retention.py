"""Corollary (directional retention) and the one-token condition of Theorem 2."""
from __future__ import annotations

import math

import torch

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.certificates.constants import ConstraintSet

PAPER = ConstraintSet(s_b=1.0, s_c=1.0, s_delta=0.5, s_out=1.0, delta_min=1e-3, delta_max=0.5,
                      lambda_min=0.05, lambda_max=1.0, x_max=1.0)


def test_one_token_condition_threshold() -> None:
    # ℓ* ≥ 1 iff ‖h‖ ≥ c/(ρ_min − α): 0.5/(0.6065−0.5) = 4.69
    thr = PAPER.c / (PAPER.rho_min - 0.5)
    assert math.isclose(thr, 4.69, abs_tol=0.01)
    assert not PAPER.one_token_certified(4.0, 0.5) and PAPER.one_token_certified(4.7, 0.5)
    assert PAPER.ell_star(4.7, 0.5) >= 1.0 > PAPER.ell_star(4.0, 0.5)


def test_certified_cosine_paper_value() -> None:
    assert math.isclose(PAPER.certified_cosine(4.0, 1), 0.43, abs_tol=0.01)
    assert PAPER.certified_cosine(4.0, 1) <= 1.0
    # at ℓ = 1 the directional and norm certificates coincide (S_1 = c)
    assert math.isclose(PAPER.injected_norm_bound(1), PAPER.c)
    assert math.isclose(PAPER.directional_retention_lower_bound(4.0, 1), 4.0 * PAPER.retention_lower_bound(4.0, 1))
    # S_ℓ ≤ c ℓ and grows with ℓ
    assert PAPER.injected_norm_bound(8) <= 8 * PAPER.c + 1e-12 and PAPER.injected_norm_bound(8) > PAPER.injected_norm_bound(1)


def test_directional_bound_holds_on_random_trajectories() -> None:
    """⟨h_{t0+ℓ}, h_{t0}⟩ and cos∠ never fall below the corollary on a real scan."""
    torch.manual_seed(0)
    cfg = LipMambaConfig(vocab_size=64, n_layers=1, d_model=16, d_inner=32, state_dim=4, conv_kernel=3,
                         delta_max=0.5, lambda_max=1.0)
    m = LipMambaModel(cfg).eval()
    ssm = m.blocks[0].ssm
    cs = m.constraints
    x = torch.randn(8, 20, 32)
    x = ssm.input_clip(x)
    delta = ssm.compute_delta(x); b_t = ssm.x_to_b(x)
    a_bar = ssm.A.discretise(delta); b_bar = delta.unsqueeze(-1) * b_t.unsqueeze(-2)
    h = x.new_zeros(8, 32, 4); states = []
    for t in range(20):
        h = a_bar[:, t] * h + b_bar[:, t] * x[:, t].unsqueeze(-1); states.append(h.clone())
    t0 = 10
    h0 = states[t0 - 1].flatten(1)
    for ell in (1, 2, 4, 8):
        hl = states[t0 - 1 + ell].flatten(1)
        ip = (hl * h0).sum(-1); n0 = h0.norm(dim=-1)
        for b in range(8):
            lb = cs.directional_retention_lower_bound(float(n0[b]), ell)
            assert float(ip[b]) >= lb - 1e-5
            cos = float(ip[b] / (n0[b] * hl[b].norm() + 1e-12))
            assert cos >= cs.certified_cosine(float(n0[b]), ell) - 1e-5
