"""ConformalGuard (Module 5): validity of the e-value, calibration, alarm behaviour."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.integrate import quad

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.attacks import AdaptiveClampAttack, AdaptiveClampConfig
from lipmamba.monitoring import ConformalGuard, ConformalGuardConfig, mixture_e_value


def test_mixture_e_value_integrates_to_one_under_uniform_p() -> None:
    """E_{p~U(0,1)}[e(p)] = 1 (an e-value); checked by quadrature because the
    Monte-Carlo mean has infinite variance (e(p) ~ 1/(p ln²p) near 0)."""
    from lipmamba.monitoring.conformal_guard import KAPPA_GRID
    f = lambda p: float(mixture_e_value(torch.tensor([p], dtype=torch.float64)))
    delta = 1e-12                                           # numerical clamp inside mixture_e_value
    val, err = quad(f, delta, 1.0, limit=400)
    expected = 1.0 - float(np.mean([delta**k for k in KAPPA_GRID]))   # ∫_δ^1 κ p^{κ−1} dp = 1 − δ^κ
    assert abs(val - expected) < 2e-3, (val, expected, err)
    assert f(1e-6) > 100.0 and abs(f(1.0) - 0.5) < 1e-6      # mean of κ over the grid = 0.5


def _model():
    return LipMambaModel(LipMambaConfig(vocab_size=64, n_layers=2, d_model=16, d_inner=32, state_dim=4,
                                        conv_kernel=3, n_classes=3)).eval()


def test_false_alarm_rate_on_clean_is_controlled() -> None:
    torch.manual_seed(0)
    m = _model()
    guard = ConformalGuard(m, ConformalGuardConfig(alpha=0.05)).fit(torch.randint(0, 64, (64, 24)))
    res = guard.monitor(torch.randint(0, 64, (200, 24)))
    assert float(res.alarmed.float().mean()) <= 0.10      # nominal 0.05; slack for finite calibration


def test_extreme_telemetry_triggers_alarm_and_tightening() -> None:
    torch.manual_seed(0)
    m = _model()
    clean = torch.randint(0, 64, (64, 16))
    guard = ConformalGuard(m, ConformalGuardConfig(alpha=0.05)).fit(clean)
    T = 16
    feats = {"delta_sat": torch.ones(2, T), "norm_loss": torch.ones(2, T), "injection": torch.full((2, T), 10.0)}
    res = guard.monitor(feats=feats)
    assert bool(res.alarmed.all()) and int(res.alarm_time.max()) < T
    assert guard.recommend_delta_max(res) < m.constraints.delta_max


def test_monitor_runs_on_attacked_embeddings() -> None:
    torch.manual_seed(0)
    m = _model()
    clean = torch.randint(0, 64, (64, 16))
    guard = ConformalGuard(m).fit(clean)
    att = AdaptiveClampAttack(m, AdaptiveClampConfig(objective="saturate", trigger_length=8, n_steps=10))
    trig, _ = att.attack(clean[:4])
    res = guard.monitor(emb=torch.cat([m.embed_tokens(clean[:4]), trig], 1))
    assert res.martingale.shape == (4, 24) and torch.isfinite(res.martingale).all()


def test_dashboard_payload_fields() -> None:
    torch.manual_seed(0)
    m = _model()
    ids = torch.randint(0, 64, (4, 12))
    guard = ConformalGuard(m).fit(ids)
    res = guard.monitor(ids)
    logits = m(ids)["cls_logits"]; l_loc = torch.ones(4)
    payload = guard.dashboard_payload(ids, logits, l_loc, res)
    assert len(payload) == 4 and {"prediction", "ll_radius_epsilon_star", "martingale_final", "alarm",
                                   "recommended_delta_max"} <= set(payload[0])
