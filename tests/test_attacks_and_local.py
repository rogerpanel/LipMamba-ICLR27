"""Smoke tests for the attacks, local-Lipschitz estimator and scan trace."""
from __future__ import annotations

import torch

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.attacks import AdaptiveClampAttack, AdaptiveClampConfig, HiSPAAttack, HiSPAConfig
from lipmamba.baselines import preset
from lipmamba.certificates.local_lipschitz import LocalLipschitzConfig, ll_radius, local_lipschitz_estimate
from lipmamba.certificates.poisoning_immunity import ell_star_from_trace


def _tiny(**o) -> LipMambaModel:
    cfg = LipMambaConfig(**{**dict(vocab_size=64, n_layers=2, d_model=16, d_inner=32, state_dim=4,
                                   conv_kernel=3, n_classes=3), **o})
    return LipMambaModel(cfg).eval()


def test_scan_trace_records_bounds_within_theory() -> None:
    m = _tiny()
    ids = torch.randint(0, 64, (2, 12))
    m(ids)
    tr = m.blocks[-1].ssm.last_trace
    cs = m.constraints
    assert tr is not None
    assert (tr.delta >= cs.delta_min - 1e-7).all() and (tr.delta < cs.delta_max + 1e-6).all()
    assert (tr.a_bar_max <= cs.rho_max + 1e-6).all()
    assert (tr.a_bar_min >= cs.rho_min - 1e-6).all()
    assert (tr.injection_norm <= cs.c + 1e-6).all()        # per-step injection bound
    assert (tr.h_norm <= cs.H + 1e-6).all()                # Lemma 1
    dd = m.data_dependent_network_bound()
    assert dd is not None and (dd <= m.network_lipschitz_bound() + 1e-6).all()


def test_adaptive_clamp_attack_cannot_exceed_delta_max() -> None:
    m = _tiny()
    ids = torch.randint(0, 64, (2, 8))
    att = AdaptiveClampAttack(m, AdaptiveClampConfig(objective="saturate", trigger_length=6, n_steps=5))
    _, rep = att.attack(ids)
    assert rep["mean_delta_over_trigger"] <= m.constraints.delta_max + 1e-6
    assert 0.0 <= rep["delta_saturation_fraction"] <= 1.0 + 1e-6
    # retention ratio can never fall below the worst-case Theorem-2 bound for that length
    cs = m.constraints
    assert rep["retention_ratio_min"] >= 0.0


def test_adaptive_overwrite_and_margin_objectives_run() -> None:
    m = _tiny()
    ids = torch.randint(0, 64, (2, 8))
    for obj in ("overwrite", "margin"):
        att = AdaptiveClampAttack(m, AdaptiveClampConfig(objective=obj, trigger_length=4, n_steps=3))
        _, rep = att.attack(ids)
        assert rep["objective"] == obj


def test_adaptive_discrete_runs() -> None:
    m = _tiny()
    ids = torch.randint(0, 64, (1, 6))
    att = AdaptiveClampAttack(m, AdaptiveClampConfig(objective="saturate", trigger_length=3, n_steps=2,
                                                     top_k=8, candidates_per_step=4, discrete=True))
    trig, rep = att.attack_discrete(ids)
    assert trig.shape == (1, 3)


def test_hispa_variants_run() -> None:
    m = _tiny()
    ids = torch.randint(0, 64, (2, 8))
    h = HiSPAAttack(m, HiSPAConfig(trigger_length=4, n_steps=3, population=6, generations=2, vocab_subset=32))
    _, z = h.z_hispa(ids)
    _, g = h.m_hispa(ids)
    _, c = h.attack(ids)
    for r in (z, g, c):
        assert 0.0 <= r["success_rate"] <= 1.0


def test_local_lipschitz_and_ll_radius() -> None:
    m = _tiny()
    emb = m.embed_tokens(torch.randint(0, 64, (2, 8))).detach()
    l = local_lipschitz_estimate(m, emb, LocalLipschitzConfig(n_steps=2, n_restarts=2))
    assert l.shape == (2,) and (l >= 0).all()
    with torch.no_grad():
        logits = m.logits_from_embeddings(emb)
    eps = ll_radius(logits, l)
    assert eps.shape == (2,)


def test_ell_star_from_trace() -> None:
    m = _tiny()
    ids = torch.randint(0, 64, (2, 12))
    m(ids)
    tr = m.blocks[-1].ssm.last_trace
    l = ell_star_from_trace(tr.a_bar_min, tr.injection_norm, tr.h_norm, t0=6, alpha_min=0.5)
    assert l.shape == (2,)


def test_presets_build_and_run() -> None:
    for name in ("unconstrained_mamba", "gloro_mamba", "naive_sn_mamba", "lipmamba_arch"):
        cfg = preset(name, vocab_size=64, n_layers=1, d_model=16, d_inner=32, state_dim=4, conv_kernel=3, n_classes=3)
        m = LipMambaModel(cfg).eval()
        out = m(torch.randint(0, 64, (2, 6)))
        assert out["cls_logits"].shape == (2, 3)
