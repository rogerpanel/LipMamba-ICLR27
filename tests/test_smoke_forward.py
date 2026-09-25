"""End-to-end smoke test: a tiny LipMambaModel runs and produces logits."""
from __future__ import annotations

import math

import torch

from lipmamba import LipMambaConfig, LipMambaModel


def test_tiny_forward_returns_logits() -> None:
    cfg = LipMambaConfig(vocab_size=100, n_layers=2, d_model=32, d_inner=64, state_dim=8,
                         conv_kernel=3, n_classes=4)
    out = LipMambaModel(cfg)(torch.randint(0, 100, (2, 16)))
    assert out["lm_logits"].shape == (2, 16, 100)
    assert out["cls_logits"].shape == (2, 4)


def test_global_bound_is_product_of_block_bounds_and_vacuous_at_depth() -> None:
    cfg = LipMambaConfig(vocab_size=50, n_layers=24, d_model=16, d_inner=32, state_dim=4)
    model = LipMambaModel(cfg)
    log10 = model.log10_network_lipschitz_bound()
    assert math.isclose(log10, 24 * math.log10(1 + model.constraints.l_block), rel_tol=1e-9)
    assert log10 > 100   # Remark 4: the global GloRo radius is vacuous
