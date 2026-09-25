"""PAC-Bayes / KL invariants."""
from __future__ import annotations

import math

import torch

from lipmamba.certificates.pac_bayes import (
    PACBayesConfig,
    gaussian_kl_divergence,
    pac_bayes_complexity,
    pac_bayes_training_term,
)


def test_kl_zero_when_distributions_match() -> None:
    mu = torch.zeros(16)
    kl = gaussian_kl_divergence(mu, mu, sigma_post=0.1, sigma_prior=0.1)
    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)


def test_complexity_decreases_with_n() -> None:
    kl = torch.tensor(5.0)
    small = pac_bayes_complexity(kl, n=100, delta=0.05).item()
    large = pac_bayes_complexity(kl, n=10_000, delta=0.05).item()
    assert large < small


def test_training_term_components_finite() -> None:
    cfg = PACBayesConfig(n_train=1024)
    posterior = torch.zeros(32)
    prior = torch.zeros(32)
    pieces = pac_bayes_training_term(posterior, prior, l_ssm=torch.tensor(5.0), cfg=cfg)
    for k, v in pieces.items():
        assert torch.isfinite(v).all(), k
