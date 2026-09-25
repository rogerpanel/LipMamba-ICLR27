"""Randomized smoothing certification (Cohen, Rosenfeld & Kolter, ICML 2019).

An independent, well-established certification baseline for the LL-Acc
curve of Figure 3: for a base classifier f and Gaussian noise σ, the
smoothed classifier g(x) = argmax_c P(f(x + N(0,σ²I)) = c) is certified
robust in ℓ₂ radius R = σ Φ⁻¹(p_A_lower) whenever p_A_lower > 1/2, where
p_A_lower is a Clopper-Pearson lower confidence bound on the top-class
probability estimated from n Monte-Carlo samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from scipy.stats import beta, norm


@dataclass
class SmoothingConfig:
    sigma: float = 0.25
    n0: int = 64          # selection samples
    n: int = 512          # estimation samples
    alpha: float = 0.001  # confidence
    batch: int = 64


class RandomizedSmoothing:
    def __init__(self, model: nn.Module, cfg: SmoothingConfig) -> None:
        self.model = model
        self.cfg = cfg

    @torch.no_grad()
    def _counts(self, emb: torch.Tensor, n: int, n_classes: int) -> torch.Tensor:
        counts = torch.zeros(n_classes, device=emb.device)
        for s in range(0, n, self.cfg.batch):
            b = min(self.cfg.batch, n - s)
            noise = torch.randn(b, *emb.shape[1:], device=emb.device) * self.cfg.sigma
            preds = self.model.logits_from_embeddings(emb.expand(b, -1, -1) + noise).argmax(-1)
            counts += torch.bincount(preds, minlength=n_classes).float()
        return counts

    @torch.no_grad()
    def certify(self, emb: torch.Tensor, n_classes: int) -> tuple[int, float]:
        """Return (predicted class or -1 for abstain, certified radius)."""
        c0 = self._counts(emb, self.cfg.n0, n_classes)
        c_a = int(c0.argmax())
        c1 = self._counts(emb, self.cfg.n, n_classes)
        k = int(c1[c_a])
        p_lower = beta.ppf(self.cfg.alpha, k, self.cfg.n - k + 1) if k > 0 else 0.0
        if p_lower > 0.5:
            return c_a, float(self.cfg.sigma * norm.ppf(p_lower))
        return -1, 0.0

    @torch.no_grad()
    def certified_curve(self, embs: torch.Tensor, labels: torch.Tensor, n_classes: int, radii) -> dict[float, float]:
        preds, rads = [], []
        for i in range(embs.size(0)):
            p, r = self.certify(embs[i: i + 1], n_classes)
            preds.append(p); rads.append(r)
        preds = torch.tensor(preds, device=labels.device); rads = torch.tensor(rads, device=labels.device)
        return {float(r): float(((preds == labels) & (rads >= r)).float().mean()) for r in radii}
