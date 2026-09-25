"""Closed-form constants of the ICLR manuscript (Assumption 1, Lemma 1,
Theorem 1, Theorem 2).  Pure Python, no torch dependency, so the same
numbers can be reproduced from a shell one-liner.

Notation (Assumption 1)
-----------------------
    c        = s_B · Δ_max · X_max²               per-step injection bound ‖B̄_t x_t‖ ≤ c
    ρ_max    = exp(−Δ_min · λ_min)                 upper bound on ‖Ā_t‖₂
    ρ_min    = exp(−Δ_max · λ_max)                 lower bound on σ_min(Ā_t)
    H        = c / (1 − ρ_max)                    Lemma 1 (bounded state)
    γ        = 2 s_B Δ_max X_max + s_Δ (λ_max H + s_B X_max²)     (proof step (iii): |Δ−Δ′|‖B_t‖‖x_t‖ ≤ s_Δ‖δ‖·s_B X_max·X_max)
    L_block  = s_out · L_SiLU · s_C · ( X_max γ / (1 − ρ_max) + H )     Theorem 1
    κ        = c / ((1 − ρ_min) ‖h_{t0}‖)
    ℓ*       = log((α_min + κ)/(1 + κ)) / log(ρ_min)                     Theorem 2, Eq. (lstar)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

L_SILU: float = 1.0998  # Lipschitz constant of SiLU (Appendix B.1 of the source paper)


@dataclass(frozen=True)
class ConstraintSet:
    """The constraint budget of one LipMamba block."""

    s_b: float = 1.0
    s_c: float = 1.0
    s_delta: float = 0.5
    s_out: float = 1.0
    delta_min: float = 1e-3
    delta_max: float = 0.5
    lambda_min: float = 0.05
    lambda_max: float = 1.0
    x_max: float = 1.0
    l_silu: float = L_SILU

    def __post_init__(self) -> None:
        if not (0.0 < self.delta_min < self.delta_max):
            raise ValueError("require 0 < delta_min < delta_max")
        if not (0.0 < self.lambda_min <= self.lambda_max):
            raise ValueError("require 0 < lambda_min <= lambda_max")
        if self.x_max <= 0:
            raise ValueError("x_max must be positive")

    # -- Assumption 1 -------------------------------------------------------
    @property
    def c(self) -> float:
        return self.s_b * self.delta_max * self.x_max**2

    @property
    def rho_max(self) -> float:
        return math.exp(-self.delta_min * self.lambda_min)

    @property
    def rho_min(self) -> float:
        return math.exp(-self.delta_max * self.lambda_max)

    # -- Lemma 1 ------------------------------------------------------------
    @property
    def H(self) -> float:  # noqa: N802
        return self.c / (1.0 - self.rho_max)

    # -- Theorem 1 ----------------------------------------------------------
    @property
    def gamma(self) -> float:
        return 2.0 * self.s_b * self.delta_max * self.x_max + self.s_delta * (
            self.lambda_max * self.H + self.s_b * self.x_max**2
        )

    @property
    def l_block(self) -> float:
        return (
            self.s_out
            * self.l_silu
            * self.s_c
            * (self.x_max * self.gamma / (1.0 - self.rho_max) + self.H)
        )

    def l_network(self, n_blocks: int, residual: bool = False) -> float:
        """Product bound L_SSM ≤ ∏ L_block.  ``residual=True`` uses (1 + L_block)
        per block, which is the correct factor when the block is wrapped in a
        skip connection (the manuscript's f_block has no skip)."""
        per = (1.0 + self.l_block) if residual else self.l_block
        return per**n_blocks

    # -- Theorem 2 ----------------------------------------------------------
    def kappa(self, h0_norm: float) -> float:
        return self.c / ((1.0 - self.rho_min) * h0_norm)

    def retention_lower_bound(self, h0_norm: float, ell: int) -> float:
        """Eq. (retention): ‖h_{t0+ℓ}‖ ≥ ρ_min^ℓ ‖h_{t0}‖ − c (1 − ρ_min^ℓ)/(1 − ρ_min)."""
        r = self.rho_min**ell
        return r * h0_norm - self.c * (1.0 - r) / (1.0 - self.rho_min)

    def ell_star(self, h0_norm: float, alpha_min: float = 0.5) -> float:
        """Eq. (lstar).  Real-valued; the certified integer length is floor()."""
        if not (0.0 < alpha_min < 1.0):
            raise ValueError("alpha_min must lie in (0, 1)")
        k = self.kappa(h0_norm)
        return math.log((alpha_min + k) / (1.0 + k)) / math.log(self.rho_min)

    def ell_star_int(self, h0_norm: float, alpha_min: float = 0.5) -> int:
        return int(math.floor(self.ell_star(h0_norm, alpha_min)))

    def one_token_certified(self, h0_norm: float, alpha_min: float = 0.5) -> bool:
        """Theorem 2 condition ℓ* ≥ 1  ⇔  ρ_min(1+κ) − κ ≥ α_min."""
        k = self.kappa(h0_norm)
        return self.rho_min * (1.0 + k) - k >= alpha_min

    # -- Corollary (directional retention) ---------------------------------
    def injected_norm_bound(self, ell: int) -> float:
        """S_ℓ = c (1 − ρ_max^ℓ)/(1 − ρ_max) ≤ c ℓ: total norm the trigger can inject.

        Uses the *upper* contraction ρ_max (each injection is only guaranteed to
        shrink by ≤ ρ_max afterwards); the ρ_min-discounted sum of Theorem 2 is
        valid for the norm recursion but not for the inner product."""
        return self.c * (1.0 - self.rho_max**ell) / (1.0 - self.rho_max)

    def directional_retention_lower_bound(self, h0_norm: float, ell: int) -> float:
        """⟨h_{t0+ℓ}, h_{t0}⟩ ≥ ‖h_{t0}‖ (ρ_min^ℓ ‖h_{t0}‖ − S_ℓ)."""
        return h0_norm * (self.rho_min**ell * h0_norm - self.injected_norm_bound(ell))

    def certified_cosine(self, h0_norm: float, ell: int) -> float:
        """cos∠(h_{t0+ℓ}, h_{t0}) ≥ (ρ_min^ℓ − S_ℓ/‖h0‖) / (1 + S_ℓ/‖h0‖)   (≤ 0 means no certificate)."""
        s = self.injected_norm_bound(ell) / h0_norm
        return (self.rho_min**ell - s) / (1.0 + s)

    # -- reporting ----------------------------------------------------------
    def summary(self, n_blocks: int = 24, h0_norm: float = 4.0, alpha_min: float = 0.5) -> dict:
        d = asdict(self)
        d.update(
            c=self.c,
            rho_max=self.rho_max,
            rho_min=self.rho_min,
            H=self.H,
            gamma=self.gamma,
            L_block=self.l_block,
            L_network_product=self.l_network(n_blocks),
            log10_L_network=math.log10(self.l_network(n_blocks)),
            kappa=self.kappa(h0_norm),
            ell_star=self.ell_star(h0_norm, alpha_min),
            ell_star_int=self.ell_star_int(h0_norm, alpha_min),
        )
        return d


def required_delta_lambda_product_for_ell_star(
    ell_target: int, alpha_min: float = 0.5, kappa: float = 0.0
) -> float:
    """Invert Eq. (lstar) for the product Δ_max·λ_max needed to certify a
    trigger length ``ell_target``.  With κ → 0 this is −ln(α_min^{1/ℓ})."""
    ratio = (alpha_min + kappa) / (1.0 + kappa)
    rho_needed = ratio ** (1.0 / ell_target)
    return -math.log(rho_needed)


def data_dependent_ell_star(
    rho_min_observed: float,
    injection_observed: float,
    h0_norm: float,
    alpha_min: float = 0.5,
) -> float:
    """Theorem 2 with the *observed* per-token σ_min(Ā_t) and ‖B̄_t x_t‖ along a
    specific trigger, instead of the worst-case ρ_min and c.  This is the
    per-input refinement analogous to the D_t tracking of Algorithm 1 and is
    what should be reported as a distribution (Remark 5)."""
    if not (0.0 < rho_min_observed < 1.0):
        return float("inf") if rho_min_observed >= 1.0 else 0.0
    k = injection_observed / ((1.0 - rho_min_observed) * h0_norm)
    return math.log((alpha_min + k) / (1.0 + k)) / math.log(rho_min_observed)
