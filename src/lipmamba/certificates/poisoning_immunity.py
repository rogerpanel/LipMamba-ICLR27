"""State-retention bound (Theorem 2, formerly "poisoning immunity").

    ‖h_{t0+ℓ}‖₂ ≥ ρ_min^ℓ ‖h_{t0}‖₂ − c (1 − ρ_min^ℓ)/(1 − ρ_min)
    ℓ* = log((α_min + κ)/(1 + κ)) / log(ρ_min),   κ = c / ((1 − ρ_min) ‖h_{t0}‖₂)

where ρ_min = exp(−Δ_max λ_max) lower-bounds σ_min(Ā_t) (the *smallest*
singular value, which for diagonal Ā_t equals min_i e^{Δ_t a_i}).

**What this certifies (Remark 5).**  Norm, not content.  It rules out the
HiSPA collapse mechanism Ā_t → 0 because under the clamp σ_min(Ā_t) ≥ ρ_min
for every token; it does not exclude an adversary that keeps the norm large
while overwriting the state through B̄_t x_t.  ℓ* depends on ‖h_{t0}‖ and is
therefore per-input; report its distribution.

Two versions are provided:

* worst-case: uses the constraint constants (ρ_min, c) → ``ConstraintSet``;
* data-dependent: uses the observed per-token σ_min(Ā_t) and ‖B̄_t x_t‖ from a
  ``ScanTrace`` along the actual trigger → tighter, per-input.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .constants import ConstraintSet, data_dependent_ell_star

__all__ = [
    "retention_lower_bound", "ell_star", "ell_star_distribution",
    "ell_star_from_trace", "sweep_ell_star", "retention_summary",
    # backward-compatible aliases
    "poisoning_immunity_lower_bound", "max_certified_trigger_length", "certified_immunity_summary",
]


def retention_lower_bound(cs: ConstraintSet, h0_norm: float, ell: int) -> float:
    return cs.retention_lower_bound(h0_norm, ell)


def ell_star(cs: ConstraintSet, h0_norm: float, alpha_min: float = 0.5) -> float:
    return cs.ell_star(h0_norm, alpha_min)


def ell_star_distribution(cs: ConstraintSet, h0_norms: torch.Tensor | np.ndarray, alpha_min: float = 0.5) -> dict:
    """Per-input ℓ* over observed pre-trigger norms ‖h_{t0}‖ (Remark 5)."""
    h = np.asarray(h0_norms, dtype=float).ravel()
    h = h[h > 0]
    ls = np.array([cs.ell_star(float(v), alpha_min) for v in h])
    return {
        "n": int(ls.size),
        "alpha_min": alpha_min,
        "rho_min": cs.rho_min,
        "c": cs.c,
        "h0_norm_median": float(np.median(h)) if h.size else float("nan"),
        "ell_star_min": float(ls.min()) if ls.size else float("nan"),
        "ell_star_p05": float(np.percentile(ls, 5)) if ls.size else float("nan"),
        "ell_star_median": float(np.median(ls)) if ls.size else float("nan"),
        "ell_star_p95": float(np.percentile(ls, 95)) if ls.size else float("nan"),
        "ell_star_max": float(ls.max()) if ls.size else float("nan"),
        "ell_star_int_min": int(math.floor(ls.min())) if ls.size else 0,
        "values": ls.tolist(),
    }


def ell_star_from_trace(
    a_bar_min: torch.Tensor, injection_norm: torch.Tensor, h_norm: torch.Tensor,
    t0: int, alpha_min: float = 0.5,
) -> torch.Tensor:
    """Data-dependent ℓ* from a ScanTrace, per sample.

    Uses the minimum σ_min(Ā_t) and the maximum ‖B̄_t x_t‖ observed *after*
    position t0 as the constants of Theorem 2 (still a valid bound for that
    trigger, and far tighter than the worst case)."""
    rho = a_bar_min[:, t0:].amin(dim=1)
    inj = injection_norm[:, t0:].amax(dim=1)
    h0 = h_norm[:, t0 - 1] if t0 > 0 else h_norm[:, 0]
    out = []
    for r, i, h in zip(rho.tolist(), inj.tolist(), h0.tolist()):
        out.append(data_dependent_ell_star(r, i, max(h, 1e-12), alpha_min))
    return torch.tensor(out)


def sweep_ell_star(
    base: ConstraintSet, h0_norm: float = 4.0, alpha_min: float = 0.5,
    delta_max_grid=(0.5, 0.25, 0.1, 0.05, 0.03, 0.02, 0.01),
    lambda_max_grid=(1.0, 0.5, 0.25, 0.1),
) -> list[dict]:
    """ℓ* over a (Δ_max, λ_max) grid — the two knobs that set ρ_min."""
    rows = []
    for dm in delta_max_grid:
        for lm in lambda_max_grid:
            if lm < base.lambda_min:
                continue
            cs = ConstraintSet(**{**base.__dict__, "delta_max": dm, "lambda_max": lm,
                                  "delta_min": min(base.delta_min, dm / 2)})
            rows.append({"delta_max": dm, "lambda_max": lm, "rho_min": cs.rho_min,
                         "ell_star": cs.ell_star(h0_norm, alpha_min),
                         "L_block": cs.l_block})
    return rows


def retention_summary(cs: ConstraintSet, h0_norm: float = 4.0, alpha_min: float = 0.5) -> dict:
    l = cs.ell_star(h0_norm, alpha_min)
    return {
        "rho_min": cs.rho_min, "c": cs.c, "kappa": cs.kappa(h0_norm),
        "ell_star": l, "ell_star_int": int(math.floor(l)),
        "lower_bound_at_ell_star_int": cs.retention_lower_bound(h0_norm, int(math.floor(l))),
    }


# ---------------------------------------------------------------------------
# Backward-compatible aliases (old names used "immunity"; the manuscript now
# says "state retention").  ρ_min is now exp(−Δ_max λ_max) as in Assumption 1.
# ---------------------------------------------------------------------------

def poisoning_immunity_lower_bound(rho_min: float, h0_norm: float, b_bar_max: float, x_max: float, ell: int) -> float:
    c = b_bar_max * x_max
    r = rho_min**ell
    return r * h0_norm - c * (1 - r) / (1 - rho_min)


def max_certified_trigger_length(rho_min: float, h0_norm: float, b_bar_max: float, x_max: float, alpha: float) -> int:
    c = b_bar_max * x_max
    k = c / ((1 - rho_min) * h0_norm)
    ratio = (alpha + k) / (1 + k)
    if ratio >= 1.0:
        return 0
    return int(max(0, math.floor(math.log(ratio) / math.log(rho_min))))


def certified_immunity_summary(*, delta_min: float, lambda_min: float, s_b: float, delta_max: float,
                               x_max: float, h0_norm: float, alpha: float = 0.05,
                               lambda_max: float = 1.0) -> dict:
    cs = ConstraintSet(s_b=s_b, delta_min=delta_min, delta_max=delta_max,
                       lambda_min=lambda_min, lambda_max=lambda_max, x_max=x_max)
    return retention_summary(cs, h0_norm, alpha)
