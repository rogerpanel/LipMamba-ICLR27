"""Selective state-space recurrence with Lipschitz-bounded projections.

Implements the S6 layer of Eq. (s6) under the LipMamba constraints and
Algorithm 1 (forward pass with online tracking of the analytical constant):

    h_t = Ā_t h_{t-1} + B̄_t x_t ,   y_t = C_tᵀ h_t
    Ā_t = exp(Δ_t A),   B̄_t = Δ_t B_t,   Δ_t ∈ [Δ_min, Δ_max)

with the *data-dependent refinement* of Theorem 1

    γ_t = 2 s_B Δ_t X_max + s_Δ (λ_max ‖h_{t-1}‖₂ + s_B X_max²)
    D_t = ‖Ā_t‖₂ D_{t-1} + γ_t
    L_block(data) = s_out L_SiLU s_C ( X_max · max_t D_t + max_t ‖h_t‖₂ )

which is always ≤ the worst-case closed form of Theorem 1 (it replaces
ρ_max, H by their observed values) and is the quantity plotted as the
"analytical" curve of Figure 2 *if* the figure is meant to be data-dependent.
Both are exposed so the two can be compared.

The scan is a pure-PyTorch loop for CPU portability.  It also records, per
token, σ_min(Ā_t) and ‖B̄_t x_t‖ so that the retention bound (Theorem 2) can
be evaluated with observed rather than worst-case constants.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..certificates.constants import L_SILU, ConstraintSet
from .clipped_delta import ClippedDelta
from .eigen_reparam import EigenReparamA
from .hippo import hippo_init
from .input_clip import InputNormClip
from .spectral_norm import SpectralNormLinear


@dataclass
class SSMConfig:
    """Selective SSM hyper-parameters (Sections 3-4 of the manuscript)."""

    d_model: int = 1024
    d_inner: int = 2048
    state_dim: int = 16
    s_b: float = 1.0
    s_c: float = 1.0
    s_delta: float = 0.5
    s_out: float = 1.0
    delta_min: float = 1e-3
    delta_max: float = 0.5
    lambda_min: float = 0.05
    lambda_max: float = 1.0
    x_max: float = 1.0
    n_power_iters: int = 1
    track_lipschitz: bool = True
    # Ablation switches (Table 3)
    clamp_delta: bool = True          # False → vanilla softplus Δ_t
    reparam_eigen: bool = True        # False → free (unconstrained) A
    spectral_norm: bool = True        # False → plain linear projections

    def constraints(self) -> ConstraintSet:
        return ConstraintSet(
            s_b=self.s_b, s_c=self.s_c, s_delta=self.s_delta, s_out=self.s_out,
            delta_min=self.delta_min, delta_max=self.delta_max,
            lambda_min=self.lambda_min, lambda_max=self.lambda_max, x_max=self.x_max,
        )


@dataclass
class ScanTrace:
    """Per-token statistics recorded during a forward pass."""

    delta: torch.Tensor                     # (B, T, D)
    a_bar_max: torch.Tensor                 # (B, T)  max_i e^{Δ_t a_i}   = ‖Ā_t‖₂
    a_bar_min: torch.Tensor                 # (B, T)  min_i e^{Δ_t a_i}   = σ_min(Ā_t)
    injection_norm: torch.Tensor            # (B, T)  ‖B̄_t x_t‖₂ (max over channels)
    h_norm: torch.Tensor                    # (B, T)  ‖h_t‖₂ (max over channels)
    d_t: torch.Tensor                       # (B, T)  Algorithm 1 accumulator
    extras: dict = field(default_factory=dict)


class _PlainLinear(nn.Linear):
    """Linear with the SpectralNormLinear interface (for the ablation)."""

    @property
    def sigma(self) -> torch.Tensor:
        return torch.linalg.matrix_norm(self.weight, ord=2)


class SelectiveSSM(nn.Module):
    """Lipschitz-bounded selective scan over ``D`` channels."""

    def __init__(self, cfg: SSMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n, d = cfg.state_dim, cfg.d_inner

        self.input_clip = InputNormClip(cfg.x_max)

        if cfg.spectral_norm:
            self.x_to_b = SpectralNormLinear(d, n, s=cfg.s_b, n_power_iters=cfg.n_power_iters)
            self.x_to_c = SpectralNormLinear(d, n, s=cfg.s_c, n_power_iters=cfg.n_power_iters)
        else:
            self.x_to_b = _PlainLinear(d, n)
            self.x_to_c = _PlainLinear(d, n)

        self.delta_proj = ClippedDelta(
            d_model=d, d_inner=d,
            delta_min=cfg.delta_min, delta_max=cfg.delta_max,
            s_delta=cfg.s_delta if cfg.spectral_norm else 1e9,
            n_power_iters=cfg.n_power_iters,
        )

        self.A = EigenReparamA(
            state_dim=n, n_channels=d,
            lambda_min=cfg.lambda_min, lambda_max=cfg.lambda_max,
            free=not cfg.reparam_eigen,
        )
        hippo_init(self.A.alpha, cfg.lambda_min, cfg.lambda_max, free=not cfg.reparam_eigen)

        self.D = nn.Parameter(torch.ones(d))
        self._last_trace: ScanTrace | None = None

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.clamp_delta:
            return self.delta_proj(x)
        # Ablation: vanilla softplus (unbounded above, → 0 below)
        return torch.nn.functional.softplus(self.delta_proj.proj(x))

    def forward(self, x: torch.Tensor, record_trace: bool | None = None) -> torch.Tensor:
        cfg = self.cfg
        x = self.input_clip(x)                       # Assumption 1 by construction
        b, t, d = x.shape
        n = cfg.state_dim
        record = cfg.track_lipschitz if record_trace is None else record_trace

        delta = self.compute_delta(x)                # (B, T, D)
        b_t = self.x_to_b(x)                         # (B, T, N)
        c_t = self.x_to_c(x)                         # (B, T, N)
        a_bar = self.A.discretise(delta)             # (B, T, D, N)  = exp(Δ ⊙ A)
        b_bar = delta.unsqueeze(-1) * b_t.unsqueeze(-2)  # (B, T, D, N) = Δ_t B_t

        h = x.new_zeros(b, d, n)
        ys = []
        if record:
            a_max = x.new_zeros(b, t); a_min = x.new_zeros(b, t)
            inj = x.new_zeros(b, t); hn = x.new_zeros(b, t); dt = x.new_zeros(b, t)
            D_prev = x.new_zeros(b)
            with torch.no_grad():
                s_b_eff = cfg.s_b if cfg.spectral_norm else float(self.x_to_b.sigma)
                s_d_eff = cfg.s_delta if cfg.spectral_norm else float(self.delta_proj.proj.sigma)

        for tt in range(t):
            inj_t = b_bar[:, tt] * x[:, tt].unsqueeze(-1)              # (B, D, N)
            h_prev = h
            h = a_bar[:, tt] * h_prev + inj_t
            y = (h * c_t[:, tt].unsqueeze(1)).sum(dim=-1)             # (B, D)
            ys.append(y)

            if record:
                with torch.no_grad():
                    am = a_bar[:, tt].amax(dim=(1, 2)); an = a_bar[:, tt].amin(dim=(1, 2))
                    inj_norm = inj_t.norm(dim=-1).amax(dim=-1)        # max over channels
                    hprev_norm = h_prev.norm(dim=-1).amax(dim=-1)
                    gamma_t = (2.0 * s_b_eff * delta[:, tt].amax(dim=-1) * cfg.x_max
                               + s_d_eff * (cfg.lambda_max * hprev_norm + s_b_eff * cfg.x_max**2))
                    D_t = am * D_prev + gamma_t
                    a_max[:, tt] = am; a_min[:, tt] = an; inj[:, tt] = inj_norm
                    hn[:, tt] = h.norm(dim=-1).amax(dim=-1); dt[:, tt] = D_t
                    D_prev = D_t

        y = torch.stack(ys, dim=1) + self.D * x
        if record:
            self._last_trace = ScanTrace(
                delta=delta.detach(), a_bar_max=a_max, a_bar_min=a_min,
                injection_norm=inj, h_norm=hn, d_t=dt,
            )
        return y

    # ------------------------------------------------------------------ #
    # Certificates                                                       #
    # ------------------------------------------------------------------ #

    @property
    def constraints(self) -> ConstraintSet:
        return self.cfg.constraints()

    def block_lipschitz_bound(self, **_ignored) -> torch.Tensor:
        """Worst-case closed form of Theorem 1 (input-independent)."""
        return torch.tensor(self.constraints.l_block)

    def state_bound_H(self) -> float:
        """Lemma 1."""
        return self.constraints.H

    @torch.no_grad()
    def data_dependent_block_bound(self) -> torch.Tensor | None:
        """Algorithm 1: L_block = s_out L_SiLU s_C (X_max max_t D_t + max_t ‖h_t‖)."""
        tr = self._last_trace
        if tr is None:
            return None
        cfg = self.cfg
        return (cfg.s_out * L_SILU * cfg.s_c
                * (cfg.x_max * tr.d_t.amax(dim=1) + tr.h_norm.amax(dim=1)))   # (B,)

    @property
    def last_trace(self) -> ScanTrace | None:
        return self._last_trace

    @torch.no_grad()
    def lipschitz_state(self) -> dict[str, float]:
        out = {"L_block_worst_case": float(self.constraints.l_block), "H": self.constraints.H}
        dd = self.data_dependent_block_bound()
        if dd is not None:
            out["L_block_data_dependent_max"] = float(dd.max())
            out["rho_observed_max"] = float(self._last_trace.a_bar_max.max())
            out["h_norm_observed_max"] = float(self._last_trace.h_norm.max())
        return out
