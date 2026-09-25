"""LipMamba block — selective SSM wrapped with bounded gating + projections.

    x_in ── LayerNorm ─┬─► W_x↑ ─► Conv1d ─► SiLU ─► [Π_Xmax] ─► SelectiveSSM ─┐
                      └─► W_z↑ ─► SiLU ─────────────────────────────(gate)────⊙
                                                                              │
                                                              W_out↓ ─► (+ residual)

Π_Xmax is the projection onto the X_max ball (Assumption 1); it lives inside
SelectiveSSM so the theorem's bounded-input domain holds by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..certificates.constants import ConstraintSet
from .selective_ssm import SelectiveSSM, SSMConfig
from .spectral_norm import SpectralNormLinear


@dataclass
class LipMambaBlockConfig:
    d_model: int = 1024
    d_inner: int = 2048
    state_dim: int = 16
    conv_kernel: int = 4
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
    residual: bool = True
    # ablations
    clamp_delta: bool = True
    reparam_eigen: bool = True
    spectral_norm: bool = True
    sn_in_proj: bool = False   # True = "Naive SN" (every matrix); False = LipMamba-Arch

    def ssm_config(self) -> SSMConfig:
        return SSMConfig(
            d_model=self.d_model, d_inner=self.d_inner, state_dim=self.state_dim,
            s_b=self.s_b, s_c=self.s_c, s_delta=self.s_delta, s_out=self.s_out,
            delta_min=self.delta_min, delta_max=self.delta_max,
            lambda_min=self.lambda_min, lambda_max=self.lambda_max, x_max=self.x_max,
            n_power_iters=self.n_power_iters, track_lipschitz=self.track_lipschitz,
            clamp_delta=self.clamp_delta, reparam_eigen=self.reparam_eigen,
            spectral_norm=self.spectral_norm,
        )


class LipMambaBlock(nn.Module):
    def __init__(self, cfg: LipMambaBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        if cfg.spectral_norm and cfg.sn_in_proj:
            self.in_proj_x = SpectralNormLinear(cfg.d_model, cfg.d_inner, s=1.0, bias=False,
                                                n_power_iters=cfg.n_power_iters)
            self.in_proj_z = SpectralNormLinear(cfg.d_model, cfg.d_inner, s=1.0, bias=False,
                                                n_power_iters=cfg.n_power_iters)
        else:
            self.in_proj_x = nn.Linear(cfg.d_model, cfg.d_inner, bias=False)
            self.in_proj_z = nn.Linear(cfg.d_model, cfg.d_inner, bias=False)
        if cfg.spectral_norm:
            self.out_proj = SpectralNormLinear(cfg.d_inner, cfg.d_model, s=cfg.s_out, bias=False,
                                               n_power_iters=cfg.n_power_iters)
        else:
            self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False)

        self.conv = nn.Conv1d(cfg.d_inner, cfg.d_inner, kernel_size=cfg.conv_kernel,
                              groups=cfg.d_inner, padding=cfg.conv_kernel - 1, bias=True)
        self.ssm = SelectiveSSM(cfg.ssm_config())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_n = self.norm(x)
        x_proj = self.in_proj_x(x_n)
        z_proj = self.in_proj_z(x_n)
        t = x_proj.size(1)
        xc = F.silu(self.conv(x_proj.transpose(1, 2))[..., :t].transpose(1, 2))
        ys = self.ssm(xc) * F.silu(z_proj)
        out = self.out_proj(ys)
        return residual + out if self.cfg.residual else out

    # -- certificates ------------------------------------------------------
    @property
    def constraints(self) -> ConstraintSet:
        return self.ssm.constraints

    def block_lipschitz_bound(self, **_ignored) -> torch.Tensor:
        return self.ssm.block_lipschitz_bound()

    def data_dependent_block_bound(self) -> torch.Tensor | None:
        return self.ssm.data_dependent_block_bound()

    @torch.no_grad()
    def lipschitz_state(self) -> dict[str, float]:
        return self.ssm.lipschitz_state()
