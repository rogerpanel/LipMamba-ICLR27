"""Baselines required by the audit, expressed as configuration presets of the
same code path (so the comparison is like-for-like) plus randomized
smoothing (Cohen et al., ICML 2019) as an independent certification method.

Presets (Table 1 / Figure 3 of the manuscript)
----------------------------------------------
``unconstrained_mamba``   vanilla S6: softplus Δ, free A, no spectral norm, plain head.
``gloro_mamba``           vanilla S6 + GloRo head + margin training (head only).
``naive_sn_mamba``        spectral norm on *every* matrix incl. in-projections.
``lipmamba_arch``         the paper's model: SN on W_B, W_C, W_Δ, W_out + clamp + eigen reparam.

Ablations (Table 3)
-------------------
``ablate("no_sn")``, ``ablate("no_clamp")``, ``ablate("free_A")``, ``ablate("no_gloro")``.
"""
from __future__ import annotations

from ..models.lipmamba_model import LipMambaConfig
from .randomized_smoothing import RandomizedSmoothing, SmoothingConfig

__all__ = ["preset", "ablate", "PRESETS", "RandomizedSmoothing", "SmoothingConfig"]


def _base(**o) -> LipMambaConfig:
    return LipMambaConfig(**o)


def preset(name: str, **overrides) -> LipMambaConfig:
    if name == "unconstrained_mamba":
        cfg = _base(spectral_norm=False, clamp_delta=False, reparam_eigen=False, gloro_head=False)
    elif name == "gloro_mamba":
        cfg = _base(spectral_norm=False, clamp_delta=False, reparam_eigen=False, gloro_head=True)
    elif name == "naive_sn_mamba":
        cfg = _base(spectral_norm=True, clamp_delta=True, reparam_eigen=True, gloro_head=True,
                    sn_in_proj=True)
    elif name == "lipmamba_arch":
        cfg = _base()
    else:
        raise KeyError(f"unknown preset {name!r}; choose from {PRESETS}")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


PRESETS = ("unconstrained_mamba", "gloro_mamba", "naive_sn_mamba", "lipmamba_arch")


def ablate(which: str, **overrides) -> LipMambaConfig:
    table = {
        "no_sn": dict(spectral_norm=False),
        "no_clamp": dict(clamp_delta=False),
        "free_A": dict(reparam_eigen=False),
        "no_gloro": dict(gloro_head=False),
    }
    if which not in table:
        raise KeyError(f"unknown ablation {which!r}; choose from {sorted(table)}")
    return _base(**{**table[which], **overrides})
