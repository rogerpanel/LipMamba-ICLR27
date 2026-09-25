"""Full LipMamba language / classification model."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..certificates.constants import ConstraintSet
from .glorot_head import GloroNetHead
from .lipmamba_block import LipMambaBlock, LipMambaBlockConfig


@dataclass
class LipMambaConfig:
    """Top-level model configuration.

    ============   ========   =========   ========
    Variant        n_layers   d_model     d_inner
    ============   ========   =========   ========
    LipMamba-130M  24         768         1536
    LipMamba-370M  48         1024        2048
    LipMamba-1.3B  48         2048        4096
    ============   ========   =========   ========

    These match the public ``state-spaces/mamba-{130m,370m,1.4b}`` shapes so
    that a base checkpoint can be converted (scripts/todo3_init_from_hf_mamba.py).
    """

    vocab_size: int = 50280
    n_layers: int = 24
    d_model: int = 768
    d_inner: int = 1536
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
    sn_in_proj: bool = False
    gloro_head: bool = True

    # heads
    n_classes: int = 0
    s_head: float = 1.0
    epsilon_train: float = 0.18
    extras: dict = field(default_factory=dict)

    @classmethod
    def lipmamba_130m(cls, **o) -> "LipMambaConfig":
        return cls(**{**dict(n_layers=24, d_model=768, d_inner=1536), **o})

    @classmethod
    def lipmamba_370m(cls, **o) -> "LipMambaConfig":
        return cls(**{**dict(n_layers=48, d_model=1024, d_inner=2048), **o})

    @classmethod
    def lipmamba_1300m(cls, **o) -> "LipMambaConfig":
        return cls(**{**dict(n_layers=48, d_model=2048, d_inner=4096), **o})

    def block_config(self) -> LipMambaBlockConfig:
        return LipMambaBlockConfig(
            d_model=self.d_model, d_inner=self.d_inner, state_dim=self.state_dim,
            conv_kernel=self.conv_kernel, s_b=self.s_b, s_c=self.s_c, s_delta=self.s_delta,
            s_out=self.s_out, delta_min=self.delta_min, delta_max=self.delta_max,
            lambda_min=self.lambda_min, lambda_max=self.lambda_max, x_max=self.x_max,
            n_power_iters=self.n_power_iters, track_lipschitz=self.track_lipschitz,
            residual=self.residual, clamp_delta=self.clamp_delta,
            reparam_eigen=self.reparam_eigen, spectral_norm=self.spectral_norm,
            sn_in_proj=self.sn_in_proj,
        )

    def constraints(self) -> ConstraintSet:
        return self.block_config().ssm_config().constraints()


class LipMambaModel(nn.Module):
    def __init__(self, cfg: LipMambaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(LipMambaBlock(cfg.block_config()) for _ in range(cfg.n_layers))
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        if cfg.n_classes > 0:
            self.cls_head: GloroNetHead | None = GloroNetHead(
                cfg.d_model, cfg.n_classes, s_head=cfg.s_head,
                epsilon_train=cfg.epsilon_train, spectral_norm=cfg.gloro_head,
            )
        else:
            self.cls_head = None
        self.register_buffer("_verbalizer", None, persistent=False)

    # -- forward -----------------------------------------------------------
    def encode_from_embeddings(self, h: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            h = blk(h)
        return self.norm_f(h)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_from_embeddings(self.embed_tokens(input_ids))

    # -- verbalizer (RoBench-25 true/false answering with the LM head) ------
    def set_verbalizer(self, token_ids: list[int] | None) -> None:
        """Restrict the LM head to ``token_ids`` (e.g. the ids of " true" and
        " false") so that the model acts as a K-way classifier through its
        language-modelling head.  Used for RoBench-25, where the answer is
        read off the next-token logits after the question prompt.  The
        GloRo radius, LL-Acc and the margin attack then operate on these
        K logits.  ``None`` clears the verbalizer."""
        dev = self.embed_tokens.weight.device
        self._verbalizer = None if token_ids is None else torch.as_tensor(token_ids, dtype=torch.long, device=dev)

    def _head(self, h_last: torch.Tensor) -> torch.Tensor:
        if self.cls_head is not None:
            return self.cls_head(h_last)
        logits = self.lm_head(h_last)
        v = getattr(self, "_verbalizer", None)
        return logits[:, v] if v is not None else logits

    def logits_from_embeddings(self, emb: torch.Tensor) -> torch.Tensor:
        """Differentiable path embeddings → classification logits (attacks / L_loc)."""
        return self._head(self.encode_from_embeddings(emb)[:, -1])

    def forward(self, input_ids: torch.Tensor, return_logits: bool = True) -> dict[str, torch.Tensor]:
        h = self.encode(input_ids)
        out = {"hidden_states": h}
        if return_logits:
            out["lm_logits"] = self.lm_head(h)
        if self.cls_head is not None or getattr(self, "_verbalizer", None) is not None:
            out["cls_logits"] = self._head(h[:, -1])
        return out

    # -- certificates --------------------------------------------------------
    @property
    def constraints(self) -> ConstraintSet:
        return self.cfg.constraints()

    def network_lipschitz_bound(self, **_ignored) -> torch.Tensor:
        """Worst-case Theorem-1 product L_SSM ≤ ∏ L_block (× s_head)."""
        l = self.constraints.l_network(self.cfg.n_layers, residual=self.cfg.residual)
        if self.cls_head is not None:
            l *= float(self.cls_head.s_head)
        return torch.tensor(l)

    def log10_network_lipschitz_bound(self) -> float:
        c = self.constraints
        per = math.log10((1.0 + c.l_block) if self.cfg.residual else c.l_block)
        extra = math.log10(self.cls_head.s_head) if self.cls_head is not None else 0.0
        return self.cfg.n_layers * per + extra

    @torch.no_grad()
    def data_dependent_network_bound(self) -> torch.Tensor | None:
        """Product of Algorithm-1 data-dependent block bounds from the last forward."""
        prod = None
        for blk in self.blocks:
            b = blk.data_dependent_block_bound()
            if b is None:
                return None
            b = (1.0 + b) if self.cfg.residual else b
            prod = b if prod is None else prod * b
        if self.cls_head is not None:
            prod = prod * float(self.cls_head.s_head)
        return prod

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
