"""Load a public Mamba checkpoint into this code base, with or without the constraints.

The public checkpoints (``state-spaces/mamba-130m-hf`` and siblings) use

    Δ_t = softplus(W_dt W_x[:r] x_t + b_dt),   A = -exp(A_log),   RMSNorm,

with a rank-``r`` step projection.  Here the step projection is one d x d
matrix, so it is initialised with the product ``W_dt W_x[:r]``; every other
tensor is copied.  With the ``unconstrained_mamba`` preset the converted
model computes the SAME function as the public checkpoint
(:func:`reference_logits` and the ``transformers`` implementation are used
to check this).  With the constraints switched on, the same weights are the
starting point and the constraints act as a projection:

* eigenvalues ``exp(A_log)`` are clipped into ``[lambda_min, lambda_max]``;
* spectral normalisation rescales W_B, W_C, W_Δ, W_out in the forward pass;
* the step is clamped into ``[delta_min, delta_max)``;
* inputs of every scan are projected onto a ball whose radius is calibrated
  per block on clean data (:func:`calibrate_x_max`) or set by hand.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .baselines import preset
from .models.lipmamba_model import LipMambaConfig, LipMambaModel
from .models.spectral_norm import SpectralNormLinear

__all__ = [
    "ARCHS", "hf_state_dict", "tiny_state_dict", "infer_shapes", "build_from_mamba",
    "calibrate_x_max", "calibrate_constraints", "apply_calibration", "set_x_max", "refresh_spectral_norms", "reference_logits", "conversion_error",
]

# name -> (preset, overrides).  All arms share one code path.
ARCHS: dict[str, tuple[str, dict]] = {
    "mamba":        ("unconstrained_mamba", {}),
    "lip":          ("lipmamba_arch", {}),
    "lip_no_sn":    ("lipmamba_arch", {"spectral_norm": False}),
    "lip_no_clamp": ("lipmamba_arch", {"clamp_delta": False}),
    "lip_free_A":   ("lipmamba_arch", {"reparam_eigen": False}),
    "lip_no_clip":  ("lipmamba_arch", {"clip_input": False}),
}


def _norm_keys(sd: dict, dtype=torch.float32) -> dict:
    """Accept both the original (``backbone.embedding``) and the HF (``backbone.embeddings``) key names."""
    out = {}
    for k, v in sd.items():
        k = k.replace("backbone.embedding.weight", "backbone.embeddings.weight")
        out[k] = v.detach().to(dtype).cpu()
    return out


def hf_state_dict(name: str = "state-spaces/mamba-130m-hf", revision: str | None = None):
    """Download a public checkpoint.  Returns ``(state_dict, hf_model)``; the HF model is kept only for verification."""
    from transformers import MambaForCausalLM

    hf = MambaForCausalLM.from_pretrained(name, revision=revision).eval()
    return _norm_keys(hf.state_dict()), hf


def tiny_state_dict(vocab_size: int, d_model: int = 48, n_layers: int = 2, state_dim: int = 8,
                    dt_rank: int = 4, expand: int = 2, conv_kernel: int = 4, seed: int = 0) -> dict:
    """A random checkpoint in the public key layout (CPU smoke tests; no download)."""
    g = torch.Generator().manual_seed(seed)
    d_in = expand * d_model
    r = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc
    sd = {"backbone.embeddings.weight": r(vocab_size, d_model, sc=0.05), "backbone.norm_f.weight": torch.ones(d_model)}
    for i in range(n_layers):
        p = f"backbone.layers.{i}."
        sd[p + "norm.weight"] = torch.ones(d_model)
        sd[p + "mixer.in_proj.weight"] = r(2 * d_in, d_model, sc=d_model ** -0.5)
        sd[p + "mixer.conv1d.weight"] = r(d_in, 1, conv_kernel, sc=0.5)
        sd[p + "mixer.conv1d.bias"] = r(d_in, sc=0.1)
        sd[p + "mixer.x_proj.weight"] = r(dt_rank + 2 * state_dim, d_in, sc=d_in ** -0.5)
        sd[p + "mixer.dt_proj.weight"] = r(d_in, dt_rank, sc=dt_rank ** -0.5)
        sd[p + "mixer.dt_proj.bias"] = r(d_in, sc=0.5) - 2.0
        sd[p + "mixer.A_log"] = torch.log(torch.arange(1, state_dim + 1).float()).repeat(d_in, 1)
        sd[p + "mixer.D"] = torch.ones(d_in)
        sd[p + "mixer.out_proj.weight"] = r(d_model, d_in, sc=d_in ** -0.5)
    return sd


def infer_shapes(sd: dict) -> dict:
    sd = _norm_keys(sd)
    vocab, d_model = sd["backbone.embeddings.weight"].shape
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("backbone.layers."))
    d_inner, state_dim = sd["backbone.layers.0.mixer.A_log"].shape
    return dict(vocab_size=vocab, d_model=d_model, n_layers=n_layers, d_inner=d_inner, state_dim=state_dim,
                conv_kernel=sd["backbone.layers.0.mixer.conv1d.weight"].shape[-1],
                dt_rank=sd["backbone.layers.0.mixer.dt_proj.weight"].shape[1])


def build_from_mamba(sd: dict, arch: str = "mamba", x_max: list[float] | float | None = None,
                     calibration: dict | None = None, **overrides) -> LipMambaModel:
    """Build the ``arch`` variant and initialise it from a public Mamba state dict.

    ``overrides`` are :class:`LipMambaConfig` fields (e.g. ``delta_max=1.0``).
    ``x_max``: per-block radii, one radius, or ``None`` to keep the config value.
    ``calibration``: per-block constants from :func:`calibrate_constraints`; applied before the weights are
    projected, so the eigenvalues are projected into each block's own interval.
    """
    sd = _norm_keys(sd)
    shp = infer_shapes(sd)
    r = shp.pop("dt_rank")
    name, flags = ARCHS[arch]
    cfg: LipMambaConfig = preset(name, **shp, norm_type="rms", track_lipschitz=False, n_classes=0,
                                 **{**flags, **overrides})
    model = LipMambaModel(cfg)
    if calibration is not None:
        apply_calibration(model, calibration)
    n = cfg.state_dim
    with torch.no_grad():
        model.embed_tokens.weight.copy_(sd["backbone.embeddings.weight"])
        model.norm_f.weight.copy_(sd["backbone.norm_f.weight"])
        for i, blk in enumerate(model.blocks):
            p = f"backbone.layers.{i}."
            m = lambda k: sd[p + "mixer." + k]
            blk.norm.weight.copy_(sd[p + "norm.weight"])
            w_in = m("in_proj.weight")
            blk.in_proj_x.weight.copy_(w_in[: cfg.d_inner]); blk.in_proj_z.weight.copy_(w_in[cfg.d_inner:])
            blk.conv.weight.copy_(m("conv1d.weight")); blk.conv.bias.copy_(m("conv1d.bias"))
            blk.out_proj.weight.copy_(m("out_proj.weight"))
            xp = m("x_proj.weight")
            s = blk.ssm
            s.delta_proj.proj.weight.copy_(m("dt_proj.weight") @ xp[:r])
            s.delta_proj.proj.bias.copy_(m("dt_proj.bias"))
            s.x_to_b.weight.copy_(xp[r: r + n]); s.x_to_c.weight.copy_(xp[r + n: r + 2 * n])
            if s.x_to_b.bias is not None:
                s.x_to_b.bias.zero_(); s.x_to_c.bias.zero_()
            s.D.copy_(m("D"))
            a_log = m("A_log")
            if cfg.reparam_eigen:        # project the eigenvalues into the admissible interval, away from saturation
                u = (a_log.double().exp() - s.A.lambda_min) / (s.A.lambda_max - s.A.lambda_min)
                if calibration is not None and "lambda_max" in calibration:
                    # every eigenvalue lies inside the calibrated interval: keep it exactly.  (Clamping u away
                    # from 0 by even 1e-3 of the range would move the slow eigenvalues, the model's long memory,
                    # up by orders of magnitude.)
                    u = u.clamp(1e-12, 1 - 1e-9)
                else:                    # paper constants: most eigenvalues are outside; stay off saturation
                    u = u.clamp(0.02, 0.98)
                s.A.alpha.copy_(torch.logit(u).float())
            else:
                s.A.alpha.copy_(a_log)
    if x_max is not None:
        set_x_max(model, x_max)
    refresh_spectral_norms(model)
    return model


def set_x_max(model: LipMambaModel, x_max: list[float] | float) -> None:
    """Set the input-projection radius of every block; the config keeps the largest (used by the constants)."""
    xs = [float(x_max)] * len(model.blocks) if isinstance(x_max, (int, float)) else [float(v) for v in x_max]
    for blk, v in zip(model.blocks, xs, strict=True):
        blk.ssm.input_clip.x_max = v
    model.cfg.x_max = max(xs)
    for blk in model.blocks:
        blk.cfg.x_max = blk.ssm.cfg.x_max = max(xs)


@torch.no_grad()
def calibrate_constraints(model: LipMambaModel, batches: list[torch.Tensor], q_x: float = 0.999,
                          q_delta: float = 0.999, delta_headroom: float = 2.0, margin: float = 1.25) -> dict:
    """Per-block constraint constants at the clean-data envelope of the public checkpoint.

    Run on the UNCONSTRAINED converted model.  With these constants the constrained model starts as (almost)
    the same function as the checkpoint, and every constraint is a bound the attacker and the fine-tuning
    cannot exceed:

    * spectral bounds s_B, s_C, s_Δ, s_out = the checkpoint's own spectral norms;
    * eigenvalue interval = [min/margin, max*margin] of the checkpoint's exp(A_log);
    * step interval: Δ_max = ``delta_headroom`` times the ``q_delta`` quantile of the clean step sizes (the
      smooth clamp compresses steps near Δ_max, so some headroom is needed to keep clean steps unchanged),
      Δ_min = a tenth of their 0.001 quantile (at most 1e-3);
    * input radius X_max = the ``q_x`` quantile of the clean scan-input norms.
    """
    import torch.nn.functional as F_
    nb = len(model.blocks)
    xs: list[list[torch.Tensor]] = [[] for _ in range(nb)]
    ds: list[list[torch.Tensor]] = [[] for _ in range(nb)]
    hooks = []
    for i, blk in enumerate(model.blocks):
        hooks.append(blk.ssm.register_forward_pre_hook(lambda _m, a, i=i: xs[i].append(a[0].norm(dim=-1).flatten().float().cpu())))
        hooks.append(blk.ssm.delta_proj.proj.register_forward_hook(
            lambda _m, _a, out, i=i: ds[i].append(F_.softplus(out).flatten()[::7].float().cpu())))
    was = model.training
    model.eval()
    for ids in batches:
        model.encode(ids)
    for h in hooks:
        h.remove()
    model.train(was)
    sig = lambda w: float(torch.linalg.matrix_norm(w.detach().double(), ord=2))
    qt = lambda v, q: float(torch.quantile(torch.cat(v)[:4_000_000], q))
    cal: dict = {k: [] for k in ("x_max", "delta_min", "delta_max", "lambda_min", "lambda_max", "s_b", "s_c", "s_delta", "s_out")}
    for i, blk in enumerate(model.blocks):
        s = blk.ssm
        lam = s.A.lambdas
        d_hi = delta_headroom * qt(ds[i], q_delta)
        cal["x_max"].append(qt(xs[i], q_x))
        cal["delta_max"].append(d_hi); cal["delta_min"].append(min(1e-3, 0.1 * qt(ds[i], 0.001), 0.01 * d_hi))
        cal["lambda_min"].append(float(lam.min()) / margin); cal["lambda_max"].append(float(lam.max()) * margin)
        cal["s_b"].append(sig(s.x_to_b.weight)); cal["s_c"].append(sig(s.x_to_c.weight))
        cal["s_delta"].append(sig(s.delta_proj.proj.weight)); cal["s_out"].append(sig(blk.out_proj.weight))
    return cal


def apply_calibration(model: LipMambaModel, cal: dict) -> None:
    """Set per-block constants (any subset of the keys of :func:`calibrate_constraints`) on the constraints the
    model actually has.  The configs keep the worst case over blocks, which is what the closed-form constants use."""
    cfg = model.cfg
    for i, blk in enumerate(model.blocks):
        s = blk.ssm
        if "x_max" in cal:
            s.input_clip.x_max = float(cal["x_max"][i])
        if cfg.clamp_delta and "delta_max" in cal:
            s.delta_proj.delta_min, s.delta_proj.delta_max = float(cal["delta_min"][i]), float(cal["delta_max"][i])
        if cfg.reparam_eigen and "lambda_max" in cal:
            s.A.lambda_min, s.A.lambda_max = float(cal["lambda_min"][i]), float(cal["lambda_max"][i])
        if cfg.spectral_norm and "s_b" in cal:
            s.x_to_b.s, s.x_to_c.s = float(cal["s_b"][i]), float(cal["s_c"][i])
            s.delta_proj.proj.s = s.delta_proj.s_delta = float(cal["s_delta"][i]); blk.out_proj.s = float(cal["s_out"][i])
    worst = {k: (min if k.endswith("_min") else max)(v) for k, v in cal.items()}
    for c in [cfg] + [b.cfg for b in model.blocks] + [b.ssm.cfg for b in model.blocks]:
        for k, v in worst.items():
            setattr(c, k, float(v))


@torch.no_grad()
def calibrate_x_max(model: LipMambaModel, batches: list[torch.Tensor], q: float = 0.99) -> list[float]:
    """Per-block ``q``-quantile of the scan-input norm ‖x_t‖₂ on clean token batches.

    Run it on the UNCONSTRAINED converted model, so the radius reflects the
    public checkpoint's own activation scale and the projection is inactive
    on about ``q`` of clean tokens at initialisation."""
    norms: list[list[torch.Tensor]] = [[] for _ in model.blocks]
    hooks = [blk.ssm.register_forward_pre_hook(lambda _m, a, i=i: norms[i].append(a[0].norm(dim=-1).flatten().float().cpu()))
             for i, blk in enumerate(model.blocks)]
    was = model.training
    model.eval()
    for ids in batches:
        model.encode(ids)
    for h in hooks:
        h.remove()
    model.train(was)
    return [float(torch.quantile(torch.cat(v)[:2_000_000], q)) for v in norms]


@torch.no_grad()
def refresh_spectral_norms(model: torch.nn.Module) -> None:
    """Set every spectral-norm layer's running estimate to the exact σ_max.

    In eval mode the layers scale by the running estimate; after conversion or
    training it can lag the true value, which would let ‖W̄‖₂ exceed its bound.
    Call before every evaluation or attack."""
    for mod in model.modules():
        if isinstance(mod, SpectralNormLinear):
            w = mod.weight.detach().double()
            u, s, _ = torch.linalg.svd(w, full_matrices=False)
            mod._sigma_ema.copy_(s[0].to(mod._sigma_ema.dtype))
            mod._u.copy_(u[:, 0].to(mod._u.dtype))


@torch.no_grad()
def reference_logits(sd: dict, ids: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """Vanilla Mamba written directly from the public checkpoint's tensors (no code shared with the model classes)."""
    sd = {k: v.to(ids.device) for k, v in _norm_keys(sd, dtype).items()}
    shp = infer_shapes(sd)
    d_in, n, r, k = shp["d_inner"], shp["state_dim"], shp["dt_rank"], shp["conv_kernel"]
    rms = lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    h = sd["backbone.embeddings.weight"][ids]
    t = ids.size(1)
    for i in range(shp["n_layers"]):
        m = lambda key: sd[f"backbone.layers.{i}.mixer.{key}"]
        xz = rms(h, sd[f"backbone.layers.{i}.norm.weight"]) @ m("in_proj.weight").T
        x, z = xz[..., :d_in], xz[..., d_in:]
        x = F.conv1d(x.transpose(1, 2), m("conv1d.weight"), m("conv1d.bias"), padding=k - 1, groups=d_in)[..., :t]
        x = F.silu(x.transpose(1, 2))
        p = x @ m("x_proj.weight").T
        dt = F.softplus(p[..., :r] @ m("dt_proj.weight").T + m("dt_proj.bias"))
        b, c = p[..., r: r + n], p[..., r + n:]
        a = -m("A_log").exp()
        st = x.new_zeros(x.size(0), d_in, n)
        ys = []
        for tt in range(t):
            st = torch.exp(dt[:, tt, :, None] * a) * st + dt[:, tt, :, None] * b[:, tt, None, :] * x[:, tt, :, None]
            ys.append((st * c[:, tt, None, :]).sum(-1))
        y = torch.stack(ys, 1) + x * m("D")
        h = h + (y * F.silu(z)) @ m("out_proj.weight").T
    return rms(h, sd["backbone.norm_f.weight"]) @ sd["backbone.embeddings.weight"].T


def describe(model: LipMambaModel) -> dict:
    """Constants of the built model, per block, for the result files."""
    cfg = model.cfg
    out = dict(params=sum(p.numel() for p in model.parameters()), n_layers=cfg.n_layers, d_model=cfg.d_model,
               spectral_norm=cfg.spectral_norm, clamp_delta=cfg.clamp_delta, reparam_eigen=cfg.reparam_eigen,
               clip_input=cfg.clip_input)
    if cfg.clip_input:
        out["x_max"] = [b.ssm.input_clip.x_max for b in model.blocks]
    if cfg.clamp_delta:
        out["delta_min"] = [b.ssm.delta_proj.delta_min for b in model.blocks]
        out["delta_max"] = [b.ssm.delta_proj.delta_max for b in model.blocks]
    if cfg.reparam_eigen:
        out["lambda_min"] = [b.ssm.A.lambda_min for b in model.blocks]
        out["lambda_max"] = [b.ssm.A.lambda_max for b in model.blocks]
    if cfg.spectral_norm:
        out.update(s_b=[b.ssm.x_to_b.s for b in model.blocks], s_c=[b.ssm.x_to_c.s for b in model.blocks],
                   s_delta=[b.ssm.delta_proj.proj.s for b in model.blocks], s_out=[b.out_proj.s for b in model.blocks])
    if cfg.clamp_delta and cfg.reparam_eigen:    # per-token retention factor guaranteed by the clamp, per block
        out["rho_min"] = [math.exp(-dm * lm) for dm, lm in zip(out["delta_max"], out["lambda_max"])]
        out["rho_max"] = [math.exp(-dm * lm) for dm, lm in zip(out["delta_min"], out["lambda_min"])]
    return out


@torch.no_grad()
def conversion_error(model: LipMambaModel, sd: dict, ids: torch.Tensor) -> float:
    """Max relative logit difference between the converted UNCONSTRAINED model and the checkpoint's function,
    both evaluated in float64.  Single precision cannot decide this: over 24 blocks, rounding alone moves the
    logits of the public implementation by about 1e-3 (it differs from itself by that much between float32
    and float64).  In float64 a correct conversion agrees to about 1e-10; a wrong one differs at order one.
    The model is left in float32 on return."""
    r = infer_shapes(sd)["dt_rank"]
    model.double()
    for i, blk in enumerate(model.blocks):       # the step matrix as an exact float64 product
        p = f"backbone.layers.{i}.mixer."
        w = sd[p + "dt_proj.weight"].double() @ sd[p + "x_proj.weight"][:r].double()
        blk.ssm.delta_proj.proj.weight.copy_(w.to(ids.device))
    ours = model.lm_head(model.encode(ids))
    ref = reference_logits(sd, ids, dtype=torch.float64)
    model.float()
    return float((ours - ref).abs().max() / ref.abs().max())
