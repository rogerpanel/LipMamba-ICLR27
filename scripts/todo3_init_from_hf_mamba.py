#!/usr/bin/env python
"""TODO 3 — state the pre-training corpus and the base checkpoints.

The manuscript's perplexity overhead (Figure 3) is uninterpretable without a
named base model and corpus.  This script fixes the provenance by
*constructing* LipMamba-130M / 370M from the public Mamba checkpoints

    state-spaces/mamba-130m   (24 layers, d_model 768,  trained on The Pile, GPT-NeoX tokenizer)
    state-spaces/mamba-370m   (48 layers, d_model 1024, trained on The Pile, GPT-NeoX tokenizer)

by mapping every tensor into the constrained parameterisation and projecting
it onto the constraint set:

    in_proj.weight[:d_inner]      → in_proj_x.weight
    in_proj.weight[d_inner:]      → in_proj_z.weight
    conv1d.weight / bias          → conv.weight / bias
    x_proj.weight[dt_rank:+N]     → x_to_b.weight     (B rows)
    x_proj.weight[dt_rank+N:]     → x_to_c.weight     (C rows)
    dt_proj.weight @ x_proj[:dt_rank] → delta_proj.proj.weight   (low-rank product, (d_inner,d_inner))
    dt_proj.bias                  → delta_proj.proj.bias (τ)
    A_log                         → alpha via σ⁻¹((clip(exp(A_log), λ_min, λ_max) − λ_min)/(λ_max−λ_min))
    D                             → D
    out_proj.weight               → out_proj.weight
    norm.weight (RMSNorm)         → norm.weight (LayerNorm, bias = 0)
    embedding.weight / lm_head    → embed_tokens.weight (tied)

The spectral normalisation is applied lazily by the forward pass (power
iteration), so the copied W_B, W_C, W_Δ, W_out are automatically scaled into
their budgets on first use; the recorded provenance therefore states the
base checkpoint, the constraint budgets and the corpus.

After conversion, evaluate:
    python scripts/perplexity_overhead.py --base state-spaces/mamba-130m --lip runs/lipmamba_130m_init.pt \
        --corpus wikitext103 --split validation
Then fine-tune with scripts/train.py (Algorithm 2) starting from runs/lipmamba_130m_init.pt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.utils import save_checkpoint

BASES = {
    "state-spaces/mamba-130m": dict(n_layers=24, d_model=768, d_inner=1536, vocab_size=50280),
    "state-spaces/mamba-370m": dict(n_layers=48, d_model=1024, d_inner=2048, vocab_size=50280),
    "state-spaces/mamba-1.4b": dict(n_layers=48, d_model=2048, d_inner=4096, vocab_size=50280),
}
CORPUS = {
    "pretraining": "The Pile (Gao et al., 2020) — the corpus the state-spaces/mamba-* checkpoints were trained on",
    "tokenizer": "EleutherAI/gpt-neox-20b",
    "perplexity_eval": "WikiText-103 validation (raw), GPT-NeoX tokenizer, block size 1024",
}


def load_hf_state_dict(model_id: str, revision: str | None) -> dict[str, torch.Tensor]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(model_id, "pytorch_model.bin", revision=revision)
    return torch.load(p, map_location="cpu", weights_only=True)


def sigmoid_inverse(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


@torch.no_grad()
def convert(sd: dict[str, torch.Tensor], cfg: LipMambaConfig) -> tuple[LipMambaModel, dict]:
    model = LipMambaModel(cfg)
    N = cfg.state_dim
    dt_rank = sd["backbone.layers.0.dt_proj.weight"].shape[1]
    stats = {"dt_rank": dt_rank, "layers": cfg.n_layers, "A_clipped_fraction": []}
    model.embed_tokens.weight.copy_(sd["backbone.embedding.weight"])
    model.norm_f.weight.copy_(sd["backbone.norm_f.weight"]); model.norm_f.bias.zero_()
    for i, blk in enumerate(model.blocks):
        p = f"backbone.layers.{i}."
        blk.norm.weight.copy_(sd[p + "norm.weight"]); blk.norm.bias.zero_()
        in_w = sd[p + "mixer.in_proj.weight"]
        blk.in_proj_x.weight.copy_(in_w[: cfg.d_inner]); blk.in_proj_z.weight.copy_(in_w[cfg.d_inner:])
        blk.conv.weight.copy_(sd[p + "mixer.conv1d.weight"]); blk.conv.bias.copy_(sd[p + "mixer.conv1d.bias"])
        xw = sd[p + "mixer.x_proj.weight"]                       # (dt_rank + 2N, d_inner)
        ssm = blk.ssm
        ssm.x_to_b.weight.copy_(xw[dt_rank: dt_rank + N]); ssm.x_to_c.weight.copy_(xw[dt_rank + N: dt_rank + 2 * N])
        if ssm.x_to_b.bias is not None:
            ssm.x_to_b.bias.zero_(); ssm.x_to_c.bias.zero_()
        w_delta = sd[p + "mixer.dt_proj.weight"] @ xw[:dt_rank]  # (d_inner, d_inner)
        ssm.delta_proj.proj.weight.copy_(w_delta); ssm.delta_proj.proj.bias.copy_(sd[p + "mixer.dt_proj.bias"])
        lam = torch.exp(sd[p + "mixer.A_log"])                  # (d_inner, N) = |λ_i|
        clipped = ((lam < cfg.lambda_min) | (lam > cfg.lambda_max)).float().mean().item()
        stats["A_clipped_fraction"].append(clipped)
        lam = lam.clamp(cfg.lambda_min, cfg.lambda_max)
        ssm.A.alpha.copy_(sigmoid_inverse((lam - cfg.lambda_min) / (cfg.lambda_max - cfg.lambda_min)))
        ssm.D.copy_(sd[p + "mixer.D"])
        blk.out_proj.weight.copy_(sd[p + "mixer.out_proj.weight"])
    return model, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="state-spaces/mamba-130m", choices=sorted(BASES))
    ap.add_argument("--revision", default=None, help="HF revision (commit hash) — record it for provenance")
    ap.add_argument("--out", default=None)
    ap.add_argument("--delta-min", type=float, default=1e-3); ap.add_argument("--delta-max", type=float, default=0.5)
    ap.add_argument("--lambda-min", type=float, default=0.05); ap.add_argument("--lambda-max", type=float, default=1.0)
    ap.add_argument("--x-max", type=float, default=1.0)
    args = ap.parse_args()

    cfg = LipMambaConfig(**BASES[args.base], delta_min=args.delta_min, delta_max=args.delta_max,
                         lambda_min=args.lambda_min, lambda_max=args.lambda_max, x_max=args.x_max)
    sd = load_hf_state_dict(args.base, args.revision)
    model, stats = convert(sd, cfg)
    out = args.out or f"runs/{args.base.split('/')[-1].replace('mamba', 'lipmamba')}_init.pt"
    provenance = {"base_model_id": args.base, "base_revision": args.revision or "main (record the commit hash!)",
                  **CORPUS, "constraints": model.constraints.summary(n_blocks=cfg.n_layers),
                  "conversion_stats": {**stats, "A_clipped_fraction_mean": sum(stats["A_clipped_fraction"]) / len(stats["A_clipped_fraction"])}}
    save_checkpoint(model, None, step=0, path=out, extra=provenance)
    Path(out).with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({k: v for k, v in provenance.items() if k != "constraints"}, indent=2))
    print("saved", out)
    print("\nNOTE: fraction of |λ_i| outside [λ_min, λ_max] that were clipped =",
          f"{provenance['conversion_stats']['A_clipped_fraction_mean']:.3f};",
          "a large value means the eigenvalue budget is far from the pretrained spectrum and will cost perplexity.")


if __name__ == "__main__":
    main()
