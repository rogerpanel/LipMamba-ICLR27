#!/usr/bin/env python
"""TODO 1 / TODO 7 — recompute ℓ* from Eq. (lstar) and re-draw Figure 4's shaded region.

Produces, for the trained constraint values:

  (a) the worst-case ℓ* of Theorem 2 (constants only);
  (b) the per-input ℓ* distribution over observed pre-trigger norms ‖h_{t0}‖
      (Remark 5: "we report its distribution rather than a single value");
  (c) the data-dependent ℓ* using the observed σ_min(Ā_t) and ‖B̄_t x_t‖ along
      the actual trigger tokens (tighter, still a valid bound for that input);
  (d) a sweep over (Δ_max, λ_max) showing what ℓ* each choice certifies, and
      the product Δ_max·λ_max required for a target ℓ*;
  (e) pgfplots coordinates for the shaded region of Figure 4 and a PNG.

Usage
-----
  python scripts/todo1_ell_star.py --config configs/lipmamba_130m.yaml \
      --checkpoint runs/lipmamba_130m/final.pt --data data_cache/robench25.jsonl \
      --t0 512 --alpha 0.5 --out runs/todo1_ell_star.json
  python scripts/todo1_ell_star.py --demo            # tiny random model, no data
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.certificates.constants import ConstraintSet, required_delta_lambda_product_for_ell_star
from lipmamba.certificates.poisoning_immunity import ell_star_distribution, ell_star_from_trace, sweep_ell_star
from lipmamba.utils import load_checkpoint, set_seed


def constants_from_yaml(cfg: dict) -> ConstraintSet:
    m = cfg["model"]
    return ConstraintSet(
        s_b=m.get("s_b", 1.0), s_c=m.get("s_c", 1.0), s_delta=m.get("s_delta", 0.5), s_out=m.get("s_out", 1.0),
        delta_min=m.get("delta_min", 1e-3), delta_max=m.get("delta_max", 0.5),
        lambda_min=m.get("lambda_min", 0.05), lambda_max=m.get("lambda_max", 1.0), x_max=m.get("x_max", 1.0),
    )


@torch.no_grad()
def observed_norms_and_trace(model: LipMambaModel, ids: torch.Tensor, t0: int, layer: int = -1):
    model.eval()
    model(ids)
    tr = list(model.blocks)[layer].ssm.last_trace
    h0 = tr.h_norm[:, t0 - 1]
    return h0, tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--checkpoint"); ap.add_argument("--data")
    ap.add_argument("--t0", type=int, default=None, help="trigger insertion position (default: T/2)")
    ap.add_argument("--alpha", type=float, default=0.5, help="α_min retention threshold")
    ap.add_argument("--target-ell", type=int, default=24)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--out", default="runs/todo1_ell_star.json")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    set_seed(42)

    if args.demo or not args.config:
        cfg_model = LipMambaConfig(vocab_size=256, n_layers=4, d_model=32, d_inner=64, state_dim=8, conv_kernel=3)
        model = LipMambaModel(cfg_model)
        cs = model.constraints
        ids = torch.randint(0, 256, (64, 64))
        provenance = "DEMO: random 4-layer model, random tokens — numbers illustrate the pipeline only"
    else:
        cfg = yaml.safe_load(Path(args.config).read_text())
        cs = constants_from_yaml(cfg)
        from train import build_model  # noqa: E402
        model = build_model(dict(cfg["model"]))
        if args.checkpoint:
            load_checkpoint(args.checkpoint, model)
        if args.data:
            import numpy as np
            tok = np.fromfile(args.data, dtype=np.int32) if args.data.endswith(".bin") else None
            if tok is None:
                raise SystemExit("For RoBench-25 prompts tokenise to a .bin first (see docs/REPRODUCIBILITY.md)")
            T = 1024
            n = min(64, tok.size // T)
            ids = torch.from_numpy(tok[: n * T].reshape(n, T)).long()
        else:
            ids = torch.randint(0, cfg["model"]["vocab_size"], (32, 256))
        provenance = f"config={args.config} checkpoint={args.checkpoint} data={args.data}"

    t0 = args.t0 or ids.size(1) // 2
    h0, tr = observed_norms_and_trace(model, ids, t0, args.layer)

    worst = cs.summary(n_blocks=getattr(model.cfg, "n_layers", 24), h0_norm=4.0, alpha_min=args.alpha)
    dist = ell_star_distribution(cs, h0, alpha_min=args.alpha)
    dd = ell_star_from_trace(tr.a_bar_min, tr.injection_norm, tr.h_norm, t0=t0, alpha_min=args.alpha)
    dd_np = dd[torch.isfinite(dd)]
    sweep = sweep_ell_star(cs, h0_norm=float(h0.median()), alpha_min=args.alpha)
    req = required_delta_lambda_product_for_ell_star(args.target_ell, args.alpha, kappa=0.0)

    result = {
        "provenance": provenance,
        "constants": {k: v for k, v in worst.items() if k in ("s_b", "delta_min", "delta_max", "lambda_min", "lambda_max", "x_max", "c", "rho_min", "rho_max", "H")},
        "alpha_min": args.alpha, "t0": t0,
        "worst_case_ell_star_at_h0_4": worst["ell_star"],
        "per_input_worst_case": {k: v for k, v in dist.items() if k != "values"},
        "per_input_data_dependent": {
            "n": int(dd_np.numel()),
            "min": float(dd_np.min()) if dd_np.numel() else None,
            "p05": float(dd_np.quantile(0.05)) if dd_np.numel() else None,
            "median": float(dd_np.median()) if dd_np.numel() else None,
            "p95": float(dd_np.quantile(0.95)) if dd_np.numel() else None,
            "observed_rho_min": float(tr.a_bar_min[:, t0:].min()),
            "observed_injection_max": float(tr.injection_norm[:, t0:].max()),
        },
        "required_delta_max_times_lambda_max_for_target": {"target_ell": args.target_ell, "product": req,
                                                            "current_product": cs.delta_max * cs.lambda_max},
        "sweep": sweep,
        "figure4_shaded_region": {
            "worst_case_ell_star_int": dist["ell_star_int_min"],
            "data_dependent_p05": float(dd_np.quantile(0.05)) if dd_np.numel() else None,
            "pgfplots": (
                f"\\addplot[fill=green!15,draw=none] coordinates {{(0,0) (0,100) "
                f"({max(dist['ell_star_int_min'], 0)},100) ({max(dist['ell_star_int_min'], 0)},0)}} \\closedcycle; "
                "% worst-case Theorem-2 region (constants only)"
            ),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "sweep"}, indent=2))
    print("\nℓ* sweep (median observed ‖h_t0‖ = %.3f):" % float(h0.median()))
    for r in sweep:
        print(f"  Δmax={r['delta_max']:<5} λmax={r['lambda_max']:<5} ρmin={r['rho_min']:.4f}  ℓ*={r['ell_star']:6.2f}  L_block={r['L_block']:.2e}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(dist["values"], bins=30, alpha=0.6, label="worst-case ℓ* (per input)")
        if dd_np.numel():
            ax.hist(dd_np.numpy(), bins=30, alpha=0.6, label="data-dependent ℓ* (per input)")
        ax.axvline(args.target_ell, ls="--", c="r", label=f"ℓ={args.target_ell} (old shaded boundary)")
        ax.set_xlabel("certified trigger length ℓ*"); ax.set_ylabel("count"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(Path(args.out).with_suffix(".png"), dpi=150)
        print("figure:", Path(args.out).with_suffix(".png"))
    except Exception as e:  # pragma: no cover
        print("matplotlib unavailable:", e)


if __name__ == "__main__":
    main()
