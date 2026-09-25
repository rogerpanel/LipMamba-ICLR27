#!/usr/bin/env python
"""Print every closed-form constant of the manuscript for a config (or the
paper defaults).  Use it to check a proposed constraint set *before* training.

    python scripts/report_constants.py
    python scripts/report_constants.py --config configs/lipmamba_130m.yaml --h0 4 --alpha 0.5
    python scripts/report_constants.py --delta-min 0.25 --lambda-min 1.0 --delta-max 0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from lipmamba.certificates.constants import ConstraintSet, required_delta_lambda_product_for_ell_star


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--n-blocks", type=int, default=24)
    ap.add_argument("--h0", type=float, default=4.0); ap.add_argument("--alpha", type=float, default=0.5)
    for k, d in (("s-b", 1.0), ("s-c", 1.0), ("s-delta", 0.5), ("s-out", 1.0), ("delta-min", 1e-3),
                 ("delta-max", 0.5), ("lambda-min", 0.05), ("lambda-max", 1.0), ("x-max", 1.0)):
        ap.add_argument(f"--{k}", type=float, default=None)
    a = ap.parse_args()
    kw = {}
    if a.config:
        m = yaml.safe_load(Path(a.config).read_text())["model"]
        kw = {k: m[k] for k in ("s_b", "s_c", "s_delta", "s_out", "delta_min", "delta_max", "lambda_min", "lambda_max", "x_max") if k in m}
    for k in ("s_b", "s_c", "s_delta", "s_out", "delta_min", "delta_max", "lambda_min", "lambda_max", "x_max"):
        v = getattr(a, k)
        if v is not None:
            kw[k] = v
    cs = ConstraintSet(**kw)
    s = cs.summary(n_blocks=a.n_blocks, h0_norm=a.h0, alpha_min=a.alpha)
    s["required_delta_max_lambda_max_for_ell_star_24"] = required_delta_lambda_product_for_ell_star(24, a.alpha)
    print(json.dumps(s, indent=2))
    print("\nReadings:")
    print(f"  Lemma 1  H = {cs.H:.4g}   (state bound; 1/(1-ρ_max) = {1/(1-cs.rho_max):.4g})")
    print(f"  Thm 1    L_block = {cs.l_block:.4g}   → {a.n_blocks} blocks: 10^{s['log10_L_network']:.1f}")
    print(f"  Thm 2    ℓ* = {s['ell_star']:.3f} at ‖h_t0‖={a.h0}, α={a.alpha}   (need Δmax·λmax ≤ {s['required_delta_max_lambda_max_for_ell_star_24']:.4f} for ℓ*=24; now {cs.delta_max*cs.lambda_max:.3f})")


if __name__ == "__main__":
    main()
