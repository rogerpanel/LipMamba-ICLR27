#!/usr/bin/env python
"""TODO 5 — regenerate Figure 2 (Lipschitz estimate vs depth) honestly.

For depths L ∈ {4, 8, 16, 24} the script computes four curves:

  worst_case      Theorem-1 product ∏ L_block from the constants only        (upper bound, input-free)
  data_dependent  Algorithm-1 product using observed ‖Ā_t‖, ‖h_t‖ on real inputs (upper bound for those inputs)
  op_norm         bare product of ‖W̄_•‖₂ over the spectrally normalised layers  (neither bound; reference)
  empirical_lb    max gradient norm found by PGD (Appendix E)                (LOWER bound)

The audit's objection to the old Figure 2 was that "empirical LB" tracked the
analytical bound within 2%.  A correct plot must show empirical_lb ≪
data_dependent ≤ worst_case.  With the manuscript's constants
(Δ_min = 1e-3, λ_min = 0.05) worst_case is ~10^8 per block, i.e. 10^193 at
depth 24 — not 10^10.  The script prints both so the text can be made
consistent with whichever curve is plotted.

    python scripts/todo5_fig2_lipschitz_depth.py --config configs/lipmamba_130m.yaml \
        --checkpoint runs/lipmamba_130m/final.pt --tokens data_cache/wikitext103_val.bin
    python scripts/todo5_fig2_lipschitz_depth.py --demo
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.certificates.lipschitz import empirical_lipschitz_lower_bound, operator_norm_product
from lipmamba.utils import load_checkpoint, set_seed


def truncated(model: LipMambaModel, depth: int) -> LipMambaModel:
    """Same model with only the first ``depth`` blocks (shares parameters)."""
    import copy
    m = copy.copy(model)
    # nn.Module shallow copies share the _modules dict: give the copy its own
    # so that replacing `blocks` does not truncate the original model.
    m._modules = dict(model._modules)
    m.blocks = torch.nn.ModuleList(list(model.blocks)[:depth])
    m.cfg = copy.copy(model.cfg); m.cfg.n_layers = depth
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--checkpoint"); ap.add_argument("--tokens")
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 8, 16, 24])
    ap.add_argument("--n-inputs", type=int, default=16); ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--attacks", type=int, default=8, help="restarts for the empirical lower bound")
    ap.add_argument("--out", default="runs/todo5_fig2.json"); ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    set_seed(42)

    if args.demo or not args.config:
        model = LipMambaModel(LipMambaConfig(vocab_size=256, n_layers=max(args.depths), d_model=32, d_inner=64,
                                             state_dim=8, conv_kernel=3, n_classes=4))
        ids = torch.randint(0, 256, (args.n_inputs, 32)); prov = "DEMO random model"
    else:
        cfg = yaml.safe_load(Path(args.config).read_text())
        from train import build_model
        cfg["model"].setdefault("n_classes", 2)
        model = build_model(dict(cfg["model"]))
        if args.checkpoint:
            load_checkpoint(args.checkpoint, model)
        import numpy as np
        tok = np.fromfile(args.tokens, dtype=np.int32)
        ids = torch.from_numpy(tok[: args.n_inputs * args.seq].reshape(args.n_inputs, args.seq)).long()
        prov = f"config={args.config} checkpoint={args.checkpoint} tokens={args.tokens}"
    model.eval()
    if torch.cuda.is_available():
        model.cuda(); ids = ids.cuda()

    cs = model.constraints
    rows = []
    for d in args.depths:
        m = truncated(model, d)
        with torch.no_grad():
            m(ids)
            dd = m.data_dependent_network_bound()
        emb = m.embed_tokens(ids).detach()
        lb = empirical_lipschitz_lower_bound(m, emb, n_steps=20, n_restarts=args.attacks, radius=0.3)
        row = {"depth": d,
               "log10_worst_case": m.log10_network_lipschitz_bound(),
               "log10_data_dependent_max": float(torch.log10(dd.max())) if dd is not None else None,
               "log10_data_dependent_median": float(torch.log10(dd.median())) if dd is not None else None,
               "log10_op_norm_product": math.log10(max(operator_norm_product(m), 1e-300)),
               "log10_empirical_lb_max": float(torch.log10(lb.max().clamp_min(1e-300))),
               "log10_empirical_lb_median": float(torch.log10(lb.median().clamp_min(1e-300)))}
        rows.append(row)
        print(f"L={d:2d}  log10: worst={row['log10_worst_case']:7.1f}  data-dep={row['log10_data_dependent_max']:7.2f}  "
              f"op-norm={row['log10_op_norm_product']:6.2f}  empirical-LB={row['log10_empirical_lb_max']:6.2f}")

    gap_ok = all(r["log10_empirical_lb_max"] < r["log10_data_dependent_max"] - 0.5 for r in rows if r["log10_data_dependent_max"] is not None)
    out = {"provenance": prov, "constants": cs.summary(n_blocks=max(args.depths)), "rows": rows,
           "consistency_check": {"empirical_lb_well_below_bound": gap_ok,
                                 "note": "If the empirical LB is within ~2% of an upper bound the plot is wrong (TODO 5)."},
           "pgfplots": {
               "worst_case": " ".join(f"({r['depth']},{r['log10_worst_case']:.2f})" for r in rows),
               "data_dependent": " ".join(f"({r['depth']},{r['log10_data_dependent_max']:.2f})" for r in rows),
               "op_norm": " ".join(f"({r['depth']},{r['log10_op_norm_product']:.2f})" for r in rows),
               "empirical_lb": " ".join(f"({r['depth']},{r['log10_empirical_lb_max']:.2f})" for r in rows),
               "axis_note": "y axis is log10(K); use ymode=normal with these coordinates or 10^y with ymode=log"}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        D = [r["depth"] for r in rows]
        ax.plot(D, [r["log10_worst_case"] for r in rows], "s-", c="red", label="Analytical worst case (Thm. 1)")
        ax.plot(D, [r["log10_data_dependent_max"] for r in rows], "^-", c="purple", label="Analytical, data-dependent (Alg. 1)")
        ax.plot(D, [r["log10_op_norm_product"] for r in rows], "p:", c="orange", label="Op-norm product")
        ax.plot(D, [r["log10_empirical_lb_max"] for r in rows], "v--", c="blue", label=f"Empirical LB ({args.attacks} attacks)")
        ax.set_xlabel("Layer depth L"); ax.set_ylabel("log$_{10}$ Lipschitz estimate K"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(Path(args.out).with_suffix(".png"), dpi=150); print("figure:", Path(args.out).with_suffix(".png"))
    except Exception as e:  # pragma: no cover
        print("matplotlib unavailable:", e)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
