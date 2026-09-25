#!/usr/bin/env python
"""TODO 2 — adaptive white-box attack against the step clamp.

Runs, for trigger lengths ℓ ∈ {4, 8, 12, 16, 20, 24, 32, 48}:

  * AdaptiveClampAttack(objective="saturate")  — GCG-style objective on Δ_t (continuous + discrete)
  * AdaptiveClampAttack(objective="overwrite") — content-overwrite adversary (Remark 5)
  * AdaptiveClampAttack(objective="margin")    — margin attack on the GloRo head
  * Z-HiSPA and M-HiSPA (published, black-box)   — for reference

and reports retention ratio, Δ-saturation, label-flip rate and — for the
comparison with Theorem 2 — the worst-case ℓ* for the same constants.

Usage
-----
  python scripts/todo2_adaptive_attack.py --config configs/lipmamba_130m.yaml \
      --checkpoint runs/lipmamba_130m/final.pt --data data_cache/robench25_prefixes.bin --out runs/todo2_adaptive.json
  python scripts/todo2_adaptive_attack.py --demo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.attacks import AdaptiveClampAttack, AdaptiveClampConfig, HiSPAAttack, HiSPAConfig
from lipmamba.utils import load_checkpoint, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--checkpoint"); ap.add_argument("--data")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 32, 48])
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--n-prefixes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/todo2_adaptive.json")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--skip-discrete", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)

    if args.demo or not args.config:
        model = LipMambaModel(LipMambaConfig(vocab_size=256, n_layers=2, d_model=32, d_inner=64, state_dim=8,
                                             conv_kernel=3, n_classes=4))
        ids = torch.randint(0, 256, (8, 32)); labels = torch.randint(0, 4, (8,))
        args.n_steps = min(args.n_steps, 20); args.lengths = [4, 8, 16, 24]
        prov = "DEMO random model"
    else:
        cfg = yaml.safe_load(Path(args.config).read_text())
        from train import build_model
        model = build_model(dict(cfg["model"]))
        if args.checkpoint:
            load_checkpoint(args.checkpoint, model)
        import numpy as np
        tok = np.fromfile(args.data, dtype=np.int32)
        T = 512; n = min(args.n_prefixes, tok.size // T)
        ids = torch.from_numpy(tok[: n * T].reshape(n, T)).long()
        labels = None
        prov = f"config={args.config} checkpoint={args.checkpoint} data={args.data} seed={args.seed}"
    model.eval()
    if torch.cuda.is_available():
        model.cuda(); ids = ids.cuda(); labels = labels.cuda() if labels is not None else None

    cs = model.constraints
    rows = []
    for L in args.lengths:
        row = {"trigger_length": L, "theorem2_worst_case_ell_star_h0_4": cs.ell_star(4.0, 0.5)}
        for obj in ("saturate", "overwrite", "margin"):
            att = AdaptiveClampAttack(model, AdaptiveClampConfig(objective=obj, trigger_length=L, n_steps=args.n_steps))
            _, rep = att.attack(ids, labels)
            row[f"adaptive_{obj}_continuous"] = rep
            if not args.skip_discrete and obj == "saturate":
                att_d = AdaptiveClampAttack(model, AdaptiveClampConfig(objective=obj, trigger_length=L,
                                                                       n_steps=max(10, args.n_steps // 4), discrete=True))
                _, rep_d = att_d.attack_discrete(ids[:4], labels[:4] if labels is not None else None)
                row["adaptive_saturate_discrete"] = {k: v for k, v in rep_d.items() if k != "trigger_ids"}
        h = HiSPAAttack(model, HiSPAConfig(trigger_length=L, n_steps=args.n_steps, generations=10, population=16, vocab_subset=256))
        _, z = h.z_hispa(ids); _, g = h.m_hispa(ids, seed=args.seed)
        row["z_hispa"] = z; row["m_hispa"] = {k: v for k, v in g.items() if k != "history"}
        rows.append(row)
        print(f"ℓ={L:3d}  adaptive-saturate: retention={row['adaptive_saturate_continuous']['retention_ratio_mean']:.3f} "
              f"Δsat={row['adaptive_saturate_continuous']['delta_saturation_fraction']:.2f} "
              f"flip={row['adaptive_saturate_continuous']['label_flip_rate']:.2f} | "
              f"overwrite dist={row['adaptive_overwrite_continuous']['overwrite_distance_mean']:.3f} "
              f"| Z-HiSPA α={z['alpha_mean']:.3f}  M-HiSPA α={g['alpha_mean']:.3f}")

    out = {"provenance": prov, "constants": cs.summary(), "rows": rows,
           "reading_guide": {
               "retention_ratio_mean": "‖h_T‖/‖h_T^clean‖ after the trigger; Theorem 2 lower-bounds this for ℓ<ℓ*",
               "delta_saturation_fraction": "mean Δ_t over trigger / Δ_max — 1.0 means the adversary pinned every step at the clamp",
               "overwrite_distance_mean": "‖h_T − h_T^clean‖/‖h_T^clean‖ — large values with retention≈1 show the norm bound is not a content bound",
               "label_flip_rate": "fraction of prefixes whose predicted label changed (behavioural harm)",
           }}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
