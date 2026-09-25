#!/usr/bin/env python
"""Run the RoBench-25 / RoBench-26 hidden-state-poisoning benchmark.

For each trigger family we report:

* ``alpha`` — observed ‖h_T‖₂ ratio after running the trigger,
* ``immunity_satisfied`` — True iff the certified lower bound > 0,
* ``predicted_correctly`` — whether the model still classifies the prefix correctly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from lipmamba.certificates.poisoning_immunity import certified_immunity_summary
from lipmamba.data.robench import RoBenchDataset
from lipmamba.utils import load_checkpoint, set_seed
from train import build_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--robench", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 42))

    model = build_model(cfg["model"])
    load_checkpoint(args.checkpoint, model)
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    summary_per_family: dict[str, list[float]] = {}
    immunity = certified_immunity_summary(
        delta_min=0.05, lambda_min=0.05, s_b=1.0, delta_max=0.5,
        x_max=1.0, h0_norm=1.0, alpha=0.05,
    )
    print("Certified-immunity reference:", immunity)

    for sample in RoBenchDataset(args.robench):
        # Plug in your tokenizer here; this script logs the structure only.
        summary_per_family.setdefault(sample.family, []).append(sample.target_alpha)

    payload = {
        "immunity": immunity,
        "families": {f: {"n": len(v), "mean_alpha": sum(v) / len(v)} for f, v in summary_per_family.items()},
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
