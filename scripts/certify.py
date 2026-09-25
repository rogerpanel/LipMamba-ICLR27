#!/usr/bin/env python
"""Compute certified radii, certified accuracy, and poisoning-immunity bounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from lipmamba.certificates.poisoning_immunity import certified_immunity_summary
from lipmamba.evaluation import certified_eval
from lipmamba.utils import load_checkpoint, set_seed
from train import build_dataloaders, build_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path(cfg["config"]).read_text())
    set_seed(base.get("seed", 42))

    _, val_loader, num_classes = build_dataloaders(base["dataset"])
    if num_classes:
        base["model"].setdefault("n_classes", num_classes)
    model = build_model(base["model"])
    load_checkpoint(cfg["checkpoint"], model)
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    cert_results = (
        certified_eval(model, val_loader, radii=cfg["eval"]["radii"]) if val_loader else {}
    )
    immunity = certified_immunity_summary(**cfg["immunity"])

    out = {"certified": cert_results, "immunity": immunity}
    print(json.dumps(out, indent=2, default=str))
    out_path = Path(cfg["eval"]["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
