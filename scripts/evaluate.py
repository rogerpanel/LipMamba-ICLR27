#!/usr/bin/env python
"""Run clean accuracy / perplexity / certified-accuracy on a checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.evaluation import BenchmarkRunner
from lipmamba.utils import load_checkpoint, set_seed
from train import build_dataloaders, build_model  # noqa: E402  (sibling script)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 42))

    train_loader, val_loader, num_classes = build_dataloaders(cfg["dataset"])
    if num_classes:
        cfg["model"].setdefault("n_classes", num_classes)
    model = build_model(cfg["model"])
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    runner = BenchmarkRunner(model)
    if num_classes:
        result = runner.run(name=cfg["dataset"].get("name", "eval"), cls_loader=val_loader)
    else:
        result = runner.run(name=cfg["dataset"].get("name", "eval"), lm_loader=val_loader)

    payload = {
        "name": result.name,
        "clean_acc": result.clean_acc,
        "perplexity": result.perplexity,
        "certified": result.certified,
        "pacc": result.pacc,
    }
    print(json.dumps(payload, indent=2, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
