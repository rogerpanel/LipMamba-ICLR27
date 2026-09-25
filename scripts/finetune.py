#!/usr/bin/env python
"""Fine-tune a pre-trained LipMamba checkpoint on a downstream task."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from lipmamba.utils import load_checkpoint, set_seed
from lipmamba.training import LipMambaTrainer, TrainerConfig
from lipmamba.certificates.pac_bayes import PACBayesConfig
from train import build_dataloaders, build_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from", required=True, help="Path to pre-trained checkpoint")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 42))
    train_loader, val_loader, num_classes = build_dataloaders(cfg["dataset"])
    if num_classes:
        cfg["model"].setdefault("n_classes", num_classes)
    model = build_model(cfg["model"])
    load_checkpoint(args.init_from, model)
    if torch.cuda.is_available():
        model.to("cuda")

    trainer_cfg = TrainerConfig(
        **{**cfg["trainer"], "pac_bayes": PACBayesConfig(**cfg["pac_bayes"])}
    )
    trainer = LipMambaTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader, cfg=trainer_cfg
    )
    trainer.train()


if __name__ == "__main__":
    main()
