#!/usr/bin/env python
"""Run HiSPA / RoBench-25 / PGD attacks against a trained checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from lipmamba.attacks import HiSPAAttack, HiSPAConfig
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

    attack_cfg = HiSPAConfig(
        trigger_length=cfg["attack"]["trigger_length"],
        n_steps=cfg["attack"]["n_steps"],
        lr=cfg["attack"]["lr"],
        norm_budget=cfg["attack"]["norm_budget"],
        target_alpha=cfg["attack"]["target_alpha"],
        init=cfg["attack"]["init"],
    )
    attacker = HiSPAAttack(model, attack_cfg)

    summary = []
    for batch in val_loader:
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        ids = batch["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        delta, info = attacker.attack(ids)
        summary.append(info)

    out_path = Path(cfg["reporting"]["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_batches": len(summary),
        "mean_alpha": sum(s["alpha"] for s in summary) / max(1, len(summary)),
        "success_rate": sum(int(s["success"]) for s in summary) / max(1, len(summary)),
        "samples": summary[:32],
    }
    print(json.dumps(payload, indent=2, default=str))
    out_path.write_text(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
