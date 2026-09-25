#!/usr/bin/env python
"""Train (or fine-tune) a LipMamba model from a YAML config.

Usage
-----
    python scripts/train.py --config configs/lipmamba_130m.yaml

The script supports three task types automatically inferred from the config:

* ``model.n_classes == 0`` (default)   →  Language modelling (LM head).
* ``model.n_classes  > 0``             →  Sequence classification (cls head).
* ``dataset.csv_path`` present         →  Tabular IDS classification.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.certificates.pac_bayes import PACBayesConfig
from lipmamba.data.ids import IDSDataset, IDSDatasetConfig
from lipmamba.data.language import LanguageModellingDataset, collate_lm
from lipmamba.training import LipMambaTrainer, TrainerConfig
from lipmamba.utils import set_seed


def build_model(model_cfg: dict) -> LipMambaModel:
    import dataclasses
    model_cfg = dict(model_cfg)
    variant = model_cfg.pop("variant", None)
    known = {f.name for f in dataclasses.fields(LipMambaConfig)}
    unknown = {k: model_cfg.pop(k) for k in list(model_cfg) if k not in known}
    if unknown:
        print(f"[build_model] ignoring unknown model keys: {sorted(unknown)}")
    if variant == "lipmamba_130m":
        cfg = LipMambaConfig.lipmamba_130m(**model_cfg)
    elif variant == "lipmamba_370m":
        cfg = LipMambaConfig.lipmamba_370m(**model_cfg)
    elif variant == "lipmamba_1300m":
        cfg = LipMambaConfig.lipmamba_1300m(**model_cfg)
    else:
        cfg = LipMambaConfig(**model_cfg)
    return LipMambaModel(cfg)


def build_dataloaders(dataset_cfg: dict) -> tuple[DataLoader, DataLoader | None, int]:
    if "csv_path" in dataset_cfg:
        ds_cfg = IDSDatasetConfig(
            name=dataset_cfg["name"],
            csv_path=dataset_cfg["csv_path"],
            label_col=dataset_cfg.get("label_col", "Label"),
            standardise=dataset_cfg.get("standardise", True),
        )
        full = IDSDataset(ds_cfg)
        n_val = max(1, int(0.05 * len(full)))
        train, val = random_split(full, [len(full) - n_val, n_val])
        bs = dataset_cfg.get("batch_size", 256)
        nw = dataset_cfg.get("num_workers", 4)
        train_loader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=nw)
        val_loader = DataLoader(val, batch_size=bs, shuffle=False, num_workers=nw)
        return train_loader, val_loader, full.num_classes

    block = dataset_cfg.get("block_size", 1024)
    train_ds = LanguageModellingDataset(dataset_cfg["token_path"], block_size=block)
    val_path = dataset_cfg.get("val_token_path")
    val_ds = LanguageModellingDataset(val_path, block_size=block) if val_path else None
    bs = dataset_cfg.get("batch_size", 32)
    nw = dataset_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_lm
    )
    val_loader = (
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_lm)
        if val_ds is not None else None
    )
    return train_loader, val_loader, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 42))

    train_loader, val_loader, num_classes = build_dataloaders(cfg["dataset"])
    if num_classes:
        cfg["model"].setdefault("n_classes", num_classes)

    model = build_model(cfg["model"])
    if args.device:
        model.to(args.device)
    elif torch.cuda.is_available():
        model.to("cuda")

    from lipmamba.certificates.local_lipschitz import LocalLipschitzConfig
    tr = dict(cfg["trainer"])
    ll = LocalLipschitzConfig(**tr.pop("local_lipschitz", {}))
    trainer_cfg = TrainerConfig(**{**tr, "pac_bayes": PACBayesConfig(**cfg["pac_bayes"]), "local_lipschitz": ll})
    trainer = LipMambaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=trainer_cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()
