"""Checkpoint helpers."""
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    step: int,
    path: str | Path,
    extra: dict | None = None,
) -> None:
    state = {
        "step": step,
        "model": model.state_dict(),
        "optim": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    state = torch.load(str(path), map_location=map_location, weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optim") is not None:
        optimizer.load_state_dict(state["optim"])
    return state
