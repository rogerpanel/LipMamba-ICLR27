"""Data-dependent PAC-Bayes prior.

The paper fits the prior on a held-out 5% clean split.  We provide a thin
wrapper that runs ``n_steps`` of optimisation on this split with the
standard cross-entropy loss (no adversarial perturbation, no Lipschitz
penalty) and freezes the resulting parameters as the prior mean ``θ_prior``.

The prior σ₀ is a free hyper-parameter (default ``0.10`` per the paper).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from .pac_bayes import flatten_constrained_parameters


def fit_data_dependent_prior(
    model: nn.Module,
    holdout_loader,
    loss_fn: Callable[[nn.Module, dict], torch.Tensor],
    n_steps: int = 1_000,
    lr: float = 2e-4,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    log_every: int = 100,
) -> torch.Tensor:
    """Fit ``model`` on the holdout, then return its flat parameter vector.

    Parameters
    ----------
    model : the *prior* model (a fresh copy of the architecture).
    holdout_loader : iterable returning a batch dict each step.
    loss_fn : callable ``loss_fn(model, batch) -> scalar tensor``.
    n_steps : number of optimisation steps to perform.
    """
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    iterator = iter(holdout_loader)
    for step in range(n_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(holdout_loader)
            batch = next(iterator)
        loss = loss_fn(model, batch)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        if step % log_every == 0:
            print(f"[prior-fit] step={step} loss={loss.item():.4f}")
    return flatten_constrained_parameters(model)


def save_prior(prior_vector: torch.Tensor, path: str | Path) -> None:
    torch.save({"prior": prior_vector.detach().cpu()}, str(path))


def load_prior(path: str | Path) -> torch.Tensor:
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return state["prior"]
