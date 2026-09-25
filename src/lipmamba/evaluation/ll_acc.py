"""LL-Acc@ε — local-Lipschitz robust accuracy (Section 5, "Metrics").

The fraction of inputs correctly classified with ε*(x) ≥ ε, where
ε*(x) = margin(x) / (√2 L_loc(x)) and L_loc is the Appendix-E estimator.
Because L_loc is a *lower* estimate of the true local constant the radius
is an *upper* estimate and LL-Acc is *not* a deterministic certificate
(Remark 4).  The function returns, alongside LL-Acc, the same accuracy
computed under the worst-case Theorem-1 global constant so the two can be
reported side by side (the global version will be ≈ 0 at depth).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..certificates.certified_radius import certified_radius_batch
from ..certificates.local_lipschitz import LocalLipschitzConfig, ll_radius, local_lipschitz_estimate


def ll_accuracy(
    model: nn.Module,
    loader: DataLoader,
    radii: list[float] = (0.04, 0.08, 0.12, 0.16, 0.18, 0.20, 0.24, 0.28),
    cfg: LocalLipschitzConfig = LocalLipschitzConfig(),
    max_batches: int | None = None,
) -> dict:
    model.eval()
    dev = next(model.parameters()).device
    all_logits, all_targets, all_lloc = [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        ids = batch["input_ids"].to(dev); y = batch["labels"].to(dev)
        if y.dim() == 2:
            y = y[:, -1]
        emb = model.embed_tokens(ids).detach()
        lloc = local_lipschitz_estimate(model, emb, cfg)
        with torch.no_grad():
            logits = model.logits_from_embeddings(emb)
        all_logits.append(logits); all_targets.append(y); all_lloc.append(lloc)
    logits = torch.cat(all_logits); y = torch.cat(all_targets); lloc = torch.cat(all_lloc)
    correct = logits.argmax(-1) == y
    eps_local = ll_radius(logits, lloc)
    l_global = float(model.network_lipschitz_bound())
    eps_global = certified_radius_batch(logits, l_global)
    return {
        "clean_acc": float(correct.float().mean()),
        "L_loc_median": float(lloc.median()), "L_loc_max": float(lloc.max()),
        "L_global": l_global,
        "ll_acc": {float(r): float((correct & (eps_local >= r)).float().mean()) for r in radii},
        "global_cert_acc": {float(r): float((correct & (eps_global >= r)).float().mean()) for r in radii},
        "eps_local_median": float(eps_local[correct].median()) if correct.any() else 0.0,
    }
