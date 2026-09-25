"""LipMamba on a synthetic IDS-style flow tensor."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.utils import set_seed


def main() -> None:
    set_seed(0)
    n_classes = 5
    cfg = LipMambaConfig(
        vocab_size=64, n_layers=3, d_model=64, d_inner=128,
        state_dim=8, conv_kernel=3, n_classes=n_classes,
    )
    model = LipMambaModel(cfg).eval()

    ids = torch.randint(0, 64, (16, 32))
    out = model(ids)
    pred = out["cls_logits"].argmax(dim=-1)
    print("predicted classes:", pred.tolist())


if __name__ == "__main__":
    main()
