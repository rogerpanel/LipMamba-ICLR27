"""Run HiSPA against a randomly-initialised LipMamba and print the α ratio."""
from __future__ import annotations

import torch

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.attacks import HiSPAAttack, HiSPAConfig
from lipmamba.utils import set_seed


def main() -> None:
    set_seed(0)
    cfg = LipMambaConfig(
        vocab_size=128, n_layers=2, d_model=32, d_inner=64,
        state_dim=4, conv_kernel=3,
    )
    model = LipMambaModel(cfg).eval()

    prefix_ids = torch.randint(0, 128, (2, 16))
    attacker = HiSPAAttack(model, HiSPAConfig(trigger_length=8, n_steps=20, lr=0.05))
    delta, info = attacker.attack(prefix_ids)
    print(info)


if __name__ == "__main__":
    main()
