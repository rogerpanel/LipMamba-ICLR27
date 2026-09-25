"""Train a tiny LipMamba on synthetic data — runs on CPU in seconds."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.certificates.pac_bayes import PACBayesConfig
from lipmamba.training import LipMambaTrainer, TrainerConfig
from lipmamba.utils import set_seed


def main() -> None:
    set_seed(0)
    cfg = LipMambaConfig(
        vocab_size=128, n_layers=2, d_model=32, d_inner=64,
        state_dim=4, conv_kernel=3, n_classes=4, epsilon_train=0.18,
    )
    model = LipMambaModel(cfg)

    n = 256
    ids = torch.randint(0, 128, (n, 16))
    labels = torch.randint(0, 4, (n,))
    train_loader = DataLoader(
        TensorDataset(ids, labels), batch_size=16,
        collate_fn=lambda b: {
            "input_ids": torch.stack([x[0] for x in b]),
            "labels":    torch.stack([x[1] for x in b]),
        },
        shuffle=True,
    )

    # lipschitz_mode="fixed" keeps the CPU demo fast; the paper's setting is
    # "local" (Appendix-E estimator per batch), see configs/lipmamba_130m.yaml.
    trainer_cfg = TrainerConfig(
        max_steps=20, log_every=5, eval_every=10, save_every=10,
        out_dir="runs/demo", lipschitz_mode="fixed", l_fixed=1.0,
        pac_bayes=PACBayesConfig(n_train=n),
    )
    LipMambaTrainer(model=model, train_loader=train_loader, cfg=trainer_cfg).train()
    print("log10 worst-case global bound (Theorem 1 product) =", model.log10_network_lipschitz_bound())
    print("data-dependent bound after last forward =", model.data_dependent_network_bound())


if __name__ == "__main__":
    main()
