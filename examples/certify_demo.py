"""Constants and per-input retention demonstration (Lemma 1, Theorems 1–2)."""
from __future__ import annotations

import torch

from lipmamba import LipMambaConfig, LipMambaModel
from lipmamba.certificates.certified_radius import certified_radius_batch
from lipmamba.certificates.local_lipschitz import LocalLipschitzConfig, ll_radius, local_lipschitz_estimate
from lipmamba.certificates.poisoning_immunity import ell_star_distribution, ell_star_from_trace
from lipmamba.utils import set_seed


def main() -> None:
    set_seed(0)
    model = LipMambaModel(LipMambaConfig(vocab_size=128, n_layers=2, d_model=32, d_inner=64, state_dim=4,
                                         conv_kernel=3, n_classes=4)).eval()
    cs = model.constraints
    print("constants:", {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in cs.summary(n_blocks=2).items() if k in ("c", "rho_max", "rho_min", "H", "L_block", "log10_L_network")})

    ids = torch.randint(0, 128, (8, 16))
    out = model(ids)
    tr = model.blocks[-1].ssm.last_trace
    print("Lemma 1 holds on this batch:", bool((tr.h_norm <= cs.H).all()))
    print("data-dependent network bound (max):", float(model.data_dependent_network_bound().max()))

    emb = model.embed_tokens(ids).detach()
    lloc = local_lipschitz_estimate(model, emb, LocalLipschitzConfig(n_steps=5, n_restarts=2))
    print("L_loc:", [round(float(v), 3) for v in lloc])
    print("LL radius ε*(x):", [round(float(v), 4) for v in ll_radius(out["cls_logits"], lloc)])
    print("global-constant radius (vacuous):", [f"{float(v):.1e}" for v in certified_radius_batch(out["cls_logits"], model.network_lipschitz_bound())])

    t0 = 8
    dist = ell_star_distribution(cs, tr.h_norm[:, t0 - 1], alpha_min=0.5)
    print("worst-case ℓ* (per input) median:", round(dist["ell_star_median"], 3))
    print("data-dependent ℓ* (per input):", [round(float(v), 2) for v in ell_star_from_trace(tr.a_bar_min, tr.injection_norm, tr.h_norm, t0)])


if __name__ == "__main__":
    main()
