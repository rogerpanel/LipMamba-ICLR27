#!/usr/bin/env python
"""TODO 4 — produce (or delete) the GloRo-Mamba baseline row with the released harness.

Trains, under *identical* data, seeds, optimiser and steps, the four presets

    unconstrained_mamba   (Table 1 row "Mamba")
    gloro_mamba           (Table 1 row "GloRo-Mamba (head only)")
    naive_sn_mamba        (Figure 3 "Naive SN")
    lipmamba_arch         (Table 1 row "LipMamba")

and evaluates ACC, PACC (HiSPA ℓ=24, Z + M + adaptive), LL-Acc@0.18, ECE and
per-token latency, then runs Friedman + Holm over the seeds.  The output is
the JSON that scripts/fill_paper_numbers.py turns into the table rows.

    python scripts/todo4_gloro_mamba_baseline.py --config configs/ids_cic2017.yaml --seeds 42 137 2026
    python scripts/todo4_gloro_mamba_baseline.py --demo
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from lipmamba import LipMambaModel
from lipmamba.attacks import AdaptiveClampAttack, AdaptiveClampConfig, HiSPAAttack, HiSPAConfig
from lipmamba.baselines import preset
from lipmamba.certificates.local_lipschitz import LocalLipschitzConfig
from lipmamba.certificates.pac_bayes import PACBayesConfig
from lipmamba.evaluation import expected_calibration_error, friedman_holm, ll_accuracy, mean_std
from lipmamba.training import LipMambaTrainer, TrainerConfig
from lipmamba.utils import set_seed

PRESETS = ("unconstrained_mamba", "gloro_mamba", "naive_sn_mamba", "lipmamba_arch")


def _collate(b):
    return {"input_ids": torch.stack([x[0] for x in b]), "labels": torch.stack([x[1] for x in b])}


@torch.no_grad()
def pacc(model, ids, y, L=24, n_steps=50):
    clean_ok = model(ids)["cls_logits"].argmax(-1) == y
    ok = clean_ok.clone()
    h = HiSPAAttack(model, HiSPAConfig(trigger_length=L, n_steps=n_steps, generations=5, population=8, vocab_subset=256))
    for trig, _ in (h.z_hispa(ids), h.m_hispa(ids)):
        ok &= model(torch.cat([ids, trig], 1))["cls_logits"].argmax(-1) == y
    with torch.enable_grad():
        att = AdaptiveClampAttack(model, AdaptiveClampConfig(objective="margin", trigger_length=L, n_steps=n_steps))
        trig_emb, _ = att.attack(ids, y)
    emb = torch.cat([model.embed_tokens(ids), trig_emb], 1)
    ok &= model.logits_from_embeddings(emb).argmax(-1) == y
    return float(ok.float().mean()), float(clean_ok.float().mean())


def run_one(name, cfg_kwargs, train_loader, val_ids, val_y, steps, seed, ll_cfg):
    set_seed(seed)
    model = LipMambaModel(preset(name, **cfg_kwargs))
    tcfg = TrainerConfig(max_steps=steps, warmup_steps=max(1, steps // 10), log_every=max(1, steps // 4),
                         eval_every=10**9, save_every=10**9, out_dir=f"runs/todo4/{name}_s{seed}",
                         lipschitz_mode="local", local_lipschitz=ll_cfg, pac_bayes=PACBayesConfig(n_train=len(train_loader.dataset)),
                         use_margin_objective=(name != "unconstrained_mamba"))
    LipMambaTrainer(model, train_loader, cfg=tcfg).train()
    model.eval()
    with torch.no_grad():
        logits = model(val_ids)["cls_logits"]
    acc = float((logits.argmax(-1) == val_y).float().mean())
    ece = expected_calibration_error(logits, val_y)
    p, _ = pacc(model, val_ids, val_y)
    ll = ll_accuracy(model, DataLoader(TensorDataset(val_ids, val_y), batch_size=16, collate_fn=_collate), radii=[0.18], cfg=ll_cfg)
    t = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            model(val_ids[:8])
    lat_ms = (time.perf_counter() - t) / 5 / (8 * val_ids.size(1)) * 1e3
    return {"ACC": acc, "PACC": p, "LL_Acc_0.18": ll["ll_acc"][0.18], "global_cert_acc_0.18": ll["global_cert_acc"][0.18],
            "ECE": ece, "latency_ms_per_token": lat_ms, "log10_L_global": model.log10_network_lipschitz_bound()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2026])
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--out", default="runs/todo4_baselines.json")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo or not args.config:
        V, T, K, n = 128, 24, 4, 256
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, V, (n, T), generator=g)
        y = (ids[:, :8].sum(-1) % K)                       # learnable synthetic rule
        train_loader = DataLoader(TensorDataset(ids[:192], y[:192]), batch_size=32, shuffle=True, collate_fn=_collate)
        val_ids, val_y = ids[192:], y[192:]
        kw = dict(vocab_size=V, n_layers=2, d_model=32, d_inner=64, state_dim=8, conv_kernel=3, n_classes=K)
        steps = min(args.steps, 60); ll_cfg = LocalLipschitzConfig(n_steps=3, n_restarts=2)
        prov = "DEMO synthetic classification"
    else:
        cfg = yaml.safe_load(Path(args.config).read_text())
        from train import build_dataloaders
        train_loader, val_loader, num_classes = build_dataloaders(cfg["dataset"])
        vb = next(iter(val_loader)); val_ids, val_y = vb["input_ids"], vb["labels"]
        kw = {**cfg["model"], "n_classes": num_classes}; kw.pop("variant", None)
        steps = args.steps; ll_cfg = LocalLipschitzConfig(); prov = f"config={args.config}"

    results = {p: {} for p in PRESETS}
    for seed in args.seeds:
        for p in PRESETS:
            results[p][seed] = run_one(p, kw, train_loader, val_ids, val_y, steps, seed, ll_cfg)
            print(f"seed={seed} {p:<20} " + " ".join(f"{k}={v:.3f}" for k, v in results[p][seed].items() if isinstance(v, float)))

    table = {}
    for p in PRESETS:
        table[p] = {m: mean_std([results[p][s][m] for s in args.seeds]) for m in results[p][args.seeds[0]]}
    stats = {}
    for m in ("ACC", "PACC", "LL_Acc_0.18"):
        try:
            stats[m] = friedman_holm({p: [results[p][s][m] for s in args.seeds] for p in PRESETS}, reference="lipmamba_arch")
        except Exception as e:  # noqa: BLE001
            stats[m] = {"error": str(e)}
    out = {"provenance": prov, "seeds": args.seeds, "steps": steps, "per_seed": {p: {str(s): r for s, r in d.items()} for p, d in results.items()},
           "table_mean_std": table, "stats": stats}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(table, indent=2)); print("wrote", args.out)


if __name__ == "__main__":
    main()
