#!/usr/bin/env python
"""One-command driver for the experiments the ICLR re-review asked for.

Runs, on YOUR trained checkpoint and the RoBench-25 release, everything that
is "implemented but not reported" in the submission and writes one JSON that
``scripts/fill_paper_numbers.py`` turns into the missing table rows and the
adaptive-attack paragraph:

  A. RoBench-25 clean accuracy through the true/false verbalizer;
  B. PACC under Z-HiSPA, M-HiSPA and the three adaptive objectives at
     ℓ ∈ {4, 8, 12, 16, 20, 24, 32, 48} (the adaptive column of Fig. 4);
  C. LL-Acc@0.18 under L_loc (Appendix E) and the same under the global constant;
  D. randomized-smoothing certified accuracy (Cohen et al.) at the same radii;
  E. (optional) the GloRo-head-only Mamba baseline: pass --gloro-checkpoint,
     obtained by fine-tuning preset "gloro_mamba" with scripts/todo4_gloro_mamba_baseline.py.

Cost estimate (A100 80 GB, 130M, 240 questions): A+C ≈ 10 min, B ≈ 45 min per
objective for 8 lengths at 200 steps, D ≈ 20 min at n = 512.  370M ≈ 2.5×.

    python scripts/run_review_experiments.py --config configs/lipmamba_130m.yaml \
        --checkpoint runs/lipmamba_130m/final.pt --robench data_cache/robench25.jsonl \
        --tokenizer EleutherAI/gpt-neox-20b --seed 42 --out runs/review/lipmamba_130m_s42.json
    (repeat for seeds 137 and 2026, then: python scripts/fill_paper_numbers.py --runs runs)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lipmamba.attacks import AdaptiveClampAttack, AdaptiveClampConfig, HiSPAAttack, HiSPAConfig
from lipmamba.baselines import RandomizedSmoothing, SmoothingConfig
from lipmamba.certificates.certified_radius import certified_radius_batch
from lipmamba.certificates.local_lipschitz import LocalLipschitzConfig, ll_radius, local_lipschitz_estimate
from lipmamba.data.robench import RoBenchDataset
from lipmamba.evaluation import expected_calibration_error
from lipmamba.utils import load_checkpoint, set_seed


def tokenise_robench(items, tok, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids, labels = [], []
    for it in items:
        enc = tok(it.prompt(trigger=""), add_special_tokens=False)["input_ids"][-max_len:]
        ids.append(enc); labels.append(int(it.answer))
    L = max(map(len, ids)); pad = tok.pad_token_id or 0
    ids = torch.tensor([[pad] * (L - len(s)) + s for s in ids])       # left-pad: answer position = last token
    return ids, torch.tensor(labels)


@torch.no_grad()
def acc_of(model, ids, y, bs=8):
    pred = torch.cat([model(ids[i: i + bs])["cls_logits"].argmax(-1) for i in range(0, ids.size(0), bs)])
    return float((pred == y).float().mean()), pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--robench", required=True); ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--gloro-checkpoint", default=None, help="GloRo-head-only Mamba fine-tuned checkpoint (preset gloro_mamba)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 32, 48])
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", required=True)
    ap.add_argument("--rs-sigma", type=float, default=0.25); ap.add_argument("--rs-n", type=int, default=512)
    ap.add_argument("--skip-rs", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer
    from train import build_model
    import yaml
    cfg = yaml.safe_load(Path(a.config).read_text())
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    verbalizer = [tok(" true", add_special_tokens=False)["input_ids"][0], tok(" false", add_special_tokens=False)["input_ids"][0]]

    def load(ckpt, model_cfg):
        m = build_model(dict(model_cfg)); load_checkpoint(ckpt, m); m.set_verbalizer([verbalizer[1], verbalizer[0]])  # class 0 = false, 1 = true
        return m.to(dev).eval()

    model = load(a.checkpoint, cfg["model"])
    items = RoBenchDataset(a.robench).to_list()
    ids, y = tokenise_robench(items, tok, a.max_len); ids, y = ids.to(dev), y.to(dev)
    out = {"provenance": vars(a), "n_questions": len(items), "constants": model.constraints.summary(n_blocks=model.cfg.n_layers)}

    # A. clean accuracy, ECE
    acc, _ = acc_of(model, ids, y)
    with torch.no_grad():
        logits = torch.cat([model(ids[i: i + 8])["cls_logits"] for i in range(0, ids.size(0), 8)])
    out["clean"] = {"acc": acc, "ece": expected_calibration_error(logits, y)}

    # B. PACC vs trigger length: published + adaptive
    rows = []
    for L in a.lengths:
        row = {"ell": L}
        h = HiSPAAttack(model, HiSPAConfig(trigger_length=L, n_steps=a.steps))
        for name, fn in (("z_hispa", lambda: h.z_hispa(ids)), ("m_hispa", lambda: h.m_hispa(ids, seed=a.seed))):
            trig, rep = fn()
            pacc, _ = acc_of(model, torch.cat([ids, trig], 1), y)
            row[name] = {"pacc": pacc, "alpha_mean": rep["alpha_mean"]}
        for obj in ("saturate", "overwrite", "margin"):
            att = AdaptiveClampAttack(model, AdaptiveClampConfig(objective=obj, trigger_length=L, n_steps=a.steps))
            trig_emb, rep = att.attack(ids, y)
            with torch.no_grad():
                emb = torch.cat([model.embed_tokens(ids), trig_emb], 1)
                pred = torch.cat([model.logits_from_embeddings(emb[i: i + 8]).argmax(-1) for i in range(0, emb.size(0), 8)])
            row[f"adaptive_{obj}"] = {"pacc": float((pred == y).float().mean()), **{k: v for k, v in rep.items() if isinstance(v, float)}}
        row["worst_case_pacc"] = min(v["pacc"] for k, v in row.items() if isinstance(v, dict))
        rows.append(row); print(json.dumps(row))
    out["pacc_vs_length"] = rows

    # C. LL-Acc and global-constant accuracy at 0.18
    emb = model.embed_tokens(ids).detach()
    lloc = torch.cat([local_lipschitz_estimate(model, emb[i: i + 8], LocalLipschitzConfig()) for i in range(0, emb.size(0), 8)])
    correct = logits.argmax(-1) == y
    eps_local = ll_radius(logits, lloc); eps_global = certified_radius_batch(logits, float(model.network_lipschitz_bound()))
    radii = [0.04, 0.08, 0.12, 0.16, 0.18, 0.20, 0.24, 0.28]
    out["ll_acc"] = {str(r): float((correct & (eps_local >= r)).float().mean()) for r in radii}
    out["global_cert_acc"] = {str(r): float((correct & (eps_global >= r)).float().mean()) for r in radii}
    out["L_loc_median"] = float(lloc.median())

    # D. randomized smoothing
    if not a.skip_rs:
        rs = RandomizedSmoothing(model, SmoothingConfig(sigma=a.rs_sigma, n=a.rs_n))
        out["randomized_smoothing"] = {"sigma": a.rs_sigma, "n": a.rs_n,
                                       "certified_acc": {str(k): v for k, v in rs.certified_curve(emb, y, 2, radii).items()}}

    # E. GloRo-head-only Mamba
    if a.gloro_checkpoint:
        gcfg = {**cfg["model"], "spectral_norm": False, "clamp_delta": False, "reparam_eigen": False, "gloro_head": True}
        g = load(a.gloro_checkpoint, gcfg)
        gacc, _ = acc_of(g, ids, y)
        with torch.no_grad():
            gl = torch.cat([g(ids[i: i + 8])["cls_logits"] for i in range(0, ids.size(0), 8)])
        gemb = g.embed_tokens(ids).detach()
        glloc = torch.cat([local_lipschitz_estimate(g, gemb[i: i + 8], LocalLipschitzConfig()) for i in range(0, gemb.size(0), 8)])
        gcorrect = gl.argmax(-1) == y
        h = HiSPAAttack(g, HiSPAConfig(trigger_length=24, n_steps=a.steps)); trig, _ = h.m_hispa(ids, seed=a.seed)
        gpacc, _ = acc_of(g, torch.cat([ids, trig], 1), y)
        out["gloro_mamba"] = {"acc": gacc, "pacc_mhispa_24": gpacc, "ece": expected_calibration_error(gl, y),
                              "ll_acc_0.18": float((gcorrect & (ll_radius(gl, glloc) >= 0.18)).float().mean())}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
