# LipMamba

**Lipschitz-Constrained Selective State-Space Models with State-Retention and PAC-Bayesian Bounds Against Hidden-State Poisoning**

Reference implementation and reproducibility harness for the LipMamba
manuscripts (journal versions: <https://anonymous.4open.science/r/LipMamba-ICLR27>;
ICLR 2027 submission version: anonymised, see `paper/README.md`).

Reproducibility mirror of this folder: <https://anonymous.4open.science/r/LipMamba-ICLR27>

---

## What the code implements (ICLR-version numbering)

| Manuscript | Module |
| --- | --- |
| Eq. (2) two-sided step clamp Δ_t ∈ [Δ_min, Δ_max) | `models/clipped_delta.py` |
| Spectral projection ‖W̄_•‖₂ ≤ s_• (one-step power iteration) | `models/spectral_norm.py` |
| Eigenvalue reparameterisation λ_i(A) ∈ [−λ_max, −λ_min] | `models/eigen_reparam.py` |
| Assumption 1 ‖x_t‖₂ ≤ X_max (projection, by construction) | `models/input_clip.py` |
| Selective scan + Algorithm 1 online constant tracking | `models/selective_ssm.py` |
| GloRo head, ε\*(x), margin-augmented logit | `models/glorot_head.py` |
| Lemma 1 (H), Theorem 1 (L_block), Theorem 2 (ℓ\*) closed forms | `certificates/constants.py` |
| Worst-case / data-dependent / empirical-lower-bound Lipschitz | `certificates/lipschitz.py` |
| Appendix E local estimator L_loc, LL-Acc | `certificates/local_lipschitz.py`, `evaluation/ll_acc.py` |
| Theorem 3 PAC-Bayes with L_ℓ; Eq. (5) objective | `certificates/pac_bayes.py`, `training/` |
| State-retention bound, per-input ℓ\* distribution | `certificates/poisoning_immunity.py` |
| Z-HiSPA, M-HiSPA (GA), continuous HiSPA | `attacks/hispa.py` |
| Adaptive white-box attack on the clamp (saturate / overwrite / margin) | `attacks/adaptive_clamp.py` |
| Baselines: unconstrained Mamba, GloRo-Mamba, Naive-SN, randomized smoothing | `baselines/` |
| ECE (15 bins), Friedman + Holm, Wilcoxon r | `evaluation/calibration.py`, `evaluation/stats.py` |
| RoBench-25 (120 abstracts, 240 T/F questions), HarmBench/JailbreakBench/WildJailbreak, IDS sets | `data/` |

## Install

```bash
git clone https://anonymous.4open.science/r/LipMamba-ICLR27 && cd CV/lipmamba
python -m venv .venv && source .venv/bin/activate
pip install -e ".[training,dev]"
pytest -q            # 36 tests: clamp bounds, Lemma 1, Thm 1/2 constants, attacks, estimator
```

The scan is pure PyTorch (CPU works); install `mamba-ssm` for speed on GPU.

## Check the constants before you train

```bash
python scripts/report_constants.py --config configs/lipmamba_130m.yaml
```

prints c, ρ_max, ρ_min, H, γ, L_block, the 24-block product and ℓ\*.  With the
manuscript's stated constants this gives **L_block ≈ 1.1 × 10⁸** and
**ℓ\* ≈ 0.95 tokens** — see `docs/ICLR2027_AUDIT_RESPONSE.md` for why this
matters and `docs/THEORY.md` for the formulas.

## Resolve the manuscript's `\todo` markers

```bash
python scripts/regenerate_all.py --demo                 # smoke run, random model, CPU
python scripts/regenerate_all.py --config configs/lipmamba_130m.yaml \
    --checkpoint runs/lipmamba_130m/final.pt --tokens data_cache/wikitext103_val.bin
```

| Marker | Script |
| --- | --- |
| ℓ\* recomputation + Fig. 4 region | `scripts/todo1_ell_star.py` |
| adaptive white-box attack | `scripts/todo2_adaptive_attack.py` |
| base checkpoints + corpus, PPL overhead | `scripts/todo3_init_from_hf_mamba.py`, `scripts/perplexity_overhead.py` |
| GloRo-Mamba baseline row | `scripts/todo4_gloro_mamba_baseline.py` |
| Fig. 2 four curves | `scripts/todo5_fig2_lipschitz_depth.py` |
| anonymised repo bundle | `scripts/todo6_make_anonymous_bundle.py` |
| LaTeX snippets from the JSON outputs | `scripts/fill_paper_numbers.py` → `paper/todo_snippets.tex` |

## Train / evaluate / certify / attack

```bash
python scripts/train.py    --config configs/lipmamba_130m.yaml
python scripts/evaluate.py --config configs/lipmamba_130m.yaml --checkpoint runs/lipmamba_130m/final.pt
python scripts/certify.py  --config configs/certificate.yaml
python scripts/attack.py   --config configs/attack_robench25.yaml
```

Ablations (Table 3) and baselines (Table 1) are configuration presets:
`configs/ablations.yaml`, `configs/baselines.yaml`, `lipmamba.baselines.preset(...)`.

## Datasets

Canonical URLs and licences: `docs/DATASETS.md` and `lipmamba.data.registry`.
No dataset is redistributed.  **RoBench-25** is the HiSPA preprint's
long-context benchmark (120 NeurIPS-2025 abstracts, 240 true/false
questions), obtained from the authors' anonymised release — not a trigger
collection.

## Documentation

* `docs/THEORY.md` — Assumption 1, Lemma 1, Theorems 1–3, Appendix E, with the numbers.
* `docs/ICLR2027_AUDIT_RESPONSE.md` — audit findings → code changes → scripts; open decisions.
* `docs/METHODOLOGY.md`, `docs/HYPERPARAMETERS.md`, `docs/REPRODUCIBILITY.md`, `docs/DATASETS.md`, `docs/ARCHITECTURE.md`, `docs/PAPER_LINKS.md`.
* the deployment notes (omitted), the model card (omitted) — deployment notes (excluded from the anonymised bundle).

## Citing

```bibtex
@article{anonymous2026lipmamba,
  author = {Anonymous},
  title  = {LipMamba: Lipschitz-Constrained Selective State-Space Models with State-Retention and
            PAC-Bayesian Bounds Against Hidden-State Poisoning},
  year   = {2026},
  url    = {https://anonymous.4open.science/r/LipMamba-ICLR27
}
```

MIT — see `LICENSE`.
