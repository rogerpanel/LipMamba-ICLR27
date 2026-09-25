# Paper ↔ Code Cross-Reference (ICLR 2027 version, `lipmamba_iclr2027.tex`)

| Manuscript | Code |
| --- | --- |
| §3 Preliminaries, Eq. (s6); diagonal-A remark | `models/selective_ssm.py`, `models/eigen_reparam.py` |
| Definition 1 (α, ℓ)-poisoning | `attacks/hispa.py` (retention ratio α reported by every variant) |
| §4 Spectral projection (2 % safety margin) | `models/spectral_norm.py` |
| §4 Eigenvalue reparameterisation | `models/eigen_reparam.py` |
| §4 Eq. (2) two-sided clamp | `models/clipped_delta.py` |
| §4 Readout and GloRo head | `models/lipmamba_block.py`, `models/glorot_head.py` |
| §4 Eq. (5) training objective | `training/pac_objective.py`, `training/trainer.py` |
| Assumption 1 | `models/input_clip.py`, `certificates/constants.py` |
| Lemma 1 (bounded state) | `ConstraintSet.H`; runtime check in `ScanTrace.h_norm` |
| Theorem 1 (Lipschitz constant) | `ConstraintSet.l_block`, `l_network` |
| Remark 4 (global vs local) | `certificates/local_lipschitz.py`, `evaluation/ll_acc.py`, `scripts/todo5_*` |
| Theorem 2 (state-retention) + Eq. (lstar) | `ConstraintSet.retention_lower_bound`, `ell_star` |
| Remark 5 (norm not content; per-input ℓ\*) | `ell_star_distribution`, `ell_star_from_trace`; `AdaptiveClampAttack("overwrite")` |
| Theorem 3 (PAC-Bayes with L_ℓ) | `certificates/pac_bayes.py` |
| §5 Benchmarks (RoBench-25: 120 abstracts / 240 questions) | `data/robench.py` |
| §5 Baselines (Mamba, GloRo-Mamba, Naive SN) | `baselines/__init__.py`; randomized smoothing `baselines/randomized_smoothing.py` |
| §5 Attacks (Z-HiSPA, M-HiSPA, PGD-40; adaptive) | `attacks/hispa.py`, `attacks/pgd.py`, `attacks/adaptive_clamp.py` |
| §5 Metrics (ACC, PACC, ASR, LL-Acc, ECE, latency; Friedman/Holm) | `evaluation/` |
| Table 1 | `scripts/todo4_gloro_mamba_baseline.py` |
| Fig. 2 | `scripts/todo5_fig2_lipschitz_depth.py` |
| Fig. 3 (LL-Acc vs ε) | `evaluation/ll_acc.py` |
| Fig. 3 (perplexity overhead) | `scripts/perplexity_overhead.py` |
| Fig. 4 (HiSPA vs ℓ, shaded region) | `scripts/todo1_ell_star.py`, `scripts/todo2_adaptive_attack.py` |
| Table 3 (ablation) | `configs/ablations.yaml`, `lipmamba.baselines.ablate` |
| Appendix A Algorithm 1 | `SelectiveSSM.forward` (D_t accumulator) |
| Appendix B–D proofs | constants used in each step are named in `certificates/constants.py` docstrings |
| Appendix E local estimator | `certificates/local_lipschitz.py` |
| Appendix F IDS transfer | `data/ids.py`, `configs/ids_cic2017.yaml` |
| Reproducibility statement | `scripts/regenerate_all.py`, `scripts/fill_paper_numbers.py` |
| `\anonrepo` | `scripts/todo6_make_anonymous_bundle.py` |
