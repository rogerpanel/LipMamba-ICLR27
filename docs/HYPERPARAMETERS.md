# Hyperparameters Reference (ICLR 2027 version)

Every hyperparameter, mapped to where it is consumed.  Values are the
manuscript's stated 130M defaults; see `docs/THEORY.md` for what those values
imply numerically (ℓ\* ≈ 1, L_block ≈ 10⁸).

## Constraint set (Assumption 1)

| Symbol | Default | Code |
| --- | --- | --- |
| s_B, s_C | 1.0 | `SSMConfig.s_b`, `s_c` |
| s_Δ | 0.5 | `SSMConfig.s_delta` |
| s_out | 1.0 | `SSMConfig.s_out` |
| Δ_min | 1e-3 | `SSMConfig.delta_min` (two-sided clamp, Eq. 2) |
| Δ_max | 0.5 | `SSMConfig.delta_max` |
| λ_min | 0.05 | `EigenReparamA.lambda_min` |
| λ_max | 1.0 | `EigenReparamA.lambda_max` |
| X_max | 1.0 | `InputNormClip.x_max` |
| L_SiLU | 1.0998 | `certificates/constants.py::L_SILU` |
| power-iteration steps | 1 | `SpectralNormLinear.n_power_iters` |
| σ̂ safety margin for the certificate | 2 % | manuscript §3 (apply when reading `SpectralNormLinear.sigma`) |

Derived (`ConstraintSet`): c = s_B Δ_max X_max², ρ_max = e^{−Δ_min λ_min},
ρ_min = e^{−Δ_max λ_max}, H = c/(1−ρ_max), γ, L_block, ℓ\*.

## Architecture

| Variant | layers | d_model | d_inner | N | conv | vocab | base checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LipMamba-130M | 24 | 768 | 1536 | 16 | 4 | 50280 | `state-spaces/mamba-130m` |
| LipMamba-370M | 48 | 1024 | 2048 | 16 | 4 | 50280 | `state-spaces/mamba-370m` |
| LipMamba-1.3B | 48 | 2048 | 4096 | 16 | 4 | 50280 | `state-spaces/mamba-1.4b` |

## Certificates / PAC-Bayes (Theorem 3, Eq. 5)

| Symbol | Default | Code |
| --- | --- | --- |
| ε_train | 0.18 | `TrainerConfig.epsilon_train`, `GloroNetHead.epsilon_train` |
| δ | 0.05 | `PACBayesConfig.delta` |
| σ (posterior) | 0.04 | `PACBayesConfig.sigma_post` |
| σ₀ (prior) | 0.10 | `PACBayesConfig.sigma_prior` |
| β | 1.0 | `PACBayesConfig.beta` |
| L_ℓ | 1.0 | `PACBayesConfig.l_ell` |
| ½ factor in Eq. 5 | on | `PACBayesConfig.objective_half` |
| n (bound instantiation) | 10⁷ | `PACBayesConfig.n_train` |
| prior split | 5 % clean held-out | `certificates/prior_fitting.py` |

## Local Lipschitz estimator (Appendix E)

| Field | Default | Code |
| --- | --- | --- |
| radius r | 0.3 (embedding space) | `LocalLipschitzConfig.radius` |
| PGD steps | 20 | `LocalLipschitzConfig.n_steps` |
| random starts | 8 | `LocalLipschitzConfig.n_restarts` |
| Lipschitz mode in training | local | `TrainerConfig.lipschitz_mode` |

## Optimiser

| Field | Default | Code |
| --- | --- | --- |
| AdamW lr / wd / betas | 2e-4 / 0.1 / (0.9, 0.95) | `training/optim.py` |
| schedule | cosine, linear warm-up | `optim.CosineWithWarmup` |
| grad clip | 1.0 | `TrainerConfig.grad_clip` |
| epochs | 100 | manuscript; `max_steps` in configs |
| seeds | {42, 137, 2026} | `scripts/regenerate_all.py` |
| hardware | 4 × A100 80 GB | manuscript |

## Attacks

| Field | Default | Code |
| --- | --- | --- |
| HiSPA trigger lengths | {4, …, 48}; PACC at ℓ = 24 | `HiSPAConfig.trigger_length` |
| target α | 0.05 (success), α_min = 0.5 (Theorem 2) | `HiSPAConfig.target_alpha`, `ell_star(alpha_min)` |
| M-HiSPA GA | pop 64, 50 gen, mutation 0.1, tournament 4 | `HiSPAConfig` |
| PGD | ε ∈ [0.05, 0.30], 40 steps, step ε/4 | `PGDConfig` |
| adaptive clamp attack | 200 steps, lr 0.05, budget 1.0, GCG top-k 64 | `AdaptiveClampConfig` |
| randomized smoothing | σ 0.25, n₀ 64, n 512, α 0.001 | `SmoothingConfig` |

## Metrics

ACC, PACC (HiSPA ℓ = 24), ASR (HarmBench-CLS), LL-Acc@ε under L_loc, ECE (15 bins), latency/token;
Friedman + Holm post-hoc, Wilcoxon with rank-biserial r (`evaluation/stats.py`).
