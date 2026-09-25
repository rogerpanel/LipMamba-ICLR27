# LipMamba — Methodology (ICLR 2027 version)

## Algorithm 1 — forward pass with online tracking of the analytical constant

```
Require: tokens x_1..x_L, params {W_B, W_C, W_Δ, α, W_out}, budgets (s_B,s_C,s_Δ,s_out),
         (λ_min,λ_max), (Δ_min,Δ_max), X_max
 1: W̄_• ← W_• · min(1, s_•/σ̂_max^PI(W_•))                         (spectral_norm.py)
 2: A ← −diag(λ_min + (λ_max−λ_min)σ(α)); ρ_max ← e^{−Δ_min λ_min}; H ← s_B Δ_max X_max²/(1−ρ_max)
 3: h_0 ← 0; D_0 ← 0
 4: for t = 1..L:
 5:    x_t ← x_t · min(1, X_max/‖x_t‖)                                  (input_clip.py, Assumption 1)
 6:    Δ_t ← Δ_min + (Δ_max−Δ_min) tanh(softplus(W̄_Δ x_t + τ)/(Δ_max−Δ_min))   (clipped_delta.py)
 7:    B_t ← W̄_B x_t; C_t ← W̄_C x_t; Ā_t ← exp(Δ_t A); B̄_t ← Δ_t B_t
 8:    h_t ← Ā_t h_{t−1} + B̄_t x_t;  y_t ← W̄_out SiLU(C_tᵀ h_t)
 9:    γ_t ← 2 s_B Δ_t X_max + s_Δ(λ_max ‖h_{t−1}‖ + s_B X_max²);  D_t ← ‖Ā_t‖₂ D_{t−1} + γ_t
10: L_block ← s_out L_SiLU s_C (X_max max_t D_t + max_t ‖h_t‖)        (data-dependent refinement)
11: return y_{1:L}, L_block, margin z_ŷ − max_{k≠ŷ} z_k
```

Implementation: `SelectiveSSM.forward` records `ScanTrace(delta, a_bar_max,
a_bar_min, injection_norm, h_norm, d_t)`; `data_dependent_block_bound()`
evaluates line 10.  The worst-case (input-free) constant is
`ConstraintSet.l_block`; the data-dependent one is always ≤ it.

## Algorithm 2 — PAC-Bayes adversarial training (Eq. 5)

```
 1: fit prior θ_prior on a 5 % clean held-out split (clean CE, no margin, no penalty)   (prior_fitting.py)
 2: θ ← θ_prior; σ ← 0.04, σ₀ ← 0.10
 3: for each mini-batch (x, y):
 4:    L ← L_loc(x)  (Appendix E: r = 0.3, 20 PGD steps, 8 starts)   [or global / fixed, for ablation]
 5:    z ← logits;  z̃_K ← max_{k≠ŷ} z_k + √2 L ε_train;  L̂_adv ← CE([z, z̃_K], y)      (glorot_head.py)
 6:    L_lip ← ½ L_ℓ L ε_train
 7:    L_KL  ← β sqrt((KL(N(θ,σ²I) ‖ N(θ_prior,σ₀²I)) + ln(2√n/δ)) / 2n)
 8:    loss ← L̂_adv + L_lip + L_KL;  backprop; clip ‖g‖ ≤ 1; AdamW; cosine LR
 9:    (σ̂ refreshed by one-step power iteration inside every forward)
```

Implementation: `training/trainer.py` (`lipschitz_mode`), `training/pac_objective.py`.

## Evaluation protocol (Section 5)

* **ACC** clean accuracy; **PACC** accuracy under HiSPA at ℓ = 24 (Z-HiSPA, M-HiSPA
  *and* the adaptive clamp attack — worst case over the three);
* **ASR** on HarmBench with HarmBench-CLS;
* **LL-Acc@ε** fraction correctly classified with margin/(√2 L_loc) ≥ ε — reported
  next to the same quantity under the global constant (≈ 0) so the scope is visible;
* **ECE** 15 bins; **latency** per token;
* three seeds {42, 137, 2026}; Friedman over methods × seeds, Holm post-hoc,
  Wilcoxon with rank-biserial r.

Scripts: `todo4_gloro_mamba_baseline.py` (Table 1 rows + stats),
`todo2_adaptive_attack.py` (adaptive column), `todo1_ell_star.py` (Fig. 4),
`todo5_fig2_lipschitz_depth.py` (Fig. 2), `perplexity_overhead.py` (Fig. 3).

## Ablations (Table 3)

Configuration presets in `configs/ablations.yaml` / `lipmamba.baselines.ablate`:
no spectral norm, no clamp (vanilla softplus), free A, no GloRo head, no
PAC-Bayes term, no adversarial/margin term, N(0, I) prior, Δ_max ∈ {0.25, 0.5, 1, 2}.
