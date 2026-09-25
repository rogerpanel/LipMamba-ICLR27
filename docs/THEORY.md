# LipMamba — Theoretical Foundations (ICLR 2027 version)

This document states the results exactly as in the corrected manuscript
(`lipmamba_iclr2027.tex`) and points to the code that computes each
quantity.  Every constant below is produced by
`lipmamba.certificates.constants.ConstraintSet`; run
`python scripts/report_constants.py` to print them for any configuration.

## Assumption 1 (bounded inputs, constrained parameters)

* ‖x_t‖₂ ≤ X_max for all t — **enforced by construction** by the projection
  `InputNormClip` at the SSM input (`models/input_clip.py`); it is
  1-Lipschitz so it does not change any constant.
* ‖W̄_•‖₂ ≤ s_• for • ∈ {B, C, Δ, out} — `SpectralNormLinear`.
* λ_i(A) ∈ [−λ_max, −λ_min], λ_min > 0 — `EigenReparamA`.
* Δ_t ∈ [Δ_min, Δ_max], Δ_min > 0 — the **two-sided clamp** (Eq. 2)

      Δ_t = Δ_min + (Δ_max − Δ_min) · tanh( softplus(W̄_Δ x_t + τ) / (Δ_max − Δ_min) )

  (`models/clipped_delta.py`), smooth and 1-Lipschitz in its pre-activation.

Derived constants:

    c      = s_B Δ_max X_max²                (per-step injection bound ‖B̄_t x_t‖ ≤ c)
    ρ_max  = exp(−Δ_min λ_min)  < 1          (upper bound on ‖Ā_t‖₂)
    ρ_min  = exp(−Δ_max λ_max)  > 0          (lower bound on σ_min(Ā_t))

Because A is real diagonal, Ā_t = diag(e^{Δ_t a_i}) and its spectral radius,
largest and smallest singular values coincide with max_i / min_i e^{Δ_t a_i}
(audit item G: this is why "spectral radius" and "smallest singular value"
are interchangeable *only* under diagonal A; the code records both,
`ScanTrace.a_bar_max` / `a_bar_min`).

## Lemma 1 (bounded state)

    ‖h_t‖₂ ≤ H := c / (1 − ρ_max)      for all t, all inputs, h_0 = 0.

The lower clamp is the reason this exists: Δ_min = 0 ⇒ ρ_max = 1 ⇒ no H.
This is the error-explosion regime of Qi et al. (NeurIPS 2024).
Code: `ConstraintSet.H`; verified at run time by `ScanTrace.h_norm ≤ H`
(`tests/test_attacks_and_local.py::test_scan_trace_records_bounds_within_theory`).

## Theorem 1 (Lipschitz constant of the selective scan)

    ‖f_block(x) − f_block(x′)‖ ≤ L_block ‖x − x′‖,
    L_block = s_out L_SiLU s_C ( X_max γ / (1 − ρ_max) + H ),
    γ       = 2 s_B Δ_max X_max + s_Δ (λ_max H + s_B X_max²),
    L_SSM  ≤ ∏_i L_block^{(i)}          (N blocks; × (1 + L_block) per block if the block has a residual).

The ‖h‖_∞ term of the old version is replaced by the *derived* constant H
(audit item A).  No tightness claim (item C).  Code: `ConstraintSet.l_block`,
`l_network`; online data-dependent refinement of Algorithm 1 in
`SelectiveSSM.forward` (accumulator D_t) and `data_dependent_block_bound()`.

### Remark 4 (scope) — and a numerical warning the manuscript must address

With the manuscript's own 130M constants
(s_B = s_C = 1, s_Δ = 0.5, Δ_min = 10⁻³, Δ_max = 0.5, λ_min = 0.05, λ_max = 1, X_max = 1):

| quantity | value |
| --- | --- |
| ρ_max | 0.99995 |
| 1/(1 − ρ_max) | 2.0 × 10⁴ |
| H | 1.0 × 10⁴ |
| γ | 5.0 × 10³ |
| **L_block** | **1.1 × 10⁸** |
| L_SSM at 24 blocks | 10¹⁹³ |

So Figure 2's "∼10¹⁰ at 24 layers" is **not** the worst-case Theorem-1
product; only the *data-dependent* Algorithm-1 quantity (observed ‖Ā_t‖ and
‖h_t‖ instead of ρ_max and H) can be of that order.  The figure caption and
Remark 4 must say which curve is plotted.  `scripts/todo5_fig2_lipschitz_depth.py`
computes and plots both, plus the operator-norm product and the
attack-based lower estimate, so the four are never conflated again.

To bring the *worst-case* per-block constant to O(10) one needs
1 − ρ_max = O(0.1–1), i.e. Δ_min λ_min ≈ 0.1–0.7 (e.g. Δ_min = 0.25, λ_min = 1
gives L_block ≈ 16; Δ_min = 0.5, λ_min = 1 gives ≈ 7.4).  Those constraints
change the model materially; whether they are acceptable is an empirical
question the sweep in `report_constants.py` lets you answer before training.

## Theorem 2 (state-retention bound)  — formerly "poisoning immunity"

    ‖h_{t0+ℓ}‖₂ ≥ ρ_min^ℓ ‖h_{t0}‖₂ − c (1 − ρ_min^ℓ)/(1 − ρ_min)
    κ  := c / ((1 − ρ_min) ‖h_{t0}‖₂)
    ℓ* := log((α_min + κ)/(1 + κ)) / log(ρ_min)          (> 0 for α_min < 1)

The (1 + κ) denominator was missing in the old inversion (audit item F).
Code: `ConstraintSet.retention_lower_bound`, `ell_star`;
per-input distribution `certificates/poisoning_immunity.py::ell_star_distribution`;
data-dependent version from a `ScanTrace`: `ell_star_from_trace`.

### Remark 5 — what it certifies, and the number

Norm, not content (audit item E).  The adaptive `overwrite` objective in
`attacks/adaptive_clamp.py` is the adversary this remark describes; the demo
already shows it moving the state by > 1 relative unit while keeping the norm.

With (s_B, Δ_max, λ_max, X_max, α_min, ‖h_{t0}‖) = (1, 0.5, 1, 1, 0.5, 4):
ρ_min = 0.607, κ = 0.318, **ℓ* = 0.95** — the TODO in Remark 5 is confirmed.
ℓ* ≈ 24 at α_min = 0.5 requires Δ_max λ_max ≤ 0.029 (κ → 0), i.e. about 17×
smaller than the stated 0.5.  `scripts/todo1_ell_star.py` prints the
(Δ_max, λ_max) sweep, the per-input distribution and the pgfplots command for
the corrected Figure-4 shaded region.

### One-token condition

ℓ\* > 0 does not by itself certify any integer length. At least one token is
certified iff ρ_min(1+κ) − κ ≥ α_min, i.e. ‖h_{t0}‖ ≥ c/(ρ_min − α_min) when
ρ_min > α_min (4.69 for the paper's constants at α_min = 0.5).
Code: `ConstraintSet.one_token_certified`.

## Corollary (directional retention) — v4 addition

With S_ℓ := c(1 − ρ_max^ℓ)/(1 − ρ_max) ≤ cℓ the total injected norm,

    ⟨h_{t0+ℓ}, h_{t0}⟩ ≥ ‖h_{t0}‖ (ρ_min^ℓ ‖h_{t0}‖ − S_ℓ)
    cos∠(h_{t0+ℓ}, h_{t0}) ≥ (ρ_min^ℓ − S_ℓ/‖h_{t0}‖) / (1 + S_ℓ/‖h_{t0}‖)

Proof idea: the product of the diagonal transition matrices over the trigger
has entries ≥ ρ_min^ℓ, so its quadratic form is ≥ ρ_min^ℓ ‖h_{t0}‖²; the
injected terms contribute ≥ −S_ℓ‖h_{t0}‖ by Cauchy–Schwarz. **Note** the
additive term uses ρ_max (each injection is only guaranteed to shrink by
≤ ρ_max afterwards): the ρ_min-discounted recursion of Theorem 2 does not
transfer to inner products. At ℓ = 1, S_1 = c and the directional and norm
certificates coincide; certified cosine 0.43 for the paper's constants at
‖h_{t0}‖ = 4. Code: `injected_norm_bound`, `directional_retention_lower_bound`,
`certified_cosine`; test `tests/test_directional_retention.py`.

## Theorem 3 (PAC-Bayes bound on adversarial risk)

    E_Q[L_adv(θ;ε)] ≤ E_Q[L̂_S(θ)] + E_Q[L_ℓ L(θ)] ε + sqrt((KL(Q‖P) + ln(2√n/δ))/(2n)).

L_ℓ (loss Lipschitz constant in the logits) is now explicit
(`PACBayesConfig.l_ell`).  Training objective (Eq. 5) uses ½ L_ℓ L_loc ε_train
(`PACBayesConfig.objective_half`).  With the global constant of Theorem 1 the
middle term is vacuous; the code instantiates it with L_loc.
Code: `certificates/pac_bayes.py`.

## Appendix E — local Lipschitz estimator

    L_loc(x) = max_{k≠ŷ} max_{x′∈B(x,r)} ‖∇_{x′}(z_ŷ − z_k)(x′)‖₂ / √2,
    r = 0.3, 20 PGD steps, 8 random starts.

A *lower* estimate of the true local constant ⇒ ε*(x) = margin/(√2 L_loc) is an
*empirical local-Lipschitz radius*, and LL-Acc@ε is **not** a certificate.
Code: `certificates/local_lipschitz.py`; metric `evaluation/ll_acc.py`, which
also reports the (≈ 0) accuracy under the global constant next to it.

## GloRo head (Leino et al. 2021)

    ε*(x) = (z_ŷ − max_{k≠ŷ} z_k) / (√2 L),     z̃_K = max_{k≠ŷ} z_k + √2 L ε_train.

Code: `models/glorot_head.py`.  The "⊥" logit is appended as an extra class
(the original GloRo construction) rather than overwriting the runner-up.
