# LipMamba Architecture — module map

The framework figure of the journal manuscripts and the platform
(`docs/figures/fig1_LipMamba_arch.png`) shows seven modules plus the SOC
dashboard.  This table states, for each, where it lives in the code and
whether it belongs to the *research* (the ICLR manuscript) or to the
*deployment* (the deployment platform).  The image predates the corrected mathematics
in four places, listed at the end.

| # | Figure module | Code | In ICLR paper? |
| --- | --- | --- | --- |
| 1 | Spectral-norm projection W̄_B, W̄_C, W̄_Δ (σ_max ≤ s) | `models/spectral_norm.py` | yes |
| 2 | Clipped selective parameters B_t, C_t, Δ_t | `models/clipped_delta.py` (two-sided), `models/selective_ssm.py` | yes (Eq. 2) |
| — | A = −diag(λ_min + (λ_max−λ_min)σ(α)) | `models/eigen_reparam.py` | yes |
| — | Input projection ‖x_t‖ ≤ X_max (not in the image) | `models/input_clip.py` | yes (Assumption 1) |
| 3 | Constrained S6 recurrence | `models/selective_ssm.py` | yes |
| 4 | Output projection W̄_out · SiLU | `models/lipmamba_block.py` | yes |
| 5 | **ConformalGuard agent monitor (e-value martingale)** | `monitoring/conformal_guard.py` (**added**) | no — deployment component; undefined in every manuscript |
| 6 | Online Lipschitz estimator | `certificates/lipschitz.py::LipschitzTracker`, Algorithm-1 tracker in `SelectiveSSM` | yes, as Algorithm 1 (formula updated, see below) |
| 7 | Per-input certificate (GloRo head) | `models/glorot_head.py`, `certificates/local_lipschitz.py` | yes, with L_loc (see below) |
| OUT | SOC analyst dashboard (ŷ, ε\*) + agent-monitor flow | `ConformalGuard.dashboard_payload`, the deployment notes (omitted) | no — deployment |

## Where the image is behind the corrected mathematics

1. **Clamp.** Image: Δ_t = Δ_max·tanh(softplus/Δ_max) (one-sided). Paper Eq. (2)
   and code: two-sided with Δ_min > 0, which is what makes Lemma 1 (bounded
   state) true.
2. **Input projection.** Missing from the image; Assumption 1 requires
   ‖x_t‖₂ ≤ X_max and the code enforces it by construction.
3. **Tracker formula.** Image: L_t = ρ_t L_{t−1} + s_C s_out L_SiLU(β_t + s_B)
   (the v1 formula, which omitted the Δ-sensitivity and state terms). Paper
   Algorithm 1: D_t = ‖Ā_t‖₂ D_{t−1} + γ_t with
   γ_t = 2 s_B Δ_t X_max + s_Δ(λ_max‖h_{t−1}‖ + s_B X_max²), and it is a
   nominal-trajectory diagnostic, not a certificate.
4. **Certificate flow.** Image: tracker L → GloRo head (ε\* = margin/(√2 L_L)).
   Corrected paper: the analytical constant is vacuous at depth (10^193 worst
   case), so the reported radius uses the local estimate L_loc and is
   empirical (Remark 4). Drawing the certificate as fed by the tracker is the
   exact claim the audit rejected.

The ICLR figure (`paper/figures/fig1_arch.tex`, TikZ) shows modules 1–4, 6, 7
with all four corrections; the platform figure
(`docs/figures/fig1_platform.tex`) adds Module 5 and the dashboard with the
corrected formulas.

## Module 5 — what was implemented

`ConformalGuard` turns three per-token telemetry signals of the constrained
scan (step saturation Δ_t/Δ_max, norm loss 1 − ‖h_t‖/‖h_{t−1}‖, relative
injection ‖B̄_t x_t‖/‖h_{t−1}‖) into conformal p-values against a clean
calibration set, then into e-values with the simple-mixture betting function
e(p) = mean_κ κ p^{κ−1} over κ ∈ {0.05,…,0.95}, and multiplies them into a test martingale
M_t. An alarm at M_t ≥ 1/α has anytime-valid false-alarm probability ≤ α
under exchangeability of clean scores (Ville's inequality). On alarm the
monitor recommends tightening Δ_max (raising ρ_min and the certified ℓ\* of
Theorem 2) and emits the dashboard payload (ŷ, ε\*, M_t, alarm). Tests:
`tests/test_conformal_guard.py`.

Because no manuscript defines this module and it has not been evaluated, it
is documented here and in the deployment notes (omitted) only; it must not be
described in the ICLR submission.
