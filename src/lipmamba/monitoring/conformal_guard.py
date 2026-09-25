"""Module 5 of the LipMamba framework figure: ConformalGuard agent monitor.

An anytime-valid, sequential monitor of the constrained scan's telemetry that
raises an alarm when the token stream stops looking like clean data.  It is
the deployment-side counterpart of the analytical bounds: Theorem 2 says what
a trigger *cannot* do to the state norm; the monitor detects triggers that
try, from three per-token signals recorded by ``ScanTrace``:

    δ_t   = mean_d Δ_t^{(d)} / Δ_max            step saturation      (HiSPA pushes → 1)
    r_t   = 1 − ‖h_t‖ / ‖h_{t−1}‖               norm loss            (erasure pushes → 1)
    ι_t   = ‖B̄_t x_t‖ / ‖h_{t−1}‖               relative injection   (overwrite pushes ↑)

Method (conformal test martingale, Vovk et al.; e-values, Ramdas et al.):

1. *Calibration.*  On clean sequences, collect nonconformity scores
   A_t = max_j z_j(t), the largest standardised feature (z-scores against the
   calibration mean/std).  Store the calibration scores.
2. *Conformal p-values.*  For a monitored token, p_t = (1 + #{calibration A ≥ A_t})/(n+1).
   Under exchangeability of clean scores p_t is (super-)uniform.
3. *E-values.*  e_t = mean_κ κ p_t^{κ−1} over a grid of betting parameters
   κ ∈ {0.05, …, 0.95} (each κ p^{κ−1} has E[e]=1 under uniform p, hence so
   does the mixture; the "simple mixture" conformal test martingale).
4. *Test martingale.*  M_t = ∏_{s≤t} e_s.  By Ville's inequality,
   P(sup_t M_t ≥ 1/α) ≤ α under the null, so the alarm "M_t ≥ 1/α" has
   anytime-valid false-alarm probability α.  A forgetting factor lets the
   monitor recover after an alarm.
5. *Telemetry feedback (dashed red arrows in the figure).*  On alarm the
   monitor emits a recommendation to tighten Δ_max (which raises ρ_min and
   the certified ℓ* of Theorem 2) and the dashboard payload (ŷ, ε*, M_t, alarm).

Caveat, stated plainly: the false-alarm guarantee holds under
exchangeability of the *clean* per-token scores; consecutive tokens of one
document are not exactly exchangeable, so α should be read as a nominal
level and calibrated empirically on held-out clean documents.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ConformalGuardConfig:
    alpha: float = 0.05             # nominal false-alarm level (alarm at M_t ≥ 1/α)
    layer: int = -1                 # which block's ScanTrace to monitor
    forgetting: float = 1.0         # M_t ← M_{t-1}^{forgetting} · e_t  (1.0 = pure martingale)
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)   # feature weights for (δ, r, ι)
    tighten_factor: float = 0.5     # Δ_max ← tighten_factor · Δ_max on alarm (recommendation only)
    burn_in: int = 4                # tokens before r_t / ι_t are defined reliably


@dataclass
class MonitorResult:
    e_values: torch.Tensor          # (B, T)
    martingale: torch.Tensor        # (B, T)
    p_values: torch.Tensor          # (B, T)
    scores: torch.Tensor            # (B, T)
    alarm_time: torch.Tensor        # (B,) first t with M_t ≥ 1/α, or -1
    alarmed: torch.Tensor           # (B,) bool
    features: dict[str, torch.Tensor] = field(default_factory=dict)


KAPPA_GRID: tuple[float, ...] = tuple(k / 20 for k in range(1, 20))   # 0.05, 0.10, …, 0.95


def mixture_e_value(p: torch.Tensor, kappas: tuple[float, ...] = KAPPA_GRID) -> torch.Tensor:
    """Simple-mixture conformal e-value  e(p) = mean_κ κ p^{κ−1}  over a κ grid.

    Each κ p^{κ−1} integrates to one over p ∈ (0,1], so the mixture is an
    e-value (E[e] = 1 under uniform p) and is numerically stable for all p;
    the continuous-mixture closed form (1 − p + p ln p)/(p ln² p) is its
    κ → U(0,1) limit but suffers cancellation near p = 1."""
    p = p.clamp(1e-12, 1.0).to(torch.float64)
    lp = torch.log(p)
    ks = torch.tensor(kappas, dtype=torch.float64, device=p.device)
    e = (ks * torch.exp((ks - 1.0) * lp.unsqueeze(-1))).mean(-1)
    return e.to(torch.float32)


class ConformalGuard:
    def __init__(self, model: nn.Module, cfg: ConformalGuardConfig = ConformalGuardConfig()) -> None:
        self.model = model
        self.cfg = cfg
        self.calib_scores: torch.Tensor | None = None
        self.mu: torch.Tensor | None = None
        self.sd: torch.Tensor | None = None

    # -- telemetry -----------------------------------------------------------
    @torch.no_grad()
    def telemetry(self, ids: torch.Tensor | None = None, emb: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Per-token telemetry from token ids or directly from embeddings (attacks)."""
        m = self.model
        m.eval()
        if emb is not None:
            m.encode_from_embeddings(emb)
        else:
            m(ids)
        tr = list(m.blocks)[self.cfg.layer].ssm.last_trace
        cs = m.constraints
        delta_sat = tr.delta.mean(-1) / cs.delta_max                              # (B, T)
        h_prev = torch.cat([tr.h_norm[:, :1] * 0 + 1e-6, tr.h_norm[:, :-1]], dim=1)
        norm_loss = (1.0 - tr.h_norm / (h_prev + 1e-6)).clamp(min=0.0)            # (B, T)
        injection = tr.injection_norm / (h_prev + 1e-6)                            # (B, T)
        b = self.cfg.burn_in
        norm_loss[:, :b] = 0.0; injection[:, :b] = 0.0
        return {"delta_sat": delta_sat, "norm_loss": norm_loss, "injection": injection}

    def _scores(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        F = torch.stack([feats["delta_sat"], feats["norm_loss"], feats["injection"]], dim=-1)  # (B, T, 3)
        z = (F - self.mu) / self.sd
        w = torch.tensor(self.cfg.weights, device=F.device)
        return (w * z).amax(dim=-1)                                                # (B, T)

    # -- calibration -----------------------------------------------------------
    @torch.no_grad()
    def fit(self, clean_ids: torch.Tensor, batch: int = 16) -> "ConformalGuard":
        feats_all = {"delta_sat": [], "norm_loss": [], "injection": []}
        for i in range(0, clean_ids.size(0), batch):
            f = self.telemetry(clean_ids[i: i + batch])
            for k in feats_all:
                feats_all[k].append(f[k])
        feats = {k: torch.cat(v) for k, v in feats_all.items()}
        F = torch.stack([feats["delta_sat"], feats["norm_loss"], feats["injection"]], dim=-1)
        self.mu = F.reshape(-1, 3).mean(0); self.sd = F.reshape(-1, 3).std(0) + 1e-6
        self.calib_scores = self._scores(feats).reshape(-1).sort().values
        return self

    # -- monitoring ------------------------------------------------------------
    @torch.no_grad()
    def monitor(self, ids: torch.Tensor | None = None, emb: torch.Tensor | None = None,
                feats: dict[str, torch.Tensor] | None = None) -> MonitorResult:
        """Run the e-process on token ids, on embeddings, or on precomputed telemetry."""
        if self.calib_scores is None:
            raise RuntimeError("call fit() on clean sequences first")
        if feats is None:
            feats = self.telemetry(ids, emb)
        A = self._scores(feats)                                                    # (B, T)
        n = self.calib_scores.numel()
        # p_t = (1 + #{calib ≥ A_t}) / (n + 1)
        ge = n - torch.searchsorted(self.calib_scores, A.reshape(-1).contiguous(), right=False)
        p = ((1.0 + ge.float()) / (n + 1.0)).reshape(A.shape)
        e = mixture_e_value(p)
        thr = 1.0 / self.cfg.alpha
        B, T = e.shape
        M = torch.empty_like(e); alarm_time = torch.full((B,), -1, dtype=torch.long, device=e.device)
        m_prev = torch.ones(B, device=e.device)
        for t in range(T):
            m_now = m_prev.pow(self.cfg.forgetting) * e[:, t]
            M[:, t] = m_now
            newly = (m_now >= thr) & (alarm_time < 0)
            alarm_time[newly] = t
            m_prev = m_now
        return MonitorResult(e_values=e, martingale=M, p_values=p, scores=A,
                             alarm_time=alarm_time, alarmed=alarm_time >= 0, features=feats)

    # -- telemetry feedback (Module 5 → Module 2 / dashboard) -------------------
    def recommend_delta_max(self, result: MonitorResult) -> float:
        cs = self.model.constraints
        return cs.delta_max * (self.cfg.tighten_factor if bool(result.alarmed.any()) else 1.0)

    @torch.no_grad()
    def dashboard_payload(self, ids: torch.Tensor, logits: torch.Tensor, l_loc: torch.Tensor,
                          result: MonitorResult) -> list[dict]:
        """(ŷ, ε*, M_T, alarm) per sequence — the OUTPUT node of the framework figure."""
        from ..certificates.local_lipschitz import ll_radius
        eps = ll_radius(logits, l_loc)
        pred = logits.argmax(-1)
        cs = self.model.constraints
        out = []
        for b in range(ids.size(0)):
            out.append({
                "prediction": int(pred[b]),
                "ll_radius_epsilon_star": float(eps[b]),
                "martingale_final": float(result.martingale[b, -1]),
                "martingale_max": float(result.martingale[b].max()),
                "alarm": bool(result.alarmed[b]),
                "alarm_token": int(result.alarm_time[b]),
                "delta_saturation_max": float(result.features["delta_sat"][b].max()),
                "norm_loss_max": float(result.features["norm_loss"][b].max()),
                "recommended_delta_max": cs.delta_max * (self.cfg.tighten_factor if bool(result.alarmed[b]) else 1.0),
                "certified_ell_star_alpha05": cs.ell_star(4.0, 0.5),
            })
        return out
