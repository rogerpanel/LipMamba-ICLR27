"""Adaptive white-box attack against the step clamp (TODO 2 of the manuscript).

Following Athalye et al. (2018) and Carlini et al. (2019), a certified-defence
paper must evaluate against an adversary that knows the defence.  Here the
defence is the two-sided clamp on Δ_t; the adaptive attacker therefore
optimises the trigger *directly against Δ_t and the state*, not against the
loss alone.  Three objectives are provided:

``saturate``   maximise mean Δ_t over trigger positions (push every step to
               Δ_max so that Ā_t = exp(Δ_t A) is as small as the clamp
               allows) **and** minimise the post-trigger state norm —
               the HiSPA objective re-targeted at the clamp.
``overwrite``  keep ‖h_T‖ close to the clean norm while maximising the
               *distance* ‖h_T − h_T^clean‖ — the content-overwrite adversary
               that Remark 5 says Theorem 2 does not exclude.  Success here
               is evidence that the retention bound is not behavioural.
``margin``     minimise the classification margin of the clean label (the
               standard adaptive objective for a GloRo head).

Each objective is available in a continuous (embedding-space, PGD) variant
and a discrete GCG-style variant (gradient-guided token substitution with a
candidate set from the top-k embedding gradient, Zou et al. 2023).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdaptiveClampConfig:
    objective: str = "saturate"       # saturate | overwrite | margin
    trigger_length: int = 24
    n_steps: int = 200
    lr: float = 5e-2
    norm_budget: float = 1.0          # ℓ2 budget per token of the trigger embedding
    lambda_norm: float = 1.0          # weight of the state-norm term in `saturate`
    lambda_keep: float = 10.0         # weight of the norm-keeping term in `overwrite`
    # discrete GCG
    discrete: bool = False
    top_k: int = 64
    candidates_per_step: int = 32
    target_position: int | None = None   # default: last trigger token
    layer: int = -1                      # which block's state to attack (-1 = last)


class AdaptiveClampAttack:
    def __init__(self, model: nn.Module, cfg: AdaptiveClampConfig) -> None:
        self.model = model
        self.cfg = cfg

    # -- state access -------------------------------------------------------
    def _run(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward embeddings; return (logits, mean Δ over trigger, state-norm at target)."""
        m = self.model
        blocks = list(m.blocks)
        h = emb
        target_block = blocks[self.cfg.layer]
        for blk in blocks:
            h = blk(h)
        logits = m.cls_head(m.norm_f(h)[:, -1]) if m.cls_head is not None else m.lm_head(m.norm_f(h)[:, -1])
        tr = target_block.ssm.last_trace
        T = emb.size(1); L = self.cfg.trigger_length
        delta_mean = tr.delta[:, T - L:].mean(dim=(1, 2)) if tr is not None else torch.zeros(emb.size(0))
        tp = self.cfg.target_position if self.cfg.target_position is not None else T - 1
        hnorm = tr.h_norm[:, tp] if tr is not None else torch.zeros(emb.size(0))
        return logits, delta_mean, hnorm

    def _differentiable_state(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable Δ_t and final SSM state of the target block.

        Re-computes the target block's scan with grad so the attacker sees
        ∂Δ_t/∂trigger and ∂h_T/∂trigger through the clamp."""
        m = self.model
        blocks = list(m.blocks)
        idx = self.cfg.layer % len(blocks)
        h = emb
        for blk in blocks[:idx]:
            h = blk(h)
        blk = blocks[idx]
        x_n = blk.norm(h)
        x_proj = blk.in_proj_x(x_n)
        T = x_proj.size(1)
        xc = F.silu(blk.conv(x_proj.transpose(1, 2))[..., :T].transpose(1, 2))
        ssm = blk.ssm
        x = ssm.input_clip(xc)
        delta = ssm.compute_delta(x)
        b_t = ssm.x_to_b(x)
        a_bar = ssm.A.discretise(delta)
        b_bar = delta.unsqueeze(-1) * b_t.unsqueeze(-2)
        state = x.new_zeros(x.size(0), x.size(2), ssm.cfg.state_dim)
        tp = self.cfg.target_position if self.cfg.target_position is not None else T - 1
        h_target = None
        for t in range(T):
            state = a_bar[:, t] * state + b_bar[:, t] * x[:, t].unsqueeze(-1)
            if t == tp:
                h_target = state
        logits = m.logits_from_embeddings(emb)
        L = self.cfg.trigger_length
        return logits, delta[:, T - L:].mean(dim=(1, 2)), h_target.flatten(1)

    # -- objectives ----------------------------------------------------------
    def _loss(self, emb: torch.Tensor, clean_state: torch.Tensor, clean_norm: torch.Tensor,
              labels: torch.Tensor | None) -> torch.Tensor:
        logits, delta_mean, h_t = self._differentiable_state(emb)
        hn = h_t.norm(dim=-1)
        cfg = self.cfg
        if cfg.objective == "saturate":
            return (-delta_mean + cfg.lambda_norm * hn / (clean_norm + 1e-12)).mean()
        if cfg.objective == "overwrite":
            dist = (h_t - clean_state).norm(dim=-1) / (clean_norm + 1e-12)
            keep = ((hn - clean_norm) / (clean_norm + 1e-12)).pow(2)
            return (-dist + cfg.lambda_keep * keep).mean()
        if cfg.objective == "margin":
            if labels is None:
                labels = logits.argmax(-1)
            z_y = logits.gather(1, labels.unsqueeze(-1)).squeeze(-1)
            masked = logits.clone(); masked.scatter_(1, labels.unsqueeze(-1), float("-inf"))
            return (z_y - masked.max(-1).values).mean()
        raise ValueError(cfg.objective)

    # -- continuous attack --------------------------------------------------------
    @torch.enable_grad()
    def attack(self, prefix_ids: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        m = self.model
        m.eval()
        pre = m.embed_tokens(prefix_ids).detach()
        B, Tp, D = pre.shape
        with torch.no_grad():
            pad = torch.zeros(B, cfg.trigger_length, D, device=pre.device)
            _, _, clean_state = self._differentiable_state(torch.cat([pre, pad], 1))
            clean_state = clean_state.detach()
            clean_norm = clean_state.norm(dim=-1)
            clean_logits = m.logits_from_embeddings(pre)
        if labels is None:
            labels = clean_logits.argmax(-1)

        trig = (torch.randn(B, cfg.trigger_length, D, device=pre.device) * 0.01).requires_grad_(True)
        opt = torch.optim.Adam([trig], lr=cfg.lr)
        for _ in range(cfg.n_steps):
            opt.zero_grad(set_to_none=True)
            loss = self._loss(torch.cat([pre, trig], 1), clean_state, clean_norm, labels)
            loss.backward()
            opt.step()
            with torch.no_grad():
                n = trig.norm(dim=-1, keepdim=True)
                trig.mul_(torch.clamp(cfg.norm_budget / (n + 1e-12), max=1.0))
        return trig.detach(), self.report(pre, trig.detach(), clean_state, clean_norm, labels, clean_logits)

    # -- discrete GCG attack ------------------------------------------------------
    @torch.enable_grad()
    def attack_discrete(self, prefix_ids: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        m = self.model
        m.eval()
        E = m.embed_tokens.weight
        V = E.size(0)
        B, Tp = prefix_ids.shape
        pre = m.embed_tokens(prefix_ids).detach()
        trig_ids = torch.randint(0, V, (B, cfg.trigger_length), device=prefix_ids.device)
        with torch.no_grad():
            _, _, clean_state = self._differentiable_state(torch.cat([pre, torch.zeros(B, cfg.trigger_length, pre.size(-1), device=pre.device)], 1))
            clean_state = clean_state.detach(); clean_norm = clean_state.norm(dim=-1)
            clean_logits = m.logits_from_embeddings(pre)
        if labels is None:
            labels = clean_logits.argmax(-1)

        for _ in range(cfg.n_steps):
            one_hot = F.one_hot(trig_ids, V).float().requires_grad_(True)
            trig_emb = one_hot @ E
            loss = self._loss(torch.cat([pre, trig_emb], 1), clean_state, clean_norm, labels)
            grad, = torch.autograd.grad(loss, one_hot)             # (B, L, V)
            with torch.no_grad():
                pos = torch.randint(0, cfg.trigger_length, (B,), device=prefix_ids.device)
                cand = (-grad[torch.arange(B), pos]).topk(cfg.top_k, dim=-1).indices   # descent direction
                best_loss = torch.full((B,), float("inf"), device=prefix_ids.device)
                best_tok = trig_ids[torch.arange(B), pos].clone()
                for c in range(min(cfg.candidates_per_step, cfg.top_k)):
                    trial = trig_ids.clone()
                    trial[torch.arange(B), pos] = cand[:, c]
                    with torch.enable_grad():
                        l = self._loss_per_sample(torch.cat([pre, m.embed_tokens(trial)], 1), clean_state, clean_norm, labels)
                    better = l < best_loss
                    best_loss = torch.where(better, l, best_loss)
                    best_tok = torch.where(better, cand[:, c], best_tok)
                trig_ids[torch.arange(B), pos] = best_tok
        trig_emb = m.embed_tokens(trig_ids).detach()
        rep = self.report(pre, trig_emb, clean_state, clean_norm, labels, clean_logits)
        rep["trigger_ids"] = trig_ids.tolist()
        return trig_ids, rep

    def _loss_per_sample(self, emb, clean_state, clean_norm, labels) -> torch.Tensor:
        logits, delta_mean, h_t = self._differentiable_state(emb)
        hn = h_t.norm(dim=-1); cfg = self.cfg
        if cfg.objective == "saturate":
            return (-delta_mean + cfg.lambda_norm * hn / (clean_norm + 1e-12)).detach()
        if cfg.objective == "overwrite":
            dist = (h_t - clean_state).norm(dim=-1) / (clean_norm + 1e-12)
            keep = ((hn - clean_norm) / (clean_norm + 1e-12)).pow(2)
            return (-dist + cfg.lambda_keep * keep).detach()
        z_y = logits.gather(1, labels.unsqueeze(-1)).squeeze(-1)
        masked = logits.clone(); masked.scatter_(1, labels.unsqueeze(-1), float("-inf"))
        return (z_y - masked.max(-1).values).detach()

    # -- reporting ---------------------------------------------------------------
    @torch.no_grad()
    def report(self, pre, trig_emb, clean_state, clean_norm, labels, clean_logits) -> dict:
        emb = torch.cat([pre, trig_emb], 1)
        logits, delta_mean, h_t = self._differentiable_state(emb)
        hn = h_t.norm(dim=-1)
        retention = hn / (clean_norm + 1e-12)
        overwrite = (h_t - clean_state).norm(dim=-1) / (clean_norm + 1e-12)
        cs = self.model.constraints
        return {
            "objective": self.cfg.objective,
            "trigger_length": self.cfg.trigger_length,
            "mean_delta_over_trigger": float(delta_mean.mean()),
            "delta_max": cs.delta_max,
            "delta_saturation_fraction": float((delta_mean / cs.delta_max).mean()),
            "retention_ratio_mean": float(retention.mean()),
            "retention_ratio_min": float(retention.min()),
            "overwrite_distance_mean": float(overwrite.mean()),
            "label_flip_rate": float((logits.argmax(-1) != labels).float().mean()),
            "clean_label_agreement": float((clean_logits.argmax(-1) == labels).float().mean()),
        }
