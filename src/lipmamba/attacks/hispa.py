"""HiSPA — hidden-state poisoning attacks (Le Mercier, Develder & Demeester,
arXiv:2601.01972, 2026; unrefereed preprint).

Two published variants are reproduced, plus the continuous embedding-space
version used for training-time adversarial examples:

* **Z-HiSPA (zero-shot).**  No optimisation: pick the vocabulary tokens that
  individually maximise the step Δ_t of the target layer (a single forward
  scan over the vocabulary) and repeat them ℓ times.  This is the
  black-box, architecture-only attack of Def. 1.
* **M-HiSPA (optimised).**  A genetic algorithm over token triggers whose
  fitness is the post-trigger norm ratio ‖h_{t0+ℓ}‖/‖h_{t0}‖ (lower is
  better for the attacker), with tournament selection, uniform crossover
  and per-position mutation.  Black-box: uses forward passes only.
* **Continuous HiSPA.**  PGD on trigger embeddings minimising the state
  norm (white-box; used for adversarial training and Figure 2's lower bound).

All three report the observed retention ratio α, so they can be compared
directly with Theorem 2's per-input ℓ*.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HiSPAConfig:
    trigger_length: int = 24
    target_alpha: float = 0.05
    layer: int = -1
    # continuous
    n_steps: int = 200
    lr: float = 5e-2
    norm_budget: float = 1.0
    init: str = "gaussian"
    # M-HiSPA (GA)
    population: int = 64
    generations: int = 50
    mutation_rate: float = 0.1
    tournament: int = 4
    vocab_subset: int | None = 4096     # restrict GA vocabulary (None = full)


class HiSPAAttack:
    def __init__(self, model: nn.Module, cfg: HiSPAConfig) -> None:
        self.model = model
        self.cfg = cfg

    # -- helpers ----------------------------------------------------------------
    @torch.no_grad()
    def state_norm_at(self, emb: torch.Tensor, position: int) -> torch.Tensor:
        """‖h_t‖ (max over channels) of the target block at ``position``."""
        m = self.model
        h = emb
        for blk in m.blocks:
            h = blk(h)
        tr = list(m.blocks)[self.cfg.layer].ssm.last_trace
        return tr.h_norm[:, position]

    @torch.no_grad()
    def _clean_norm(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        return self.state_norm_at(self.model.embed_tokens(prefix_ids), prefix_ids.size(1) - 1)

    @torch.no_grad()
    def _ratio_for_triggers(self, prefix_ids: torch.Tensor, trig_ids: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        ids = torch.cat([prefix_ids, trig_ids], 1)
        return self.state_norm_at(self.model.embed_tokens(ids), ids.size(1) - 1) / (clean + 1e-12)

    # -- Z-HiSPA -----------------------------------------------------------------
    @torch.no_grad()
    def rank_vocab_by_delta(self, context_ids: torch.Tensor, batch: int = 512) -> torch.Tensor:
        """Mean Δ_t of the target layer at the last position for every token appended
        to ``context_ids`` (1, T).  Returns (V,) sorted indices, highest Δ first."""
        m = self.model
        V = m.embed_tokens.weight.size(0)
        scores = torch.empty(V, device=context_ids.device)
        ctx = context_ids.expand(batch, -1)
        for s in range(0, V, batch):
            toks = torch.arange(s, min(s + batch, V), device=context_ids.device)
            ids = torch.cat([ctx[: toks.numel()], toks.unsqueeze(1)], 1)
            h = m.embed_tokens(ids)
            for blk in m.blocks:
                h = blk(h)
            tr = list(m.blocks)[self.cfg.layer].ssm.last_trace
            scores[s: s + toks.numel()] = tr.delta[:, -1].mean(-1)
        return scores.argsort(descending=True)

    @torch.no_grad()
    def z_hispa(self, prefix_ids: torch.Tensor, n_top: int = 4) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        ranked = self.rank_vocab_by_delta(prefix_ids[:1])
        top = ranked[:n_top]
        trig = top[torch.arange(cfg.trigger_length, device=top.device) % n_top].unsqueeze(0).expand(prefix_ids.size(0), -1)
        clean = self._clean_norm(prefix_ids)
        ratio = self._ratio_for_triggers(prefix_ids, trig, clean)
        return trig, {"variant": "Z-HiSPA", "alpha_mean": float(ratio.mean()), "alpha_min": float(ratio.min()),
                      "success_rate": float((ratio <= cfg.target_alpha).float().mean()),
                      "top_tokens": top.tolist()}

    # -- M-HiSPA (genetic) --------------------------------------------------------
    @torch.no_grad()
    def m_hispa(self, prefix_ids: torch.Tensor, seed: int = 0) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        g = torch.Generator(device="cpu").manual_seed(seed)
        m = self.model
        V = m.embed_tokens.weight.size(0)
        vocab = V if cfg.vocab_subset is None else min(V, cfg.vocab_subset)
        # warm-start half the population from Z-HiSPA's top tokens
        ranked = self.rank_vocab_by_delta(prefix_ids[:1])[:vocab]
        B = prefix_ids.size(0)
        clean = self._clean_norm(prefix_ids)
        pop = ranked[torch.randint(0, vocab, (cfg.population, cfg.trigger_length), generator=g)]
        pop[: cfg.population // 2] = ranked[torch.randint(0, 16, (cfg.population // 2, cfg.trigger_length), generator=g)]

        def fitness(p: torch.Tensor) -> torch.Tensor:          # lower = better for attacker
            out = torch.empty(p.size(0), device=prefix_ids.device)
            for i in range(p.size(0)):
                out[i] = self._ratio_for_triggers(prefix_ids, p[i].unsqueeze(0).expand(B, -1), clean).mean()
            return out

        fit = fitness(pop)
        history = [float(fit.min())]
        for _ in range(cfg.generations):
            # tournament selection
            idx = torch.randint(0, cfg.population, (cfg.population, cfg.tournament), generator=g)
            winners = idx[torch.arange(cfg.population), fit[idx].argmin(dim=1)]
            parents = pop[winners]
            # uniform crossover
            mask = torch.rand(cfg.population, cfg.trigger_length, generator=g) < 0.5
            children = torch.where(mask, parents, parents.roll(1, dims=0))
            # mutation
            mut = torch.rand(cfg.population, cfg.trigger_length, generator=g) < cfg.mutation_rate
            children = torch.where(mut, ranked[torch.randint(0, vocab, children.shape, generator=g)], children)
            cfit = fitness(children)
            # elitism: keep best of parents ∪ children
            allp = torch.cat([pop, children]); allf = torch.cat([fit, cfit])
            keep = allf.argsort()[: cfg.population]
            pop, fit = allp[keep], allf[keep]
            history.append(float(fit.min()))
        best = pop[0].unsqueeze(0).expand(B, -1)
        ratio = self._ratio_for_triggers(prefix_ids, best, clean)
        return best, {"variant": "M-HiSPA", "alpha_mean": float(ratio.mean()), "alpha_min": float(ratio.min()),
                      "success_rate": float((ratio <= cfg.target_alpha).float().mean()), "history": history}

    # -- continuous (white-box, embedding space) ---------------------------------------
    @torch.enable_grad()
    def attack(self, prefix_ids: torch.Tensor, target_position: int | None = None) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        m = self.model
        m.eval()
        pre = m.embed_tokens(prefix_ids).detach()
        B, Tp, D = pre.shape
        dev = pre.device
        if cfg.init == "gaussian":
            delta = torch.randn(B, cfg.trigger_length, D, device=dev) * 0.01
        elif cfg.init == "uniform":
            delta = (torch.rand(B, cfg.trigger_length, D, device=dev) - 0.5) * 0.1
        else:
            delta = torch.zeros(B, cfg.trigger_length, D, device=dev)
        delta.requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=cfg.lr)
        tp = target_position if target_position is not None else Tp + cfg.trigger_length - 1

        blocks = list(m.blocks)
        idx = cfg.layer % len(blocks)

        def target_state_norm(emb: torch.Tensor) -> torch.Tensor:
            h = emb
            for blk in blocks[:idx]:
                h = blk(h)
            blk = blocks[idx]
            x_n = blk.norm(h); x_proj = blk.in_proj_x(x_n); T = x_proj.size(1)
            xc = torch.nn.functional.silu(blk.conv(x_proj.transpose(1, 2))[..., :T].transpose(1, 2))
            ssm = blk.ssm; x = ssm.input_clip(xc)
            dl = ssm.compute_delta(x); b_t = ssm.x_to_b(x)
            a_bar = ssm.A.discretise(dl); b_bar = dl.unsqueeze(-1) * b_t.unsqueeze(-2)
            s = x.new_zeros(x.size(0), x.size(2), ssm.cfg.state_dim)
            for t in range(tp + 1):
                s = a_bar[:, t] * s + b_bar[:, t] * x[:, t].unsqueeze(-1)
            return s.norm(dim=-1).amax(dim=-1)

        with torch.no_grad():
            clean = self._clean_norm(prefix_ids)
        for _ in range(cfg.n_steps):
            opt.zero_grad(set_to_none=True)
            loss = target_state_norm(torch.cat([pre, delta], 1)).mean()
            loss.backward(); opt.step()
            with torch.no_grad():
                n = delta.norm(dim=-1, keepdim=True)
                delta.mul_(torch.clamp(cfg.norm_budget / (n + 1e-12), max=1.0))
        with torch.no_grad():
            final = target_state_norm(torch.cat([pre, delta], 1))
        ratio = final / (clean + 1e-12)
        return delta.detach(), {"variant": "continuous", "final_norm": float(final.mean()),
                                "clean_norm": float(clean.mean()), "alpha": float(ratio.mean()),
                                "alpha_min": float(ratio.min()),
                                "success": bool((ratio <= cfg.target_alpha).any()),
                                "success_rate": float((ratio <= cfg.target_alpha).float().mean())}
