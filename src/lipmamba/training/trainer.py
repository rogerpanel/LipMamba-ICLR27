"""LipMamba trainer — Algorithm 2 (PAC-Bayes adversarial training).

Per step:
  1. clean mini-batch;
  2. choose the Lipschitz constant L used in the margin term and the
     PAC-Bayes gap term according to ``lipschitz_mode``:
       * ``"local"``  — Appendix-E estimator on the batch (the manuscript's
                        choice; costs 8 restarts × 20 PGD steps per batch);
       * ``"global"`` — worst-case Theorem-1 product (vacuous at depth,
                        provided for the ablation / sanity check);
       * ``"fixed"``  — a user-supplied constant ``l_fixed``;
  3. margin-augmented cross-entropy (GloRo) or empirical adversarial loss
     under PGD / HiSPA;
  4. + ½ L_ℓ L ε_train + β·complexity(KL);
  5. AdamW, grad-clip 1.0, cosine LR; spectral σ̂ refreshed by power
     iteration inside every forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..certificates.local_lipschitz import LocalLipschitzConfig, local_lipschitz_estimate
from ..certificates.pac_bayes import PACBayesConfig
from ..utils.checkpoint import save_checkpoint
from ..utils.logging import get_logger
from .adv_objective import adversarial_loss, margin_adversarial_loss
from .optim import build_optimizer, build_scheduler
from .pac_objective import pac_bayes_total_loss


@dataclass
class TrainerConfig:
    max_steps: int = 100_000
    warmup_steps: int = 1_000
    lr: float = 2e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    epsilon_train: float = 0.18
    log_every: int = 100
    eval_every: int = 5_000
    save_every: int = 5_000
    out_dir: str = "runs/lipmamba"
    pac_bayes: PACBayesConfig = field(default_factory=PACBayesConfig)
    use_margin_objective: bool = True
    use_attack: str | None = None            # {"pgd", "hispa", None}
    attack_kwargs: dict = field(default_factory=dict)
    lipschitz_mode: str = "local"            # {"local", "global", "fixed"}
    l_fixed: float = 1.0
    local_lipschitz: LocalLipschitzConfig = field(default_factory=lambda: LocalLipschitzConfig(n_restarts=2, n_steps=5))
    seed: int = 42


class LipMambaTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader | None = None,
                 prior_params: torch.Tensor | None = None, cfg: TrainerConfig | None = None) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg or TrainerConfig()
        self.optimizer = build_optimizer(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.scheduler = build_scheduler(self.optimizer, warmup_steps=self.cfg.warmup_steps, max_steps=self.cfg.max_steps)
        self.prior_params = prior_params
        self.logger = get_logger("lipmamba.train")
        Path(self.cfg.out_dir).mkdir(parents=True, exist_ok=True)

    def _attack_fn(self):
        kind = self.cfg.use_attack
        if kind is None:
            return None
        if kind == "pgd":
            from ..attacks.pgd import PGDAttack, PGDConfig
            att = PGDAttack(self.model, PGDConfig(**self.cfg.attack_kwargs))
            return lambda model, batch: att.attack(batch["input_ids"], batch["labels"])
        if kind == "hispa":
            from ..attacks.hispa import HiSPAAttack, HiSPAConfig
            att = HiSPAAttack(self.model, HiSPAConfig(**self.cfg.attack_kwargs))
            return lambda model, batch: torch.cat([model.embed_tokens(batch["input_ids"]).detach(),
                                                   att.attack(batch["input_ids"])[0]], 1)
        raise ValueError(kind)

    def _lipschitz(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mode = self.cfg.lipschitz_mode
        dev = next(self.model.parameters()).device
        if mode == "global":
            return self.model.network_lipschitz_bound().to(dev)
        if mode == "fixed":
            return torch.tensor(self.cfg.l_fixed, device=dev)
        if mode == "local":
            emb = self.model.embed_tokens(batch["input_ids"]).detach()
            was_training = self.model.training
            l = local_lipschitz_estimate(self.model, emb, self.cfg.local_lipschitz)
            self.model.train(was_training)
            return l.detach()          # per-sample (B,)
        raise ValueError(mode)

    def _step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        dev = next(self.model.parameters()).device
        batch = {k: v.to(dev) for k, v in batch.items()}
        l = self._lipschitz(batch)

        if self.cfg.use_margin_objective and self.model.cls_head is not None:
            adv = margin_adversarial_loss(self.model, batch, l_net=l, epsilon=self.cfg.epsilon_train)
        else:
            adv = adversarial_loss(self.model, batch, self._attack_fn())

        prior = self.prior_params
        if prior is None:
            from ..certificates.pac_bayes import flatten_constrained_parameters
            prior = flatten_constrained_parameters(self.model).detach().clone()
            self.prior_params = prior   # cold start: KL = 0 at step 0, grows with drift

        comp = pac_bayes_total_loss(self.model, batch, empirical_adv_loss=adv,
                                    l_net=l.mean() if l.dim() else l, prior_params=prior, cfg=self.cfg.pac_bayes)
        loss = comp["loss"]
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step(); self.scheduler.step()
        return {"loss": float(loss), "adv": float(comp["adv_loss"]), "kl": float(comp["kl"]),
                "lip": float(comp["lipschitz_term"]), "L": float(l.mean() if l.dim() else l)}

    def train(self) -> None:
        cfg = self.cfg
        step = 0
        it = iter(self.train_loader)
        while step < cfg.max_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.train_loader); batch = next(it)
            st = self._step(batch); step += 1
            if step % cfg.log_every == 0:
                self.logger.info("step=%d loss=%.4f adv=%.4f L(%s)=%.3g kl=%.2f",
                                 step, st["loss"], st["adv"], cfg.lipschitz_mode, st["L"], st["kl"])
            if step % cfg.eval_every == 0 and self.val_loader is not None:
                self.evaluate(step)
            if step % cfg.save_every == 0:
                save_checkpoint(self.model, self.optimizer, step=step, path=str(Path(cfg.out_dir) / f"step{step}.pt"))
        save_checkpoint(self.model, self.optimizer, step=step, path=str(Path(cfg.out_dir) / "final.pt"))

    @torch.no_grad()
    def evaluate(self, step: int | None = None) -> dict[str, float]:
        self.model.eval()
        dev = next(self.model.parameters()).device
        ce, n = 0.0, 0
        for batch in self.val_loader or []:
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = self.model(batch["input_ids"])
            if "cls_logits" in out:
                loss = nn.functional.cross_entropy(out["cls_logits"], batch["labels"])
            else:
                loss = nn.functional.cross_entropy(out["lm_logits"].reshape(-1, out["lm_logits"].size(-1)),
                                                   batch["labels"].reshape(-1))
            ce += float(loss) * batch["input_ids"].size(0); n += batch["input_ids"].size(0)
        res = {"ce": ce / max(1, n), "log10_L_global": self.model.log10_network_lipschitz_bound()}
        self.logger.info("[eval] step=%s ce=%.4f log10(L_global)=%.1f", step, res["ce"], res["log10_L_global"])
        return res
