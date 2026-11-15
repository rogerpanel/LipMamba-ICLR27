#!/usr/bin/env python
"""Matched-training robustness study at 130M parameters (one GPU; Colab A100 or L4).

Every arm starts from the SAME public checkpoint (``state-spaces/mamba-130m-hf``)
and is fine-tuned with the SAME data, optimiser, steps and seeds:

    mamba_released   public checkpoint, no fine-tuning (reference only)
    mamba            unconstrained, clean fine-tuning
    mamba_at         unconstrained, adversarial fine-tuning
    lipcal           all constraints, constants calibrated to the checkpoint, clean fine-tuning
    lipcal_at        the same with adversarial fine-tuning
    lipcal_no_sn / lipcal_no_clamp / lipcal_free_A / lipcal_no_clip    single-constraint ablations
    lip, lip_at      all constraints with the PAPER's constants (s=1, Δ in [1e-3, 0.5), λ in [0.05, 1])

"Calibrated" constants are each block's clean-data envelope of the public checkpoint (its own spectral
norms and eigenvalue range, and quantiles of its step sizes and input norms), so the constrained model
starts as almost the same function and every constraint is a bound that neither fine-tuning nor an
attacker can exceed.  With the paper's constants the projection changes the pretrained function
drastically (initial perplexity above 1e60), so those arms measure recovery as much as robustness.

Task: natural-language fact recall.  The context states a code word for four
names and one later update; then comes filler text from WikiText, then a
trigger slot, then a question about one name.  The answer is read from the
next-token logits over the code words.  In "update" examples the queried
name is the one that was re-assigned, so the model must revise a stored value.

Attacks, each RE-OPTIMISED white-box against every fine-tuned checkpoint and
every example, with trigger tokens drawn from the whole vocabulary except
names, code words and special tokens:
  * 8 random triggers;
  * a zero-shot trigger repeating the token with the largest step size;
  * gradient-guided token search (GCG) on the answer margin, end to end;
  * the same search on the state norm (the hidden-state-poisoning objective).
A continuous trigger optimised on the state norm gives the retention figures.

  python scripts/scale_study.py run --model lip --seed 0 --out runs/scale/lip_s0.json --preset a100
  python scripts/scale_study.py all --dir runs/scale --seeds 0 1 2 --preset a100
  python scripts/scale_study.py aggregate --dir runs/scale
  python scripts/scale_study.py run --model lip --tiny --out /tmp/t.json       # CPU smoke test, no download
"""
from __future__ import annotations

import argparse, json, math, os, re, subprocess, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

from lipmamba.pretrained import (ARCHS, apply_calibration, build_from_mamba, calibrate_constraints, calibrate_x_max,
                                 conversion_error, describe, hf_state_dict, reference_logits, refresh_spectral_norms,
                                 tiny_state_dict)

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Jack", "Kate", "Liam", "Mary",
         "Nora", "Oscar", "Paul", "Ruth", "Sam", "Tom"]
VALUES = ["red", "blue", "green", "black", "white", "gold", "silver", "pink", "brown", "gray", "orange", "purple",
          "yellow", "iron", "stone", "glass", "north", "south", "east", "west"]
CORE = ["mamba_released", "mamba", "lipcal", "mamba_at", "lipcal_at"]
PAPER = ["lip", "lip_at"]
ABLATIONS = ["lipcal_no_sn", "lipcal_no_clamp", "lipcal_free_A", "lipcal_no_clip"]
PRESETS = {  # explicit flags override these
    "a100": dict(bs=16, accum=1, grad_chunk=16, eval_chunk=256, steps=400, n_eval=64, iters=60, n_cand=32),
    "l4":   dict(bs=8, accum=1, grad_chunk=12, eval_chunk=192, steps=400, n_eval=48, iters=30, n_cand=16),
    "t4":   dict(bs=8, accum=2, grad_chunk=8, eval_chunk=128, steps=400, n_eval=48, iters=40, n_cand=32),   # 15-16 GB (Kaggle T4 / P100)
    "tiny": dict(bs=8, accum=1, grad_chunk=8, eval_chunk=64, steps=40, n_eval=8, iters=3, n_cand=8),
}


# --------------------------------------------------------------------------- data
class TinyTok:
    """Word-level stand-in for the real tokenizer (CPU smoke test)."""

    def __init__(self):
        self.v: dict[str, int] = {}

    def encode(self, s: str) -> list[int]:
        return [self.v.setdefault(w, len(self.v)) for w in re.findall(r" ?\w+|[^\w\s]|\n", s)]


class Task:
    def __init__(self, enc, n_fact: int = 4, n_values: int = 16):
        self.N_FACT = n_fact
        one = lambda s: (lambda t: t[0] if len(t) == 1 else None)(enc(s))
        self.names = [(w, one(" " + w)) for w in NAMES if one(" " + w) is not None][:12]
        self.values = [(w, one(" " + w)) for w in VALUES if one(" " + w) is not None][:n_values]
        assert len(self.names) >= min(8, n_fact + 1) and len(self.values) >= min(8, n_values), "tokenizer does not give single-token names/values"
        self.name_ids = torch.tensor([i for _, i in self.names]); self.val_ids = torch.tensor([i for _, i in self.values])
        self.head = enc("Records:")
        self.f_pre, self.f_mid, self.f_end = enc(" The code for"), enc(" is"), enc(".")
        self.u_pre, self.u_mid = enc(" Update: the code for"), enc(" is now")
        self.q_pre, self.q_post = enc("\nQuestion: what is the code for"), enc("? Answer:")
        banned = set()
        for w, _ in self.names + self.values:
            for form in {w, w.lower(), w.upper(), w.capitalize()}:
                for pre in ("", " "):
                    t = enc(pre + form)
                    if len(t) == 1:
                        banned.add(t[0])
        self._banned = banned

    def finalize(self, vocab_size: int, wiki: torch.Tensor, special_ids=()):
        self.vocab_size, self.wiki = vocab_size, wiki
        allowed = torch.ones(vocab_size, dtype=torch.bool)
        allowed[[i for i in set(self._banned) | set(special_ids) if i < vocab_size]] = False
        self.allowed = allowed; self.allowed_ids = allowed.nonzero().squeeze(1)
        return self

    def sample(self, n: int, g: torch.Generator, fill_len: int, trig_len: int, p_update: float = 0.3) -> dict:
        """ctx | fill | trig | qry.  ``trig`` holds benign text here; attacks overwrite it."""
        ri = lambda hi: int(torch.randint(0, hi, (1,), generator=g))
        ctx, qry, y, upd = [], [], [], []
        for _ in range(n):
            names = torch.randperm(len(self.names), generator=g)[: self.N_FACT].tolist()
            vals = [ri(len(self.values)) for _ in range(self.N_FACT)]
            qi = ri(self.N_FACT)
            is_upd = bool(torch.rand(1, generator=g) < p_update)
            j = qi if is_upd else (qi + 1 + ri(self.N_FACT - 1)) % self.N_FACT     # whose value is re-assigned
            new = (vals[j] + 1 + ri(len(self.values) - 1)) % len(self.values)
            sents = [self.f_pre + [self.names[a][1]] + self.f_mid + [self.values[v][1]] + self.f_end
                     for a, v in zip(names, vals)]
            sents.insert(j + 1 + ri(self.N_FACT - j), self.u_pre + [self.names[names[j]][1]] + self.u_mid + [self.values[new][1]] + self.f_end)
            ctx.append(self.head + [t for s in sents for t in s])
            qry.append(self.q_pre + [self.names[names[qi]][1]] + self.q_post)
            y.append(new if is_upd else vals[qi]); upd.append(is_upd)
        span = fill_len + trig_len
        starts = torch.randint(0, self.wiki.numel() - span - 1, (n,), generator=g)
        text = torch.stack([self.wiki[s: s + span] for s in starts.tolist()])
        return dict(ctx=torch.tensor(ctx), fill=text[:, :fill_len], trig=text[:, fill_len:], qry=torch.tensor(qry),
                    y=torch.tensor(y), upd=torch.tensor(upd))

    def random_trigger(self, n: int, L: int, g: torch.Generator) -> torch.Tensor:
        return self.allowed_ids[torch.randint(0, self.allowed_ids.numel(), (n, L), generator=g)]

    def lm_batch(self, n: int, T: int, g: torch.Generator) -> torch.Tensor:
        starts = torch.randint(0, self.wiki.numel() - T - 1, (n,), generator=g)
        return torch.stack([self.wiki[s: s + T] for s in starts.tolist()])


def load_data(a):
    """Returns (state_dict, hf_model_or_None, task, val_stream, decode)."""
    if a.tiny:
        tok = TinyTok(); task = Task(tok.encode, a.n_fact, a.n_values)
        n_word = len(tok.v); vocab = n_word + 200
        g = torch.Generator().manual_seed(123)
        wiki = torch.randint(n_word, vocab, (20_000,), generator=g); val = torch.randint(n_word, vocab, (4_096,), generator=g)
        return tiny_state_dict(vocab), None, task.finalize(vocab, wiki), val, lambda ids: " ".join(map(str, ids))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    streams = {}
    for split in ("train", "validation"):
        f = cache / f"wikitext2_{split}_{a.base.replace('/', '_')}.pt"
        if not f.exists():
            from datasets import load_dataset
            ds = None
            for repo in ("Salesforce/wikitext", "wikitext"):
                try:
                    ds = load_dataset(repo, "wikitext-2-raw-v1", split=split); break
                except Exception as e:                                   # noqa: BLE001
                    err = e
            if ds is None:
                raise RuntimeError(f"could not download WikiText-2: {err}")
            lines = [t for t in ds["text"] if t.strip()]
            ids = [i for k in range(0, len(lines), 512) for i in enc("".join(lines[k: k + 512]))]
            torch.save(torch.tensor(ids), f)
        streams[split] = torch.load(f)
    sd, hf = hf_state_dict(a.base, a.revision)
    task = Task(enc, a.n_fact, a.n_values).finalize(sd["backbone.embeddings.weight"].shape[0], streams["train"], tok.all_special_ids)
    return sd, hf, task, streams["validation"], lambda ids: tok.decode(ids)


# --------------------------------------------------------------------------- model helpers
def to_dev(ex: dict, dev) -> dict:
    return {k: v.to(dev) for k, v in ex.items()}


def cat_ids(ex: dict, trig: torch.Tensor | None = None, idx: torch.Tensor | None = None) -> torch.Tensor:
    s = (lambda v: v) if idx is None else (lambda v: v[idx])
    return torch.cat([s(ex["ctx"]), s(ex["fill"]), ex["trig"] if trig is None else trig, s(ex["qry"])], 1)


def margin(lg: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = lg.gather(1, y[:, None]).squeeze(1)
    return true - lg.scatter(1, y[:, None], float("-inf")).amax(1)


def set_keep(model, on: bool):
    for b in model.blocks:
        b.ssm.keep_state = on
        if not on:
            b.ssm._last_state = b.ssm._last_delta = b.ssm._state_seq = None


def scores(model, val_ids, y, t0: int, t1: int, ids=None, emb=None, want_norm: bool = False):
    """Answer margin, and (optionally) per-block log state-norm ratio between positions t0 and t1.
    t0 is the last token before the trigger, t1 the last trigger token."""
    set_keep(model, want_norm)
    h = model.encode(ids) if emb is None else model.encode_from_embeddings(emb)
    m = margin(model.lm_head(h[:, -1])[:, val_ids], y)
    logr = None
    if want_norm:
        logr = torch.stack([b.ssm._state_seq[:, t1].clamp_min(1e-30).log() - b.ssm._state_seq[:, t0].clamp_min(1e-30).log()
                            for b in model.blocks], 1)                                   # (n, n_blocks)
        set_keep(model, False)
    return m, logr


def fit(chunk: int, ex, L: int) -> int:
    """Chunk sizes are quoted for 96-token sequences; scale them so memory stays constant for longer ones."""
    return max(1, chunk * 96 // (ex["ctx"].size(1) + ex["fill"].size(1) + L + ex["qry"].size(1)))


@torch.no_grad()
def eval_trigger(model, task, ex, trig, chunk: int, want_norm: bool = False, idx=None):
    """No-grad margins (and log-ratios) for a trigger tensor; ``idx`` maps trigger rows to example rows."""
    n = trig.size(0); idx = torch.arange(n, device=trig.device) if idx is None else idx
    t0 = ex["ctx"].size(1) + ex["fill"].size(1) - 1; t1 = t0 + trig.size(1)
    ms, rs = [], []
    for c in torch.arange(n, device=trig.device).split(fit(chunk, ex, trig.size(1))):
        m, r = scores(model, task.val_dev, ex["y"][idx[c]], t0, t1, ids=cat_ids(ex, trig[c], idx[c]), want_norm=want_norm)
        ms.append(m); rs.append(r)
    return torch.cat(ms), (torch.cat(rs) if want_norm else None)


def trig_grad(model, task, ex, trig, rows, objective: str):
    """Gradient of the objective w.r.t. the trigger embeddings for example ``rows`` (trig is aligned with rows)."""
    E = model.embed_tokens.weight
    t0 = ex["ctx"].size(1) + ex["fill"].size(1) - 1; t1 = t0 + trig.size(1)
    emb_t = E[trig].detach().clone().requires_grad_(True)
    emb = torch.cat([E[torch.cat([ex["ctx"][rows], ex["fill"][rows]], 1)].detach(), emb_t, E[ex["qry"][rows]].detach()], 1)
    m, logr = scores(model, task.val_dev, ex["y"][rows], t0, t1, emb=emb, want_norm=objective == "norm")
    obj = m if objective == "margin" else logr.mean(1)
    return torch.autograd.grad(obj.sum(), emb_t)[0]


# --------------------------------------------------------------------------- attacks
def gcg(model, task, ex, L: int, objective: str, a, seed: int):
    """Greedy coordinate gradient search, per example.  Minimises the answer margin ('margin') or the mean
    over blocks of the log state-norm ratio across the trigger ('norm').  Returns the best trigger per example."""
    dev = ex["y"].device; n = ex["y"].numel(); K = a.n_cand
    g = torch.Generator().manual_seed(seed)
    E = model.embed_tokens.weight.detach(); banned = ~task.allowed.to(dev)
    trig = task.random_trigger(n, L, g).to(dev)
    cur_m, cur_r = eval_trigger(model, task, ex, trig, a.eval_chunk, want_norm=objective == "norm")
    cur = cur_m.clone() if objective == "margin" else cur_r.mean(1)
    for _ in range(a.iters if objective == "margin" else max(1, a.iters // 3)):   # the norm search never stops early
        act = (cur > 0).nonzero().squeeze(1) if objective == "margin" else torch.arange(n, device=dev)
        if act.numel() == 0:
            break
        grad = torch.cat([trig_grad(model, task, ex, trig[c], c, objective) for c in act.split(fit(a.grad_chunk, ex, L))])   # (a, L, d)
        na = act.numel()
        pos = torch.randint(0, L, (na, K), generator=g).to(dev)
        tok = torch.empty(na, K, dtype=torch.long, device=dev)
        pick = torch.randint(0, a.topk, (na, K), generator=g).to(dev)
        for c in torch.arange(na, device=dev).split(16):                 # first-order decrease of the objective per token
            gp = grad[c].gather(1, pos[c][..., None].expand(-1, -1, grad.size(-1)))
            sc = -(gp @ E.T); sc[..., banned] = float("-inf")
            tok[c] = sc.topk(a.topk, -1).indices.gather(2, pick[c][..., None]).squeeze(2)
        cand = trig[act][:, None, :].repeat(1, K, 1); cand.scatter_(2, pos[..., None], tok[..., None])
        m_c, r_c = eval_trigger(model, task, ex, cand.view(na * K, L), a.eval_chunk, want_norm=objective == "norm",
                                idx=act.repeat_interleave(K))
        val = (m_c if objective == "margin" else r_c.mean(1)).view(na, K)
        best, bi = val.min(1); better = best < cur[act]
        rows = act[better]
        trig[rows] = cand[better, bi[better]]; cur[rows] = best[better]
    return trig


def soft_norm_trigger(model, task, ex, L: int, a, seed: int):
    """Continuous trigger (free embedding vectors, Adam) minimising the state-norm ratio.  Not a token attack:
    it is the strongest state-erasing input we can build, used only to measure retention."""
    dev = ex["y"].device; n = ex["y"].numel(); E = model.embed_tokens.weight.detach()
    t0 = ex["ctx"].size(1) + ex["fill"].size(1) - 1; t1 = t0 + L
    init = task.random_trigger(n, L, torch.Generator().manual_seed(seed)).to(dev)
    out = []
    for c in torch.arange(n, device=dev).split(fit(a.grad_chunk, ex, L)):
        trig = E[init[c]].clone().requires_grad_(True)
        opt = torch.optim.Adam([trig], lr=0.1 * float(E.std()))
        pre = E[torch.cat([ex["ctx"][c], ex["fill"][c]], 1)]; suf = E[ex["qry"][c]]
        best = torch.full((c.numel(),), float("inf"), device=dev); best_r = None
        for _ in range(a.soft_steps):
            _, logr = scores(model, task.val_dev, ex["y"][c], t0, t1, emb=torch.cat([pre, trig, suf], 1), want_norm=True)
            s = logr.mean(1)
            with torch.no_grad():
                up = s < best; best = torch.where(up, s, best)
                best_r = logr.detach().clone() if best_r is None else torch.where(up[:, None], logr.detach(), best_r)
            opt.zero_grad(); s.sum().backward(); opt.step()
        out.append(best_r)
    return torch.cat(out)


@torch.no_grad()
def zero_shot_token(model, task, ex, a, seed: int, n_probe: int = 1024) -> int:
    """Token with the largest mean step size when repeated after the context (no optimisation)."""
    dev = ex["y"].device; g = torch.Generator().manual_seed(seed)
    cand = task.allowed_ids[torch.randperm(task.allowed_ids.numel(), generator=g)[:n_probe]].to(dev)
    t0 = ex["ctx"].size(1) + ex["fill"].size(1) - 1; zero = torch.zeros(1, dtype=torch.long, device=dev)
    best = []
    for c in cand.split(a.eval_chunk):
        set_keep(model, True)
        model.encode(cat_ids(ex, c[:, None].repeat(1, 4), zero.expand(c.numel())))
        best.append(torch.stack([b.ssm._last_delta[:, t0 + 1: t0 + 5].mean(dim=(1, 2)) for b in model.blocks]).mean(0))
        set_keep(model, False)
    return int(cand[torch.cat(best).argmax()])


def hotflip(model, task, ex, rows, n_steps: int):
    """First-order token substitution on the trigger slot (adversarial training)."""
    E = model.embed_tokens.weight; banned = ~task.allowed.to(E.device)
    trig = ex["trig"][rows]
    pre = torch.cat([ex["ctx"][rows], ex["fill"][rows]], 1); tgt = task.val_dev[ex["y"][rows]]
    for _ in range(n_steps):
        emb_t = E[trig].detach().clone().requires_grad_(True)
        h = model.encode_from_embeddings(torch.cat([E[pre].detach(), emb_t, E[ex["qry"][rows]].detach()], 1))
        g, = torch.autograd.grad(F.cross_entropy(model.lm_head(h[:, -1]), tgt), emb_t)
        sc = g @ E.detach().T; sc[..., banned] = float("-inf")
        trig = sc.argmax(-1)
    return trig


# --------------------------------------------------------------------------- train / evaluate
@torch.no_grad()
def accuracy(model, task, ex, chunk: int) -> float:
    m, _ = eval_trigger(model, task, ex, ex["trig"], chunk)
    return float((m > 0).float().mean())


@torch.no_grad()
def perplexity(model, stream: torch.Tensor, dev, n_chunks: int, T: int = 256, bs: int = 8) -> float:
    n = min(n_chunks, stream.numel() // T)
    ids = stream[: n * T].view(n, T)
    nll, cnt = 0.0, 0
    for c in ids.split(bs):
        c = c.to(dev)
        lg = model.lm_head(model.encode(c))[:, :-1]
        nll += float(F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), c[:, 1:].reshape(-1), reduction="sum")); cnt += c[:, 1:].numel()
    return math.exp(nll / cnt)


def train(model, task, a, at: bool, dev, log: list):
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    g = torch.Generator().manual_seed(a.seed)                  # identical data stream for every arm
    t_start = time.time(); run = dict(loss=0.0, acc=0.0, lm=0.0, k=0)
    for step in range(a.steps):
        model.train(); opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            L = (0, 4, 8, 16)[int(torch.randint(0, 4, (1,), generator=g))]
            ex = to_dev(task.sample(a.bs, g, int(torch.randint(8, 33, (1,), generator=g)), L), dev)
            if at and L > 0 and step >= a.steps // 4:          # clean warm-up, then adversarial half-batches
                half = torch.arange(a.bs // 2, device=dev)
                ex["trig"][half] = hotflip(model, task, ex, half, a.at_steps)
            lg = model.lm_head(model.encode(cat_ids(ex))[:, -1])
            loss = F.cross_entropy(lg, task.val_dev[ex["y"]])
            (loss / a.accum).backward()
            run["loss"] += loss.item(); run["acc"] += float((margin(lg[:, task.val_dev].detach(), ex["y"]) > 0).float().mean()); run["k"] += 1
        if a.lm_weight > 0:                                    # keeps the model a language model (perplexity is reported)
            ids = task.lm_batch(a.lm_bs, 128, g).to(dev)
            lg = model.lm_head(model.encode(ids))[:, :-1]
            lm = F.cross_entropy(lg.reshape(-1, lg.size(-1)), ids[:, 1:].reshape(-1))
            (a.lm_weight * lm).backward(); run["lm"] += lm.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % a.log_every == 0 or step + 1 == a.steps:
            el = time.time() - t_start
            rec = dict(step=step + 1, loss=run["loss"] / run["k"], acc=run["acc"] / run["k"],
                       lm_loss=run["lm"] / max(1, run["k"] // a.accum), seconds=el)
            log.append(rec); run = dict(loss=0.0, acc=0.0, lm=0.0, k=0)
            print(f"  step {rec['step']:>5}/{a.steps}  loss {rec['loss']:.3f}  acc {rec['acc']:.2f}  lm {rec['lm_loss']:.3f}  "
                  f"{el / (step + 1):.2f}s/step  eta {el / (step + 1) * (a.steps - step - 1) / 60:.0f} min", flush=True)
            if not math.isfinite(rec["loss"]):
                raise RuntimeError("training diverged (non-finite loss)")


def clean_metrics(model, task, val, a, dev) -> dict:
    g = torch.Generator().manual_seed(10_000 + a.seed)
    exc = to_dev(task.sample(a.n_clean, g, a.eval_fill, 8, p_update=0.0), dev)
    exu = to_dev(task.sample(a.n_clean, g, a.eval_fill, 8, p_update=1.0), dev)
    return dict(clean_acc=accuracy(model, task, exc, a.eval_chunk), update_acc=accuracy(model, task, exu, a.eval_chunk),
                ppl=perplexity(model, val, dev, a.ppl_chunks))


def evaluate(model, task, val, a, dev, res: dict, save, decode):
    model.eval(); refresh_spectral_norms(model)
    if "final" not in res:
        res["final"] = clean_metrics(model, task, val, a, dev); save()
        print("  final", res["final"], flush=True)
    if a.skip_attacks:
        return
    res.setdefault("attacks", {})
    f = lambda b: float(b.float().mean())
    stat = lambda logr: dict(mean=logr.exp().mean(0).tolist(), min=logr.exp().amin(0).tolist())
    for L in a.lengths:
        if str(L) in res["attacks"]:
            continue
        t = time.time(); seed = 20_000 + 100 * a.seed + L
        ex = to_dev(task.sample(a.n_eval, torch.Generator().manual_seed(seed), a.eval_fill, L, p_update=0.0), dev)
        m_clean, r_clean = eval_trigger(model, task, ex, ex["trig"], a.eval_chunk, want_norm=True)
        clean_ok = m_clean > 0
        g = torch.Generator().manual_seed(seed + 1); ok_rand = torch.ones_like(clean_ok)
        for _ in range(8):
            ok_rand &= eval_trigger(model, task, ex, task.random_trigger(a.n_eval, L, g).to(dev), a.eval_chunk)[0] > 0
        ztok = zero_shot_token(model, task, ex, a, seed + 2)
        ok_z = eval_trigger(model, task, ex, torch.full((a.n_eval, L), ztok, device=dev), a.eval_chunk)[0] > 0
        trig_m = gcg(model, task, ex, L, "margin", a, seed + 3)
        ok_m = eval_trigger(model, task, ex, trig_m, a.eval_chunk)[0] > 0
        trig_n = gcg(model, task, ex, L, "norm", a, seed + 4)
        m_n, r_tok = eval_trigger(model, task, ex, trig_n, a.eval_chunk, want_norm=True); ok_n = m_n > 0
        r_soft = soft_norm_trigger(model, task, ex, L, a, seed + 5)
        worst = ok_rand & ok_z & ok_m & ok_n
        res["attacks"][str(L)] = dict(
            acc_benign=f(clean_ok), acc_random8=f(ok_rand), acc_zero_shot=f(ok_z), acc_gcg_margin=f(ok_m), acc_gcg_norm=f(ok_n),
            pacc_worst=f(worst), asr=float((clean_ok & ~worst).float().sum() / clean_ok.float().sum().clamp_min(1)),
            n=a.n_eval, retention_benign=stat(r_clean), retention_gcg_norm=stat(r_tok), retention_soft=stat(r_soft),
            zero_shot_token=decode([ztok]), example_triggers=[decode(tr.tolist()) for tr in trig_m[:3]], seconds=time.time() - t)
        save()
        r = res["attacks"][str(L)]
        print(f"  L={L:<2} benign {r['acc_benign']:.2f} rand {r['acc_random8']:.2f} zero {r['acc_zero_shot']:.2f} gcg-margin {r['acc_gcg_margin']:.2f} "
              f"gcg-norm {r['acc_gcg_norm']:.2f} | PACC {r['pacc_worst']:.2f} ASR {r['asr']:.2f} | retention(last block, min) "
              f"benign {r['retention_benign']['min'][-1]:.3f} soft {r['retention_soft']['min'][-1]:.3f}  [{r['seconds'] / 60:.1f} min]", flush=True)


def write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


def cmd_run(a):
    out = Path(a.out)
    if out.exists() and not a.force:
        print(f"[skip] {out} exists"); return
    out.parent.mkdir(parents=True, exist_ok=True)
    partial, ckpt = out.with_suffix(".partial.json"), out.with_suffix(".ckpt.pt")
    released = a.model == "mamba_released"
    at = a.model.endswith("_at"); calibrated = a.model.startswith("lipcal")
    arch = "mamba" if released else a.model.removesuffix("_at").replace("lipcal", "lip")
    if released:
        a.steps = 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed); t_all = time.time()
    print(f"== {a.model} seed {a.seed} on {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}", flush=True)
    sd, hf, task, val, decode = load_data(a)
    task.val_dev = task.val_ids.to(dev)

    base = build_from_mamba(sd, "mamba").to(dev).eval()
    g = torch.Generator().manual_seed(999)
    probe = [cat_ids(task.sample(8, g, 24, 8)).to(dev) for _ in range(4)] + [task.lm_batch(8, 128, g).to(dev) for _ in range(4)]
    # The converted model must compute the public checkpoint's function.  Decided in float64 against an independent
    # implementation (see conversion_error); the float32 comparison with the transformers model is a sanity bound.
    verify = conversion_error(base, sd, probe[0][:2, :64])
    with torch.no_grad():
        if hf is not None:
            ours = base.lm_head(base.encode(probe[0])); ref = hf.to(dev)(input_ids=probe[0]).logits
            verify32 = float((ours - ref).abs().max() / ref.abs().max())
            agree = float((ours.argmax(-1) == ref.argmax(-1)).float().mean())
        else:
            verify32, agree = verify, 1.0
    print(f"  conversion check: float64 max relative logit difference {verify:.1e}; "
          f"float32 vs transformers {verify32:.1e}, next-token agreement {agree:.4f}", flush=True)
    if (verify > 1e-6 or verify32 > 3e-2 or agree < 0.95) and not a.skip_verify:
        raise RuntimeError("converted model does not reproduce the public checkpoint")
    del hf
    overrides = {k: float(v) for k, v in (s.split("=") for s in a.set)}
    cal = None
    if arch != "mamba":
        if calibrated:
            cal = calibrate_constraints(base, probe, q_x=a.x_max_quantile, q_delta=a.delta_quantile, delta_headroom=a.delta_headroom)
        else:                                                  # paper constants; only the input radius is calibrated
            cal = dict(x_max=[a.x_max] * len(base.blocks) if a.x_max is not None else calibrate_x_max(base, probe, q=a.x_max_quantile))
        if not ARCHS[arch][1].get("clip_input", True):
            cal.pop("x_max")
    model = base if arch == "mamba" and not overrides else build_from_mamba(sd, arch, calibration=cal, **overrides).to(dev)
    del base, sd

    if partial.exists() and ckpt.exists():                     # resume after training
        res = json.loads(partial.read_text()); state = torch.load(ckpt, map_location=dev)
        model.load_state_dict(state["model"])
        if state["cal"] is not None:
            apply_calibration(model, state["cal"])
        print("  resumed from checkpoint; continuing evaluation", flush=True)
    else:
        res = dict(model=a.model, arch=arch, adversarial_training=at, calibrated_constants=calibrated, seed=a.seed, tiny=a.tiny, base=a.base,
                   args={k: v for k, v in vars(a).items() if k not in ("fn",)}, conversion_rel_error=verify, conversion_rel_error_float32_vs_transformers=verify32, conversion_next_token_agreement=agree,
                   torch=torch.__version__, device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                   n_names=len(task.names), n_values=len(task.values), allowed_trigger_tokens=int(task.allowed.sum()))
        model.eval(); refresh_spectral_norms(model)
        res["init"] = clean_metrics(model, task, val, a, dev); print("  init", res["init"], flush=True)
        res["train_log"] = []
        if a.steps > 0:
            train(model, task, a, at, dev, res["train_log"])
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        res["config"] = describe(model); res["train_seconds"] = time.time() - t_all
        if a.steps > 0 and not a.skip_attacks:
            torch.save(dict(model=model.state_dict(), cal=cal), ckpt)
        write_json(partial, res)
    evaluate(model, task, val, a, dev, res, lambda: write_json(partial, res), decode)
    res["seconds"] = res.get("seconds", 0.0) + time.time() - t_all
    if dev.type == "cuda":
        res["max_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    write_json(out, res); partial.unlink(missing_ok=True)
    if a.keep_ckpt and ckpt.exists():
        print(f"  checkpoint kept at {ckpt}")
    else:
        ckpt.unlink(missing_ok=True)
    print(f"== done {a.model} s{a.seed} in {res['seconds'] / 60:.1f} min -> {out}", flush=True)


def cmd_diagnose(a):
    """Before any training: how far is each constrained variant from the public checkpoint?"""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd, _hf, task, val, _ = load_data(a); task.val_dev = task.val_ids.to(dev)
    a.n_clean, a.ppl_chunks = 128, 16
    base = build_from_mamba(sd, "mamba").to(dev).eval()
    g = torch.Generator().manual_seed(999)
    probe = [cat_ids(task.sample(8, g, 24, 8)).to(dev) for _ in range(4)] + [task.lm_batch(8, 128, g).to(dev) for _ in range(4)]
    cal = calibrate_constraints(base, probe, q_x=a.x_max_quantile, q_delta=a.delta_quantile, delta_headroom=a.delta_headroom)
    paper = dict(x_max=calibrate_x_max(base, probe, q=a.x_max_quantile))
    rng = lambda v: f"{min(v):.3g}..{max(v):.3g}"
    print("calibrated constants over blocks: " + ", ".join(f"{k} {rng(v)}" for k, v in cal.items()), flush=True)
    rows = [("mamba (public checkpoint)", "mamba", None), ("lipcal (all constraints, calibrated)", "lip", cal),
            ("lipcal without spectral norm", "lip_no_sn", cal), ("lipcal without step clamp", "lip_no_clamp", cal),
            ("lipcal without eigenvalue bound", "lip_free_A", cal), ("lipcal without input projection", "lip_no_clip", cal),
            ("lipcal, envelope = clean maximum", "lip", calibrate_constraints(base, probe, q_x=1.0, q_delta=1.0, delta_headroom=a.delta_headroom)),
            ("lip (paper constants)", "lip", paper)]
    out = {}
    for label, arch, c in rows:
        c = None if c is None else {k: v for k, v in c.items() if k != "x_max" or ARCHS[arch][1].get("clip_input", True)}
        m = base if arch == "mamba" else build_from_mamba(sd, arch, calibration=c).to(dev)
        m.eval(); refresh_spectral_norms(m)
        out[label] = clean_metrics(m, task, val, a, dev)
        print(f"  {label:<40} ppl {out[label]['ppl']:<12.4g} clean acc {out[label]['clean_acc']:.3f}  update acc {out[label]['update_acc']:.3f}", flush=True)
        if m is not base:
            del m
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); write_json(Path(a.out), dict(constants=cal, init=out))


def cmd_all(a):
    """Run the arms one after another, or ``--jobs`` at a time with one GPU each (logs in <dir>/logs)."""
    models = (a.models or CORE) + (PAPER if a.paper_constants else []) + (ABLATIONS if a.ablations else [])
    jobs = [(m, s) for s in a.seeds for m in models if not (m in ABLATIONS and s != a.seeds[0])]
    passthrough = [x for x in a.rest if x != "--"]
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    width = max(1, min(a.jobs, n_gpu) if n_gpu else a.jobs)
    logs = Path(a.dir) / "logs"; logs.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); failed, skipped, running = [], [], {}     # running: slot -> (name, process, log file)
    todo = [(m, s) for m, s in jobs if not (Path(a.dir) / f"{m}_s{s}.json").exists()]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} runs already finished; {len(todo)} to do, {width} at a time", flush=True)
    last_report = time.time()

    def reap():
        for slot, (name, proc, fh) in list(running.items()):
            if proc.poll() is None:
                continue
            if fh is not None:
                fh.close()
            del running[slot]
            if proc.returncode != 0:
                failed.append(name)
                print(f"!! {name} failed (exit {proc.returncode})", flush=True)
                if fh is not None:
                    print("   last lines of its log:\n   " + "\n   ".join((logs / f"{name}.log").read_text().splitlines()[-12:]), flush=True)
            else:
                print(f"== finished {name}  [{(time.time() - t0) / 3600:.2f} h elapsed]", flush=True)

    while todo or running:
        reap()
        while todo and len(running) < width:
            m, s = todo.pop(0); name = f"{m}_s{s}"
            if a.time_budget_hours and time.time() - t0 > a.time_budget_hours * 3600:
                skipped.append(name); continue
            slot = next(i for i in range(width) if i not in running)
            cmd = [sys.executable, __file__, "run", "--model", m, "--seed", str(s), "--out", str(Path(a.dir) / f"{name}.json")] + passthrough
            env = dict(os.environ)
            if n_gpu > 1:
                env["CUDA_VISIBLE_DEVICES"] = str(slot % n_gpu)
            fh = open(logs / f"{name}.log", "w") if width > 1 else None
            print(f"#### starting {name}" + (f" on GPU {slot % n_gpu}" if n_gpu > 1 else ""), flush=True)
            running[slot] = (name, subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT if fh else None), fh)
        if running:
            time.sleep(5)
            if width > 1 and time.time() - last_report > 300:
                last_report = time.time()
                for name, _, _ in running.values():
                    lines = (logs / f"{name}.log").read_text().splitlines()
                    print(f"   [{name}] {lines[-1].strip() if lines else '(starting)'}", flush=True)
    cmd_aggregate(a)
    if skipped:
        print(f"\n---- time budget reached; not started: {', '.join(skipped)}. Run the same command again to do them.", flush=True)
    if failed:
        print(f"\n!!!! {len(failed)} of {len(jobs)} runs FAILED: {', '.join(failed)}\n!!!! Their results are missing above. "
              "Look for the first 'Traceback' (in <dir>/logs when running in parallel); running the same command again retries them.", flush=True)


def cmd_aggregate(a):
    import statistics as st
    rows: dict[str, list] = {}
    for f in sorted(Path(a.dir).glob("*_s*.json")):
        if f.name.endswith(".partial.json"):
            continue
        r = json.loads(f.read_text()); rows.setdefault(r["model"], []).append(r)
    if not rows:
        print("no finished runs yet"); return
    ms = lambda xs: [st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0]
    summary = {}
    for m, rs in rows.items():
        o = dict(n_seeds=len(rs), seeds=[r["seed"] for r in rs], steps=rs[0]["args"]["steps"],
                 **{k: ms([r["final"][k] for r in rs]) for k in ("clean_acc", "update_acc", "ppl")},
                 init_ppl=ms([r["init"]["ppl"] for r in rs]), init_clean_acc=ms([r["init"]["clean_acc"] for r in rs]))
        for L in sorted({L for r in rs for L in r.get("attacks", {})}, key=int):
            have = [r["attacks"][L] for r in rs if L in r.get("attacks", {})]
            for k in ("acc_benign", "acc_random8", "acc_zero_shot", "acc_gcg_margin", "acc_gcg_norm", "pacc_worst", "asr"):
                o[f"{k}@{L}"] = ms([h[k] for h in have])
            for k in ("retention_benign", "retention_gcg_norm", "retention_soft"):
                o[f"{k}_last_min@{L}"] = ms([h[k]["min"][-1] for h in have])
                o[f"{k}_allblocks_min@{L}"] = ms([min(h[k]["min"]) for h in have])
                o[f"{k}_allblocks_mean@{L}"] = ms([st.mean(h[k]["mean"]) for h in have])
        summary[m] = o
    write_json(Path(a.dir) / "summary.json", summary)
    Ls = sorted({k.split("@")[1] for o in summary.values() for k in o if k.startswith("pacc_worst@")}, key=int)
    pc = lambda v: f"{100 * v[0]:.1f} ± {100 * v[1]:.1f}"
    head = ["model", "seeds", "clean", "update", "ppl"] + [f"PACC@{L}" for L in Ls] + [f"ASR@{L}" for L in Ls] + [f"ret.soft@{L}" for L in Ls]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in [x for x in CORE + PAPER + ABLATIONS if x in summary] + [x for x in summary if x not in CORE + PAPER + ABLATIONS]:
        o = summary[m]
        cells = [m, str(o["n_seeds"]), pc(o["clean_acc"]), pc(o["update_acc"]), f"{o['ppl'][0]:.1f}"]
        cells += [pc(o[f"pacc_worst@{L}"]) if f"pacc_worst@{L}" in o else "-" for L in Ls]
        cells += [pc(o[f"asr@{L}"]) if f"asr@{L}" in o else "-" for L in Ls]
        cells += [f"{o[f'retention_soft_last_min@{L}'][0]:.3f}" if f"retention_soft_last_min@{L}" in o else "-" for L in Ls]
        lines.append("| " + " | ".join(cells) + " |")
    (Path(a.dir) / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=CORE + PAPER + ABLATIONS + [m.replace("lipcal", "lip") for m in ABLATIONS])
    r.add_argument("--seed", type=int, default=0); r.add_argument("--out", required=True)
    r.add_argument("--preset", choices=list(PRESETS), default=None); r.add_argument("--tiny", action="store_true")
    r.add_argument("--base", default="state-spaces/mamba-130m-hf"); r.add_argument("--revision", default=None)
    r.add_argument("--cache", default="data_cache")
    for k in ("bs", "accum", "grad_chunk", "eval_chunk", "steps", "n_eval", "iters", "n_cand"):
        r.add_argument("--" + k.replace("_", "-"), type=int, default=None)
    r.add_argument("--lr", type=float, default=1e-4); r.add_argument("--lm-weight", type=float, default=0.5); r.add_argument("--lm-bs", type=int, default=4)
    r.add_argument("--at-steps", type=int, default=2, help="HotFlip refinement steps per adversarial batch")
    r.add_argument("--lengths", type=int, nargs="+", default=[8, 16, 32]); r.add_argument("--eval-fill", type=int, default=24)
    r.add_argument("--n-clean", type=int, default=256); r.add_argument("--ppl-chunks", type=int, default=64)
    r.add_argument("--topk", type=int, default=64); r.add_argument("--soft-steps", type=int, default=20)
    r.add_argument("--x-max", type=float, default=None, help="input-projection radius; default: calibrated per block on clean data")
    r.add_argument("--x-max-quantile", type=float, default=0.999); r.add_argument("--delta-quantile", type=float, default=0.999)
    r.add_argument("--delta-headroom", type=float, default=2.0, help="calibrated arms: delta_max = headroom x clean quantile")
    r.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override constraint constants, e.g. delta_max=1.0 lambda_max=4")
    r.add_argument("--n-fact", type=int, default=4, help="facts per context"); r.add_argument("--n-values", type=int, default=16, help="number of code words (answer classes)")
    r.add_argument("--log-every", type=int, default=25)
    r.add_argument("--skip-attacks", action="store_true"); r.add_argument("--skip-verify", action="store_true")
    r.add_argument("--keep-ckpt", action="store_true"); r.add_argument("--force", action="store_true")
    al = sub.add_parser("all", help="run every arm sequentially; extra flags after -- are passed to 'run'")
    al.add_argument("--dir", default="runs/scale"); al.add_argument("--seeds", type=int, nargs="+", default=[0])
    al.add_argument("--models", nargs="+", default=None); al.add_argument("--ablations", action="store_true"); al.add_argument("--paper-constants", action="store_true", help="also run lip and lip_at")
    al.add_argument("--jobs", type=int, default=1, help="runs in parallel, one GPU each (e.g. 2 on Kaggle's T4 x2)")
    al.add_argument("--time-budget-hours", type=float, default=0.0, help="do not start new runs after this many hours (0 = no limit)")
    al.add_argument("rest", nargs=argparse.REMAINDER)
    ag = sub.add_parser("aggregate"); ag.add_argument("--dir", default="runs/scale")
    dg = sub.add_parser("diagnose", help="no training: initial perplexity and accuracy of every constrained variant")
    dg.add_argument("--out", default=None); dg.add_argument("--tiny", action="store_true"); dg.add_argument("--seed", type=int, default=0)
    dg.add_argument("--base", default="state-spaces/mamba-130m-hf"); dg.add_argument("--revision", default=None); dg.add_argument("--cache", default="data_cache")
    dg.add_argument("--x-max-quantile", type=float, default=0.999); dg.add_argument("--delta-quantile", type=float, default=0.999)
    dg.add_argument("--delta-headroom", type=float, default=2.0); dg.add_argument("--eval-chunk", type=int, default=128)
    dg.add_argument("--eval-fill", type=int, default=24); dg.add_argument("--n-fact", type=int, default=4); dg.add_argument("--n-values", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "run":
        smoke = a.tiny and a.steps is None                    # pure smoke test: shrink everything
        if a.tiny and a.preset is None:
            a.preset = "tiny"
        for k, v in PRESETS[a.preset or "a100"].items():
            if getattr(a, k) is None:
                setattr(a, k, v)
        if smoke:
            a.n_clean, a.ppl_chunks, a.soft_steps, a.lengths = 16, 4, 3, a.lengths[:2]
    {"run": cmd_run, "all": cmd_all, "aggregate": cmd_aggregate, "diagnose": cmd_diagnose}[a.cmd](a)


if __name__ == "__main__":
    main()
