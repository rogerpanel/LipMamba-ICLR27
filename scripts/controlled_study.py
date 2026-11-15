#!/usr/bin/env python
"""Controlled study answering the ICLR reviewers' control questions on CPU.

A small selective-SSM is trained FROM SCRATCH on a synthetic long-context
retrieval task (key -> true/false facts, then filler, then a query), in a
2 x 2 design {unconstrained Mamba, LipMamba constraints} x {clean training,
adversarial training}, plus three single-constraint ablations.  Every model
gets the SAME data, optimiser, steps and seeds (matched training), and every
attack is RE-OPTIMISED white-box against each trained checkpoint.

It is NOT a reproduction of the paper's 130M/370M tables: it isolates the
mechanism at small scale so that the following can be measured exactly:
  * matched-training control: what do the constraints add, with and without AT;
  * adaptive attacks: GCG-style token search on the classification margin
    (end-to-end through the clamp and the readout) and on the state norm;
  * empirical state retention versus the Theorem-4 bound;
  * utility: accuracy on fact UPDATES (the model must revise a stored value).

  python scripts/controlled_study.py run --model lip --seed 0 --out runs/controlled/lip_s0.json
  python scripts/controlled_study.py all --seeds 0 1 2 --jobs 4
  python scripts/controlled_study.py aggregate --dir runs/controlled
"""
from __future__ import annotations

import argparse, json, math, subprocess, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

from lipmamba import LipMambaModel
from lipmamba.baselines import preset, ablate

N_KEY, N_PAIR, N_FILL = 8, 4, 32
KEYS = list(range(N_KEY)); VALS = [N_KEY, N_KEY + 1]
FILL = list(range(N_KEY + 2, N_KEY + 2 + N_FILL)); SEP = FILL[-1] + 1; QRY = SEP + 1
N_RARE = 32
RARE = list(range(QRY + 1, QRY + 1 + N_RARE))      # tokens that never occur in training (attacker may use them)
ATTACK = FILL + RARE
VOCAB = RARE[-1] + 1
ARCH = dict(vocab_size=VOCAB, n_layers=2, d_model=64, d_inner=128, state_dim=8, conv_kernel=4, n_classes=2,
            track_lipschitz=False)
MODELS = {
    "mamba":        lambda: preset("unconstrained_mamba", **ARCH),
    "mamba_at":     lambda: preset("unconstrained_mamba", **ARCH),
    "lip":          lambda: preset("lipmamba_arch", **ARCH),
    "lip_at":       lambda: preset("lipmamba_arch", **ARCH),
    "lip_no_sn":    lambda: ablate("no_sn", **ARCH),
    "lip_no_clamp": lambda: ablate("no_clamp", **ARCH),
    "lip_free_A":   lambda: ablate("free_A", **ARCH),
}
LENGTHS = (2, 4, 8, 16)
MAX_FILL = 16


def make_context(n: int, g: torch.Generator, update: bool):
    """Facts 'k v' x4 (+ one re-assignment if update). Returns context ids, query key, label."""
    keys = torch.stack([torch.randperm(N_KEY, generator=g)[:N_PAIR] for _ in range(n)])
    vals = torch.randint(0, 2, (n, N_PAIR), generator=g)
    ctx = torch.stack([keys, vals + N_KEY], dim=-1).flatten(1)                 # (n, 12)
    qi = torch.randint(0, N_PAIR, (n,), generator=g)
    label = vals.gather(1, qi[:, None]).squeeze(1)
    qkey = keys.gather(1, qi[:, None]).squeeze(1)
    if update:                                 # re-assign the queried key at a random LATER slot (not always last)
        label = 1 - label
        rows = []
        for i in range(n):
            pairs = [ctx[i, 2 * j:2 * j + 2] for j in range(N_PAIR)]
            slot = int(torch.randint(int(qi[i]) + 1, N_PAIR + 1, (1,), generator=g))
            pairs.insert(slot, torch.stack([qkey[i], label[i] + N_KEY]))
            rows.append(torch.cat(pairs))
        ctx = torch.stack(rows)
    else:                                      # pad with a filler pair at a random slot, same length
        rows = []
        for i in range(n):
            pairs = [ctx[i, 2 * j:2 * j + 2] for j in range(N_PAIR)]
            slot = int(torch.randint(0, N_PAIR + 1, (1,), generator=g))
            pairs.insert(slot, torch.tensor(FILL)[torch.randint(0, N_FILL, (2,), generator=g)])
            rows.append(torch.cat(pairs))
        ctx = torch.stack(rows)
    return ctx, qkey, label


def assemble(ctx, mid, qkey):
    n = ctx.size(0)
    return torch.cat([ctx, torch.full((n, 1), SEP), mid, torch.full((n, 1), QRY), qkey[:, None]], dim=1)


def random_fill(n, L, g):
    return torch.tensor(FILL)[torch.randint(0, N_FILL, (n, L), generator=g)]


def random_attack(n, L, g):
    return torch.tensor(ATTACK)[torch.randint(0, len(ATTACK), (n, L), generator=g)]


def logits_with_mid_onehot(model, ctx, onehot, qkey):
    """Differentiable logits when the middle segment is given as one-hot rows over the vocabulary."""
    E = model.embed_tokens.weight
    n = ctx.size(0)
    pre = E[torch.cat([ctx, torch.full((n, 1), SEP)], 1)]
    suf = E[torch.cat([torch.full((n, 1), QRY), qkey[:, None]], 1)]
    return model.logits_from_embeddings(torch.cat([pre, onehot @ E, suf], 1))


def margin(logits, y):
    return logits.gather(1, y[:, None]).squeeze(1) - logits.gather(1, (1 - y)[:, None]).squeeze(1)


def hotflip(model, ctx, mid, qkey, y):
    """One-step first-order token substitution on the filler (used for adversarial training)."""
    oh = F.one_hot(mid, VOCAB).float().requires_grad_(True)
    loss = F.cross_entropy(logits_with_mid_onehot(model, ctx, oh, qkey), y)
    g, = torch.autograd.grad(loss, oh)
    mask = torch.full((VOCAB,), float("-inf")); mask[ATTACK] = 0.0
    return (g + mask).argmax(-1).detach()


def train(name: str, seed: int, steps: int, bs: int = 64):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    model = LipMambaModel(MODELS[name]())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    at = name.endswith("_at")
    for step in range(steps):
        model.train()
        L = int(torch.randint(0, MAX_FILL + 1, (1,), generator=g))
        upd = bool(torch.rand(1, generator=g) < 0.3)
        ctx, qkey, y = make_context(bs, g, upd)
        mid = random_fill(bs, L, g)
        if at and L > 0 and step >= steps // 2:            # clean warm-up, then adversarial half-batches (same recipe for every model)
            h = bs // 2
            mid = mid.clone(); mid[:h] = hotflip(model, ctx[:h], mid[:h], qkey[:h], y[:h])
        loss = F.cross_entropy(model(assemble(ctx, mid, qkey))["cls_logits"], y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return model.eval()


@torch.no_grad()
def predict(model, ctx, mid, qkey):
    return model(assemble(ctx, mid, qkey))["cls_logits"]


@torch.no_grad()
def state_stats(model, ctx, mid, qkey):
    """Retention ratio ||h_{t0+l}|| / ||h_{t0}|| in every block, t0 = SEP position, and mean step size."""
    for b in model.blocks: b.ssm.keep_state = True
    model(assemble(ctx, mid, qkey))
    t0 = ctx.size(1); L = mid.size(1)
    out = []
    for b in model.blocks:
        s = b.ssm._state_seq                                # (n, T)
        out.append((s[:, t0 + L] / s[:, t0].clamp_min(1e-12), b.ssm._last_delta[:, t0 + 1:t0 + 1 + L].mean(dim=(1, 2)) if L else torch.zeros(ctx.size(0))))
    for b in model.blocks: b.ssm.keep_state = False
    return out


def gcg(model, ctx, qkey, y, L, objective="margin", iters=30, topk=8, restarts=1, seed=0):
    """Gradient-guided token search, re-optimised against THIS checkpoint, per example.
    objective='margin': minimise the classification margin (end-to-end, through clamp and readout).
    objective='norm'  : minimise the last block's post-trigger state norm (HiSPA's own objective)."""
    g = torch.Generator().manual_seed(1000 + seed); n = ctx.size(0)
    mask = torch.full((VOCAB,), float("-inf")); mask[ATTACK] = 0.0
    last = model.blocks[-1].ssm

    def score(onehot_or_ids, need_grad):
        oh = onehot_or_ids if need_grad else F.one_hot(onehot_or_ids, VOCAB).float()
        last.keep_state = objective == "norm"
        lg = logits_with_mid_onehot(model, ctx, oh, qkey)
        if objective == "margin":
            return margin(lg, y)
        s = last._state_seq[:, ctx.size(1) + L]; last.keep_state = False
        return s

    best_ids = random_attack(n, L, g)
    with torch.no_grad(): best = score(best_ids, False)
    for r in range(restarts):
        ids = random_attack(n, L, g)
        with torch.no_grad(): cur = score(ids, False)
        for _ in range(iters):
            oh = F.one_hot(ids, VOCAB).float().requires_grad_(True)
            grad, = torch.autograd.grad(score(oh, True).sum(), oh)
            pos = torch.randint(0, L, (n,), generator=g)
            cand = (-grad[torch.arange(n), pos] + mask).topk(topk, dim=-1).indices      # tokens that lower the score
            with torch.no_grad():
                for c in range(topk):
                    trial = ids.clone(); trial[torch.arange(n), pos] = cand[:, c]
                    s = score(trial, False); better = s < cur
                    ids = torch.where(better[:, None], trial, ids); cur = torch.where(better, s, cur)
        upd = cur < best
        best_ids = torch.where(upd[:, None], ids, best_ids); best = torch.where(upd, cur, best)
    return best_ids, best


def soft_trigger(model, ctx, qkey, y, L, objective="margin", steps=60, lr=0.1, seed=0):
    """Continuous relaxation: every trigger position is a free embedding vector (white-box, Adam).
    Strictly stronger than any token trigger; the block's input LayerNorm removes the norm degree of freedom."""
    torch.manual_seed(2000 + seed); n = ctx.size(0); E = model.embed_tokens.weight.detach()
    pre = E[torch.cat([ctx, torch.full((n, 1), SEP)], 1)]; suf = E[torch.cat([torch.full((n, 1), QRY), qkey[:, None]], 1)]
    trig = (E[random_attack(n, L, torch.Generator().manual_seed(seed))]).clone().requires_grad_(True)
    opt = torch.optim.Adam([trig], lr=lr); last = model.blocks[-1].ssm
    best = torch.full((n,), float("inf")); best_trig = trig.detach().clone()
    for _ in range(steps):
        last.keep_state = objective == "norm"
        lg = model.logits_from_embeddings(torch.cat([pre, trig, suf], 1))
        s = margin(lg, y) if objective == "margin" else last._state_seq[:, ctx.size(1) + L]
        last.keep_state = False
        with torch.no_grad():
            upd = s < best; best = torch.where(upd, s.detach(), best); best_trig[upd] = trig.detach()[upd]
        opt.zero_grad(); s.sum().backward(); opt.step()
    with torch.no_grad():
        lg = model.logits_from_embeddings(torch.cat([pre, best_trig, suf], 1))
    return best_trig, margin(lg, y)


@torch.no_grad()
def state_stats_emb(model, ctx, trig, qkey):
    n = ctx.size(0); E = model.embed_tokens.weight
    pre = E[torch.cat([ctx, torch.full((n, 1), SEP)], 1)]; suf = E[torch.cat([torch.full((n, 1), QRY), qkey[:, None]], 1)]
    for b in model.blocks: b.ssm.keep_state = True
    model.logits_from_embeddings(torch.cat([pre, trig, suf], 1)); t0 = ctx.size(1); L = trig.size(1)
    out = [b.ssm._state_seq[:, t0 + L] / b.ssm._state_seq[:, t0].clamp_min(1e-12) for b in model.blocks]
    dl = [b.ssm._last_delta[:, t0 + 1:t0 + 1 + L].mean(dim=(1, 2)) for b in model.blocks]
    for b in model.blocks: b.ssm.keep_state = False
    return out, dl


def evaluate(model, seed: int, n: int = 256, iters: int = 30):
    g = torch.Generator().manual_seed(7 + seed)
    ctx, qkey, y = make_context(n, g, update=False)
    uctx, uq, uy = make_context(n, g, update=True)
    res = {"clean_acc": {}, "update_acc": {}, "attacks": {}}
    for L in (0, 8, 16):
        res["clean_acc"][L] = float((predict(model, ctx, random_fill(n, L, g), qkey).argmax(-1) == y).float().mean())
        res["update_acc"][L] = float((predict(model, uctx, random_fill(n, L, g), uq).argmax(-1) == uy).float().mean())
    # step-size ranking for the zero-shot (Z-HiSPA-style) trigger: token with the largest mean step in the last block
    with torch.no_grad():
        for b in model.blocks: b.ssm.keep_state = True
        probe = torch.tensor(ATTACK)[:, None]; pc, pq, _ = make_context(len(ATTACK), g, False)
        model(assemble(pc, probe, pq)); d = model.blocks[-1].ssm._last_delta[:, pc.size(1) + 1].mean(-1)
        for b in model.blocks: b.ssm.keep_state = False
    ztok = int(torch.tensor(ATTACK)[d.argmax()])
    for L in LENGTHS:
        clean_ok = predict(model, ctx, random_fill(n, L, g), qkey).argmax(-1) == y
        ok_rand = torch.ones(n, dtype=torch.bool)
        for _ in range(8):
            ok_rand &= predict(model, ctx, random_attack(n, L, g), qkey).argmax(-1) == y
        ok_z = predict(model, ctx, torch.full((n, L), ztok), qkey).argmax(-1) == y
        _, m = gcg(model, ctx, qkey, y, L, "margin", iters=iters, seed=seed); ok_m = m > 0
        mid_n, _ = gcg(model, ctx, qkey, y, L, "norm", iters=iters, seed=seed)
        ok_n = predict(model, ctx, mid_n, qkey).argmax(-1) == y
        _, sm = soft_trigger(model, ctx, qkey, y, L, "margin", seed=seed); ok_sm = sm > 0
        trig_n, sn = soft_trigger(model, ctx, qkey, y, L, "norm", seed=seed); ok_sn = sn > 0
        tok_worst = ok_rand & ok_z & ok_m & ok_n; all_worst = tok_worst & ok_sm & ok_sn
        st_tok = state_stats(model, ctx, mid_n, qkey); ret_soft, dl_soft = state_stats_emb(model, ctx, trig_n, qkey)
        f = lambda b: float(b.float().mean())
        res["attacks"][L] = {
            "acc_random8": f(ok_rand), "acc_zero_shot": f(ok_z), "acc_gcg_margin": f(ok_m), "acc_gcg_norm": f(ok_n),
            "acc_soft_margin": f(ok_sm), "acc_soft_norm": f(ok_sn),
            "pacc_token_worst": f(tok_worst), "pacc_worst": f(all_worst),
            "asr_token": float((clean_ok & ~tok_worst).float().sum() / clean_ok.float().sum().clamp_min(1)),
            "asr": float((clean_ok & ~all_worst).float().sum() / clean_ok.float().sum().clamp_min(1)),
            "retention_token_mean": [float(r.mean()) for r, _ in st_tok], "retention_token_min": [float(r.min()) for r, _ in st_tok],
            "retention_soft_mean": [float(r.mean()) for r in ret_soft], "retention_soft_min": [float(r.min()) for r in ret_soft],
            "delta_token": [float(d.mean()) for _, d in st_tok], "delta_soft": [float(d.mean()) for d in dl_soft],
        }
    cs = model.constraints
    res["theory"] = {"rho_min": cs.rho_min, "c": cs.c,
                     "constrained": bool(model.cfg.clamp_delta and model.cfg.reparam_eigen and model.cfg.spectral_norm)}
    return res


def cmd_run(a):
    torch.set_num_threads(1); t = time.time()
    model = train(a.model, a.seed, a.steps)
    res = evaluate(model, a.seed, n=a.n_eval, iters=a.iters)
    res.update(model=a.model, seed=a.seed, steps=a.steps, seconds=time.time() - t,
               params=sum(p.numel() for p in model.parameters()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
    print(a.model, a.seed, "clean", res["clean_acc"], "upd", res["update_acc"],
          "tok", {L: round(v["pacc_token_worst"], 2) for L, v in res["attacks"].items()}, "soft", {L: round(v["pacc_worst"], 2) for L, v in res["attacks"].items()}, f"{res['seconds']:.0f}s", flush=True)


def cmd_all(a):
    core = ["mamba", "lip", "mamba_at", "lip_at"]; abl = [m for m in MODELS if m not in core]
    jobs = [(m, s) for s in a.seeds for m in core] + [(m, s) for s in a.seeds[:2] for m in abl]
    running = []
    while jobs or running:
        while jobs and len(running) < a.jobs:
            m, s = jobs.pop(0); out = f"{a.dir}/{m}_s{s}.json"
            if Path(out).exists(): continue
            running.append(subprocess.Popen([sys.executable, __file__, "run", "--model", m, "--seed", str(s), "--steps", str(a.steps),
                                             "--n-eval", str(a.n_eval), "--iters", str(a.iters), "--out", out]))
        running = [p for p in running if p.poll() is None]; time.sleep(2)


def cmd_aggregate(a):
    import statistics as st
    rows = {}
    for f in sorted(Path(a.dir).glob("*_s*.json")):
        r = json.loads(f.read_text()); rows.setdefault(r["model"], []).append(r)
    ms = lambda xs: (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)
    out = {}
    for m, rs in rows.items():
        o = {"n_seeds": len(rs), "clean_acc": ms([r["clean_acc"]["8"] for r in rs]), "update_acc": ms([r["update_acc"]["8"] for r in rs])}
        for L in map(str, LENGTHS):
            for k in ("acc_random8", "acc_zero_shot", "acc_gcg_margin", "acc_gcg_norm", "acc_soft_margin", "acc_soft_norm",
                      "pacc_token_worst", "pacc_worst", "asr_token", "asr"):
                o[f"{k}@{L}"] = ms([r["attacks"][L][k] for r in rs])
            for k in ("retention_token_mean", "retention_token_min", "retention_soft_mean", "retention_soft_min", "delta_soft"):
                for blk in (0, -1):
                    o[f"{k}[{blk}]@{L}"] = ms([r["attacks"][L][k][blk] for r in rs])
        out[m] = o
    Path(a.dir, "summary.json").write_text(json.dumps(out, indent=1))
    hdr = f"{'model':<13}{'clean':>12}{'update':>12}" + "".join(f"{'tok@'+str(L):>12}" for L in LENGTHS) + "".join(f"{'soft@'+str(L):>12}" for L in LENGTHS)
    print(hdr)
    for m, o in out.items():
        f2 = lambda v: f"{100*v[0]:6.1f}±{100*v[1]:<4.1f}"
        print(f"{m:<13}{f2(o['clean_acc']):>12}{f2(o['update_acc']):>12}" + "".join(f"{f2(o[f'pacc_token_worst@{L}']):>12}" for L in LENGTHS) + "".join(f"{f2(o[f'pacc_worst@{L}']):>12}" for L in LENGTHS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--model", required=True, choices=list(MODELS)); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--steps", type=int, default=3000); r.add_argument("--n-eval", type=int, default=256); r.add_argument("--iters", type=int, default=30); r.add_argument("--out", required=True)
    al = sub.add_parser("all"); al.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); al.add_argument("--jobs", type=int, default=4)
    al.add_argument("--steps", type=int, default=3000); al.add_argument("--n-eval", type=int, default=256); al.add_argument("--iters", type=int, default=30); al.add_argument("--dir", default="runs/controlled")
    ag = sub.add_parser("aggregate"); ag.add_argument("--dir", default="runs/controlled")
    a = ap.parse_args(); {"run": cmd_run, "all": cmd_all, "aggregate": cmd_aggregate}[a.cmd](a)
