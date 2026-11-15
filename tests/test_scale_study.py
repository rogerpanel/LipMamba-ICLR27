"""Guards for the attack code of scripts/scale_study.py."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scale_study as S  # noqa: E402

from lipmamba.pretrained import build_from_mamba  # noqa: E402


def _setup(n=8, L=4):
    a = argparse.Namespace(tiny=True, n_fact=4, n_values=16, n_cand=8, topk=16, iters=4, eval_chunk=32, grad_chunk=4, soft_steps=3)
    sd, _, task, _, _ = S.load_data(a)
    task.val_dev = task.val_ids
    model = build_from_mamba(sd, "mamba").eval()
    ex = task.sample(n, torch.Generator().manual_seed(0), 12, L, p_update=0.0)
    return a, task, model, ex


def test_candidate_evaluation_builds_no_graph():
    """Candidate scoring must run without autograd; otherwise every chunk's graph is kept and memory explodes."""
    _, task, model, ex = _setup()
    m, r = S.eval_trigger(model, task, ex, ex["trig"], 4, want_norm=True)
    assert not m.requires_grad and not r.requires_grad
    assert all(b.ssm._state_seq is None for b in model.blocks)


def test_search_lowers_its_objective_and_uses_allowed_tokens():
    a, task, model, ex = _setup()
    with torch.no_grad():                         # label = current prediction, so the margin starts positive
        ex["y"] = model.lm_head(model.encode(S.cat_ids(ex))[:, -1])[:, task.val_ids].argmax(1)
    g = torch.Generator().manual_seed(1)
    rand = task.random_trigger(8, 4, g)
    m0, r0 = S.eval_trigger(model, task, ex, rand, 8, want_norm=True)
    tm = S.gcg(model, task, ex, 4, "margin", a, 5)
    tn = S.gcg(model, task, ex, 4, "norm", a, 6)
    m1, _ = S.eval_trigger(model, task, ex, tm, 8)
    _, r1 = S.eval_trigger(model, task, ex, tn, 8, want_norm=True)
    assert task.allowed[tm].all() and task.allowed[tn].all()
    assert m1.mean() < m0.mean() and r1.mean() < r0.mean()
