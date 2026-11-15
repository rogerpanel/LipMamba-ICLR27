"""Conversion of a public-layout Mamba checkpoint: exactness without constraints, validity with them."""
import pytest
import torch

from lipmamba.pretrained import (ARCHS, build_from_mamba, calibrate_x_max, infer_shapes, reference_logits,
                                 tiny_state_dict)

V = 97


def _ids(n=3, t=20, seed=0):
    return torch.randint(0, V, (n, t), generator=torch.Generator().manual_seed(seed))


def test_unconstrained_conversion_matches_reference():
    sd = tiny_state_dict(V)
    model = build_from_mamba(sd, "mamba").eval()
    ids = _ids()
    assert torch.allclose(model(ids)["lm_logits"], reference_logits(sd, ids), atol=1e-4, rtol=1e-4)


def test_matches_transformers_implementation():
    tr = pytest.importorskip("transformers")
    cfg = tr.MambaConfig(vocab_size=V, hidden_size=48, state_size=8, num_hidden_layers=2, expand=2, conv_kernel=4,
                         time_step_rank=4, use_bias=False, use_conv_bias=True)
    torch.manual_seed(0)
    hf = tr.MambaForCausalLM(cfg).eval()
    sd = hf.state_dict()
    assert infer_shapes(sd)["dt_rank"] == 4
    ids = _ids(seed=1)
    with torch.no_grad():
        ref = hf(input_ids=ids).logits
        out = build_from_mamba(sd, "mamba").eval()(ids)["lm_logits"]
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(reference_logits(sd, ids), ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("arch", [a for a in ARCHS if a != "mamba"])
def test_constrained_variants_build_and_respect_bounds(arch):
    sd = tiny_state_dict(V)
    ids = _ids()
    xm = calibrate_x_max(build_from_mamba(sd, "mamba"), [ids])
    model = build_from_mamba(sd, arch, x_max=xm).eval()
    out = model(ids)["lm_logits"]
    assert torch.isfinite(out).all()
    cfg = model.cfg
    for blk in model.blocks:
        s = blk.ssm
        if cfg.reparam_eigen:
            assert s.A.lambdas.min() >= cfg.lambda_min and s.A.lambdas.max() <= cfg.lambda_max
        if cfg.spectral_norm:
            for lin, bound in ((s.x_to_b, cfg.s_b), (s.x_to_c, cfg.s_c), (s.delta_proj.proj, cfg.s_delta), (blk.out_proj, cfg.s_out)):
                assert torch.linalg.matrix_norm(lin.normalised_weight(), ord=2) <= bound * (1 + 1e-4)


def test_conversion_error_is_exact_in_double_and_detects_a_wrong_mapping():
    from lipmamba.pretrained import conversion_error
    sd = tiny_state_dict(V)
    ids = _ids()
    model = build_from_mamba(sd, "mamba").eval()
    assert conversion_error(model, sd, ids) < 1e-9
    assert model.embed_tokens.weight.dtype == torch.float32
    s = model.blocks[0].ssm
    s.x_to_b.weight.data, s.x_to_c.weight.data = s.x_to_c.weight.data.clone(), s.x_to_b.weight.data.clone()
    assert conversion_error(model, sd, ids) > 1e-3


def test_calibrated_constraints_start_close_to_the_checkpoint():
    from lipmamba.pretrained import calibrate_constraints, describe
    sd = tiny_state_dict(V)
    base = build_from_mamba(sd, "mamba").eval()
    batches = [_ids(8, 32, seed=s) for s in range(4)]
    cal = calibrate_constraints(base, batches)
    ids = _ids(4, 32, seed=9)
    ref = base(ids)["lm_logits"]
    rel = lambda m: float((m(ids)["lm_logits"] - ref).abs().max() / ref.abs().max())
    calibrated = build_from_mamba(sd, "lip", calibration=cal).eval()
    paper = build_from_mamba(sd, "lip", calibration={"x_max": cal["x_max"]}).eval()
    assert rel(calibrated) < 0.5 * rel(paper)
    d = describe(calibrated)
    assert len(d["delta_max"]) == 2 and all(0 < r < 1 for r in d["rho_min"])
    for blk, smax in zip(calibrated.blocks, cal["s_out"]):
        assert torch.linalg.matrix_norm(blk.out_proj.normalised_weight(), ord=2) <= smax * (1 + 1e-4)


def test_calibrated_projection_keeps_slow_eigenvalues():
    """Trained checkpoints spread exp(A_log) over orders of magnitude; the projection must not move the slow ones."""
    from lipmamba.pretrained import calibrate_constraints
    sd = tiny_state_dict(V)
    g = torch.Generator().manual_seed(0)
    for k in sd:
        if k.endswith("A_log"):
            sd[k] = torch.empty_like(sd[k]).uniform_(-7, 5, generator=g)
    base = build_from_mamba(sd, "mamba").eval()
    cal = calibrate_constraints(base, [_ids(8, 32, seed=s) for s in range(4)])
    model = build_from_mamba(sd, "lip", calibration=cal).eval()
    for b0, b1 in zip(base.blocks, model.blocks):
        assert torch.allclose(b1.ssm.A.lambdas, b0.ssm.A.lambdas, rtol=1e-3, atol=0)
