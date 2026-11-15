"""The fast scan must compute the same function and gradients as the traced loop."""
import pytest
import torch

from lipmamba.models.selective_ssm import SelectiveSSM, SSMConfig


@pytest.mark.parametrize("constrained", [True, False])
def test_fast_scan_matches_traced_loop(constrained):
    torch.manual_seed(0)
    cfg = SSMConfig(d_model=12, d_inner=24, state_dim=6, spectral_norm=constrained, clamp_delta=constrained,
                    reparam_eigen=constrained, clip_input=constrained)
    ssm = SelectiveSSM(cfg).double().eval()
    ssm.keep_state = True
    x = torch.randn(3, 17, 24, dtype=torch.double, requires_grad=True)
    outs = []
    for record in (True, False):
        y = ssm(x, record_trace=record)
        g, = torch.autograd.grad(y.pow(2).sum() + ssm._state_seq.sum(), x)
        outs.append((y, ssm._last_state, ssm._state_seq, g))
    for a, b in zip(*outs):
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-10)
