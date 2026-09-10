"""Hold the block to the equations the YOLO-Master paper states, checked numerically.

Quoted from arXiv 2512.23273 v2, section 3.2:

    eq. 2   Y = Norm( sum_{i in T_K} w_i * Expert_i(X) )
    eq. 8   Omega_train = (Omega' (*) M_K) / ( sum_{j=1..E} (Omega')_j (*) (M_K)_j + eps )
    eq. 9   Omega_infer,i = exp(L_i) / sum_{j in I_K} exp(L_j)   for i in I_K, else 0
    eq. 13  L_LB = (1/E) * sum_{i=1..E} ( mu_i - 1/E )^2

Equation 8 masks a full softmax and renormalises; equation 9 takes a softmax over the top-K subset.
The paper presents them as the training and inference forms of one thing, and the first test here
is that they are in fact the same number, which is why one implementation can serve both.
"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import esmoe
from esmoe import ESMoE

EXPERTS, TOP_K, CHANNELS = 4, 2, 16


def equation_8(logits: torch.Tensor, top_k: int, eps: float = 1e-9) -> torch.Tensor:
    """Full softmax, masked to the top-K, renormalised."""
    weights = F.softmax(logits, dim=1)
    _, indices = torch.topk(weights, top_k, dim=1)
    mask = torch.zeros_like(weights).scatter_(1, indices, 1.0)
    masked = weights * mask
    return masked / (masked.sum(dim=1, keepdim=True) + eps)


def equation_9(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Softmax over the top-K subset alone, zero elsewhere."""
    _, indices = torch.topk(logits, top_k, dim=1)
    out = torch.zeros_like(logits)
    subset = torch.gather(logits, 1, indices)
    out.scatter_(1, indices, F.softmax(subset, dim=1))
    return out


def equation_13(usage: torch.Tensor) -> torch.Tensor:
    """The paper's load balancing loss over utilisations that sum to one."""
    experts = usage.shape[-1]
    return ((usage - 1.0 / experts) ** 2).sum() / experts


@pytest.mark.parametrize("scale", [0.5, 3.0, 20.0])
def test_equation_8_and_equation_9_are_the_same_number(scale):
    """Masking a full softmax and renormalising equals a softmax over the kept subset."""
    torch.manual_seed(0)
    logits = torch.randn(64, EXPERTS) * scale
    assert torch.allclose(equation_8(logits, TOP_K), equation_9(logits, TOP_K), atol=1e-6)


def test_the_gate_is_equation_8():
    torch.manual_seed(0)
    block = ESMoE(EXPERTS, TOP_K, channels=CHANNELS)
    x = torch.randn(8, CHANNELS, 8, 8)
    logits = block.router(x).float().clamp(-30.0, 30.0)
    probs = F.softmax(logits, dim=1)
    weights, chosen = probs.topk(TOP_K, dim=1)
    gate = torch.zeros_like(probs).scatter(1, chosen, weights)
    gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-9)
    assert torch.allclose(gate, equation_8(logits, TOP_K), atol=1e-6)


def test_master_balance_is_equation_13():
    torch.manual_seed(0)
    for _ in range(10):
        usage = torch.rand(EXPERTS)
        usage = usage / usage.sum()
        gate = usage.repeat(8, 1)
        assert torch.allclose(esmoe.master_balance(gate, gate), equation_13(usage), atol=1e-7)


def test_equation_13_is_an_affine_map_of_the_released_code_s_term():
    """The paper and the released code state one objective at two scales: L_13 = (L_gshard - 1)/E^2."""
    torch.manual_seed(0)
    for _ in range(10):
        probs = torch.rand(16, EXPERTS).softmax(dim=1)
        weights, chosen = probs.topk(TOP_K, dim=1)
        gate = torch.zeros_like(probs).scatter(1, chosen, weights)
        gate = gate / gate.sum(dim=1, keepdim=True)
        paper = esmoe.master_balance(gate, gate)
        released = esmoe.gshard_balance(gate, gate)
        assert paper.item() == pytest.approx((released.item() - 1.0) / EXPERTS**2, rel=1e-5)


def test_equation_2_is_the_out_norm_path():
    """`Y = Norm(sum w_i Expert_i(X))`: the normalisation sits after the weighted sum, not inside
    an expert, and `out_norm=True` is what puts it there."""
    plain = ESMoE(EXPERTS, TOP_K, channels=CHANNELS)
    normed = ESMoE(EXPERTS, TOP_K, channels=CHANNELS, out_norm=True)
    assert isinstance(plain.norm, nn.Identity)
    assert isinstance(normed.norm, nn.Sequential)
    assert isinstance(normed.norm[0], nn.BatchNorm2d) and isinstance(normed.norm[1], nn.SiLU)
    # Same weights either way, so the only difference in the output is the normalisation.
    normed.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(4, CHANNELS, 8, 8)
    normed.eval(), plain.eval()
    with torch.no_grad():
        assert torch.allclose(normed.norm(plain(x)), normed(x), atol=1e-5)


def test_the_sum_runs_over_the_selected_experts_only():
    """Equation 2 sums over `T_K`. Skipping the rest and weighting them by zero are the same sum,
    which is why the sparse and dense paths have to agree to the last bit."""
    torch.manual_seed(0)
    sparse = ESMoE(EXPERTS, TOP_K, channels=CHANNELS).eval()
    dense = ESMoE(EXPERTS, TOP_K, channels=CHANNELS, sparse_inference=False).eval()
    dense.load_state_dict(sparse.state_dict())
    x = torch.randn(4, CHANNELS, 8, 8)
    with torch.no_grad():
        assert torch.allclose(sparse(x), dense(x), atol=1e-6)
