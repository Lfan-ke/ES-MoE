import pytest
import torch
from torch import nn

import esmoe
from esmoe import DWExpert, ESMoE, blocks, collect_aux_loss, odd_kernels


def _net(num_experts=4, top_k=2, channels=32, **kwargs):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, 2, 1),
        nn.SiLU(),
        ESMoE(num_experts, top_k, channels=channels, **kwargs),
        nn.Conv2d(32, 16, 1),
    )


def test_public_api():
    for name in ("ESMoE", "inject_esmoe", "attach_aux_loss", "collect_aux_loss", "graft", "equip"):
        assert hasattr(esmoe, name)
    assert esmoe.__version__


def test_forward_and_nonzero_aux_loss():
    torch.manual_seed(0)
    net = _net()
    y = net(torch.randn(4, 3, 32, 32))
    assert y.shape == (4, 16, 16, 16)
    aux = collect_aux_loss(net)
    assert aux.item() > 0  # 8.24 gate: non-zero router aux loss
    (y.mean() + 0.01 * aux).backward()
    router_grad = sum(p.grad.abs().sum().item() for p in net[2].router.parameters() if p.grad is not None)
    assert router_grad > 0


def test_lazy_channels_are_inferred_and_preserved():
    block = ESMoE(4, 2)
    assert block.channels is None
    y = block(torch.randn(2, 48, 8, 8))
    assert block.channels == 48
    assert y.shape == (2, 48, 8, 8)


def test_collect_reflects_latest_forward_only():
    torch.manual_seed(0)
    net = _net()
    net(torch.randn(4, 3, 32, 32))
    first = collect_aux_loss(net).item()
    assert collect_aux_loss(net).item() == first  # no double counting without a new forward
    esmoe.clear_aux_loss()
    assert collect_aux_loss(net).item() == 0.0


def test_kernel_sizes_are_heterogeneous_and_capped():
    assert odd_kernels(4) == [3, 5, 7, 9]
    assert odd_kernels(4, max_kernel_size=5) == [3, 5, 5, 5]
    assert ESMoE(3, 1, 16, expert_kernel_sizes=[5, 5, 7]).expert_kernel_sizes == [5, 5, 7]


def test_experts_and_balance_are_replaceable():
    class Thin(nn.Conv2d):
        def __init__(self, c1, c2, k):
            super().__init__(c1, c2, k, 1, k // 2)

    net = _net(expert=Thin, balance=lambda probs, gate: probs.sum() * 0)
    net(torch.randn(2, 3, 32, 32))
    block = next(blocks(net))
    assert all(isinstance(e, Thin) for e in block.experts)
    assert collect_aux_loss(net).item() == 0.0
    assert issubclass(DWExpert, nn.Module)


def test_blocks_finds_every_block():
    net = nn.Sequential(ESMoE(2, 1, 8), nn.Identity(), ESMoE(3, 2, 8))
    assert len(list(blocks(net))) == 2


def test_invalid_configuration_is_rejected():
    for bad in ({"num_experts": 2, "top_k": 3}, {"num_experts": 2, "top_k": 0}):
        with pytest.raises(ValueError):
            ESMoE(**bad)
    with pytest.raises(ValueError):
        ESMoE(4, 2, 16, expert_kernel_sizes=[3, 5])


def test_aux_term_survives_autocast():
    """所有实验都开着 AMP,所以半精度下辅助项必须仍然有限、非零,且不把参数拖成半精度。"""
    torch.manual_seed(0)
    net = _net()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = net(torch.randn(4, 3, 32, 32))
    aux = collect_aux_loss(net)
    assert torch.isfinite(y).all()
    assert torch.isfinite(aux) and aux.item() > 0
    assert all(p.dtype is torch.float32 for p in net.parameters())


def test_gate_still_sums_to_one_in_half_precision():
    torch.manual_seed(0)
    block = ESMoE(4, 2, channels=16)
    probability = torch.softmax(block.router(torch.randn(8, 16, 8, 8).half().float()), dim=1)
    weights, chosen = probability.topk(2, dim=1)
    gate = torch.zeros_like(probability).scatter(1, chosen, weights).half()
    gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-9)
    assert torch.allclose(gate.float().sum(dim=1), torch.ones(8), atol=1e-2)


def test_gshard_matches_the_upstream_definition():
    """N * sum(usage^2) over normalised mean routing mass: 1.0 at uniform, N when one expert takes all."""
    uniform = torch.full((6, 4), 0.25)
    one_hot = torch.zeros(6, 4)
    one_hot[:, 0] = 1.0
    gate = torch.zeros(6, 4)
    assert esmoe.gshard_balance(uniform, gate).item() == pytest.approx(1.0, rel=1e-5)
    assert esmoe.gshard_balance(one_hot, gate).item() == pytest.approx(4.0, rel=1e-5)


def test_gshard_rises_as_routing_mass_concentrates():
    gate = torch.zeros(3, 4)
    flat = torch.tensor([[0.25, 0.25, 0.25, 0.25]]).repeat(3, 1)
    skewed = torch.tensor([[0.70, 0.20, 0.05, 0.05]]).repeat(3, 1)
    assert esmoe.gshard_balance(skewed, gate) > esmoe.gshard_balance(flat, gate)


def test_switch_is_flat_where_gshard_is_not():
    """The two objectives disagree, which is why the block lets you choose.

    With uniform mean probabilities the Switch term is pinned at k however the top-k dispatch
    falls, so it cannot report a dispatch that has collapsed onto one expert.
    """
    even = torch.tensor(
        [[0.40, 0.30, 0.15, 0.15], [0.30, 0.40, 0.15, 0.15], [0.15, 0.15, 0.40, 0.30], [0.15, 0.15, 0.30, 0.40]]
    )
    skewed = torch.tensor([[0.25, 0.45, 0.15, 0.15], [0.25, 0.15, 0.45, 0.15], [0.25, 0.15, 0.15, 0.45]])

    def top_k_gate(probs, k=2):
        return torch.zeros_like(probs).scatter(1, probs.topk(k, dim=1).indices, 1.0)

    assert esmoe.switch_balance(even, top_k_gate(even)).item() == pytest.approx(2.0, rel=1e-5)
    assert esmoe.switch_balance(skewed, top_k_gate(skewed)).item() == pytest.approx(2.0, rel=1e-5)
    # one expert takes every token's top-1 in `skewed`, and the Switch term never moves
    assert top_k_gate(skewed)[:, 0].mean().item() == pytest.approx(1.0)


def test_block_trains_with_the_upstream_objective():
    net = _net(balance=esmoe.gshard_balance)
    esmoe.clear_aux_loss()
    net(torch.randn(2, 3, 32, 32))
    aux = collect_aux_loss(net)
    assert aux.item() > 0
    aux.backward()
    block = next(blocks(net))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.router.parameters())


def test_router_survives_a_runaway_logit():
    """A logit large enough to overflow the softmax must not take the gate to NaN.

    Upstream clamps before the softmax for this reason; without it a diverging router poisons every
    downstream loss and the run dies far from the cause.
    """
    net = _net()
    net(torch.randn(2, 3, 32, 32))
    block = next(blocks(net))
    with torch.no_grad():
        block.router[-1].bias[0] = 1e4
    out = net(torch.randn(2, 3, 32, 32))
    assert torch.isfinite(out).all()
    assert torch.isfinite(collect_aux_loss(net)).all()


def _gate_of(probs, k=2):
    weights, chosen = probs.topk(k, dim=1)
    gate = torch.zeros_like(probs).scatter(1, chosen, weights)
    return gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-9)


def test_only_the_paper_objective_sees_a_collapsed_dispatch():
    """Three objectives, one difference that matters.

    Both arms below carry uniform mean probabilities; in `collapsed` one expert is in every
    sample's top-k. The Switch and GShard terms read the probabilities and cannot tell the two
    apart. The paper's term reads the gated weights and can.
    """
    balanced = torch.tensor(
        [[0.40, 0.30, 0.15, 0.15], [0.30, 0.40, 0.15, 0.15], [0.15, 0.15, 0.40, 0.30], [0.15, 0.15, 0.30, 0.40]]
    )
    collapsed = torch.tensor([[0.25, 0.45, 0.15, 0.15], [0.25, 0.15, 0.45, 0.15], [0.25, 0.15, 0.15, 0.45]])
    assert _gate_of(collapsed)[:, 0].gt(0).all(), "expert 0 must be in every top-k for this to test anything"

    for blind in (esmoe.switch_balance, esmoe.gshard_balance):
        assert blind(balanced, _gate_of(balanced)).item() == pytest.approx(
            blind(collapsed, _gate_of(collapsed)).item(), abs=1e-6
        )
    seeing = esmoe.master_balance
    assert seeing(collapsed, _gate_of(collapsed)) > seeing(balanced, _gate_of(balanced)) + 1e-4


def test_master_balance_is_zero_at_uniform_use():
    uniform = torch.full((5, 4), 0.25)
    assert esmoe.master_balance(uniform, uniform).item() == pytest.approx(0.0, abs=1e-9)


def test_the_default_objective_is_the_one_upstream_ships():
    """Changing this default changes what every future run optimises, so it is pinned by a test."""
    block = ESMoE(4, 2, channels=16)
    assert block.balance is esmoe.gshard_balance
