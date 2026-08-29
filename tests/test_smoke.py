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
