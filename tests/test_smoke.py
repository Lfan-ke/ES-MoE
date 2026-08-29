import torch

import esmoe
from esmoe import ESMoE, collect_aux_loss


def _block_net(num_experts=4, top_k=2, channels=32):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, 2, 1),
        torch.nn.SiLU(),
        ESMoE(num_experts, top_k, channels=channels),
        torch.nn.Conv2d(32, 16, 1),
    )


def test_public_api():
    for name in ("ESMoE", "inject_esmoe", "attach_aux_loss", "collect_aux_loss", "graft"):
        assert hasattr(esmoe, name)
    assert esmoe.__version__


def test_forward_and_nonzero_aux_loss():
    torch.manual_seed(0)
    net = _block_net()
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
    net = _block_net()
    net(torch.randn(4, 3, 32, 32))
    first = collect_aux_loss(net).item()
    assert collect_aux_loss(net).item() == first  # no double counting without a new forward
    esmoe.clear_aux_loss()
    assert collect_aux_loss(net).item() == 0.0


def test_kernel_sizes_are_heterogeneous():
    assert ESMoE(4, 2, channels=16).expert_kernel_sizes == [3, 5, 7, 9]
    assert ESMoE(3, 1, channels=16, expert_kernel_sizes=[5, 5, 7]).expert_kernel_sizes == [5, 5, 7]
