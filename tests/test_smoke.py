import torch

import esmoe
from esmoe import ESMoE
from esmoe.aux_loss import collect_aux_loss


def test_public_api():
    assert hasattr(esmoe, "ESMoE")
    assert callable(esmoe.inject_esmoe)
    assert esmoe.__version__


def test_forward_and_nonzero_aux_loss():
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, 2, 1), torch.nn.SiLU(),
        ESMoE(32, 32, num_experts=4, top_k=2),
        torch.nn.Conv2d(32, 16, 1),
    )
    x = torch.randn(4, 3, 32, 32)
    y = net(x)
    assert y.shape == (4, 16, 16, 16)
    aux = collect_aux_loss(net)
    assert aux.item() > 0  # 8.24 gate: non-zero router aux loss
    (y.mean() + 0.01 * aux).backward()
    router_grad = sum(
        p.grad.abs().sum().item() for p in net[2].router.parameters() if p.grad is not None
    )
    assert router_grad > 0
