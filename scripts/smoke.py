# 8.24 gate smoke: minimal ES-MoE forward/backward on an external backbone stand-in,
# asserting a non-zero router aux loss.
import torch

from esmoe import ESMoE
from esmoe.aux_loss import collect_aux_loss


def main():
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, 2, 1), torch.nn.SiLU(),
        ESMoE(32, 32, num_experts=4, top_k=2),
        torch.nn.Conv2d(32, 16, 1),
    )
    x = torch.randn(4, 3, 32, 32)
    y = net(x)
    aux = collect_aux_loss(net)
    loss = y.mean() + 0.01 * aux
    loss.backward()
    print(f"out={tuple(y.shape)} experts_kernels={net[2].expert_kernel_sizes}")
    print(f"aux_loss={aux.item():.6f}")
    assert aux.item() > 0, "aux loss must be non-zero"
    grads = sum(p.grad.abs().sum().item() for p in net[2].router.parameters() if p.grad is not None)
    assert grads > 0, "router must receive gradient"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
