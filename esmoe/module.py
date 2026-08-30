"""ES-MoE block: softmax router + top-k over heterogeneous experts, with a load-balancing loss.

Mirrors the structure of YOLO-Master's ES_MOE while staying independent of it. Experts, router
width and the balancing objective are all replaceable, so the block is a base to extend rather
than a fixed recipe.
"""

from collections.abc import Callable, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import registry

ExpertFactory = Callable[[int, int, int], nn.Module]
BalanceFn = Callable[[Tensor, Tensor], Tensor]


def odd_kernels(num_experts: int, max_kernel_size: int = 15) -> list[int]:
    """Heterogeneous odd kernels 3, 5, 7, ... capped at ``max_kernel_size``."""
    return [min(3 + 2 * i, max_kernel_size) | 1 for i in range(num_experts)]


class DWExpert(nn.Module):
    """Depthwise-separable expert: the default branch, and a template for custom ones."""

    def __init__(self, c1: int, c2: int, k: int):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, 1, k // 2, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


def switch_balance(probs: Tensor, gate: Tensor) -> Tensor:
    """Switch-Transformer load balancing: routing mass times realised load, summed over experts."""
    importance = probs.mean(dim=0)
    load = (gate > 0).float().mean(dim=0)
    return probs.shape[1] * (importance * load).sum()


class ESMoE(nn.Module):
    """Channel-preserving mixture-of-experts block.

    Channels are inferred on the first forward pass unless ``channels`` is given. That is what lets
    a stock Ultralytics ``model.yaml`` write ``[-1, 1, ESMoE, [4, 2]]``: ``parse_model`` forwards
    YAML args verbatim for third-party modules and assumes ``c2 == c1``.

    Args:
        num_experts: Number of expert branches.
        top_k: Experts activated per sample.
        channels: Channel count; inferred on first forward when omitted.
        reduction: Router bottleneck ratio.
        max_kernel_size: Cap for the generated odd kernels.
        expert_kernel_sizes: Explicit per-expert kernels, overriding the generated ones.
        expert: Factory ``(c1, c2, k) -> Module`` for a custom expert branch.
        balance: Auxiliary loss ``(probs, gate) -> scalar``.
    """

    def __init__(
        self,
        num_experts: int = 4,
        top_k: int = 2,
        channels: int | None = None,
        *,
        reduction: int = 8,
        max_kernel_size: int = 15,
        expert_kernel_sizes: Sequence[int] | None = None,
        expert: ExpertFactory = DWExpert,
        balance: BalanceFn = switch_balance,
    ):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        kernels = list(expert_kernel_sizes) if expert_kernel_sizes else odd_kernels(num_experts, max_kernel_size)
        if len(kernels) != num_experts:
            raise ValueError(f"expert_kernel_sizes needs {num_experts} entries, got {len(kernels)}")
        self.num_experts, self.top_k, self.expert_kernel_sizes = num_experts, top_k, kernels
        self.reduction, self.channels = reduction, None
        self.expert_factory, self.balance = expert, balance
        self.experts, self.router = nn.ModuleList(), nn.Sequential()
        if channels:
            self.build(channels)

    def build(self, channels: int) -> None:
        if self.channels is not None:
            return
        hidden = max(channels // self.reduction, 1)
        self.experts = nn.ModuleList(self.expert_factory(channels, channels, k) for k in self.expert_kernel_sizes)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.num_experts),
        )
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        if self.channels is None:
            self.build(x.shape[1])
            self.to(x.device)  # never x.dtype: autocast feeds half here, params must stay fp32
        probs = F.softmax(self.router(x), dim=1)
        weights, chosen = probs.topk(self.top_k, dim=1)
        gate = torch.zeros_like(probs).scatter(1, chosen, weights)
        gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(x)
        for index, expert in enumerate(self.experts):
            share = gate[:, index].view(-1, 1, 1, 1)
            if torch.count_nonzero(share):
                out = out + share * expert(x)
        registry.publish(self, self.balance(probs, gate))
        return out

    @property
    def aux_loss(self) -> Tensor:
        value = registry.take(self)
        return registry.zeros() if value is None else value

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, kernels={self.expert_kernel_sizes}"
        )


def blocks(model: nn.Module) -> Iterator[ESMoE]:
    """Every ESMoE block inside ``model``, in module order."""
    return (m for m in model.modules() if isinstance(m, ESMoE))
