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


def _utilisation(weights: Tensor) -> Tensor:
    """Mean routing mass per expert, normalised to sum to one."""
    usage = weights.mean(dim=0)
    return usage / usage.sum().clamp_min(1e-6)


def switch_balance(probs: Tensor, gate: Tensor) -> Tensor:
    """Switch-Transformer load balancing: routing mass times realised load, summed over experts.

    The only one of the three that multiplies the probabilities by a top-k indicator. Since the
    realised loads sum to k whatever the skew, the term sits at k once the mean probabilities are
    near uniform and stops reporting concentration.
    """
    importance = probs.mean(dim=0)
    load = (gate > 0).float().mean(dim=0)
    return probs.shape[1] * (importance * load).sum()


def gshard_balance(probs: Tensor, gate: Tensor) -> Tensor:
    """``N * sum(usage^2)`` over the gated weights - the term upstream's ES_MOE optimises.

    Upstream feeds this the output of its routing layer, which in training is the softmax *after*
    the top-k mask and renormalisation, not the raw probabilities. Reading the gate is what lets
    the term see a dispatch that has collapsed onto one expert.
    """
    usage = _utilisation(gate)
    return gate.shape[1] * (usage * usage).sum()


def master_balance(probs: Tensor, gate: Tensor) -> Tensor:
    """The YOLO-Master paper's load balancing loss: mean squared deviation from uniform use.

    Equal to `gshard_balance` up to an affine map -- with the utilisations summing to one,
    ``sum((u - 1/E)^2) == sum(u^2) - 1/E``, so this is ``(gshard - 1) / E^2``. Same minimiser,
    gradients proportional. Both are kept because the paper and the released code state the same
    objective at different scales.
    """
    usage = _utilisation(gate)
    return ((usage - 1.0 / gate.shape[1]) ** 2).mean()


def gshard_probs_balance(probs: Tensor, gate: Tensor) -> Tensor:
    """`gshard_balance` computed on the raw probabilities instead of the gated weights.

    Not what upstream does. It exists to isolate one variable -- whether the objective reads the
    probabilities or the dispatch -- because a collapse can hide behind flat probabilities.
    """
    usage = _utilisation(probs)
    return probs.shape[1] * (usage * usage).sum()


# The block settings a model.yaml can carry, in the order the constructor takes them.
SETTINGS = ("balance", "out_norm", "dense_training")

BALANCES: dict[str, BalanceFn] = {
    "switch": switch_balance,
    "gshard": gshard_balance,
    "master": master_balance,
    "gshard_probs": gshard_probs_balance,
}


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
        balance: Auxiliary loss ``(probs, gate) -> scalar``, or a name from `BALANCES`. Defaults
            to the objective YOLO-Master's released ES_MOE optimises; `master_balance` is the
            one its paper specifies, and `switch_balance` the Switch-Transformer form.
        out_norm: Normalise the mixed output, as upstream and the paper's equation 2 do.
            Off by default so the runs already in ``results/`` stay reproducible.
        dense_training: Run every expert while training, weighting the unrouted ones by zero.
            That is what upstream does, and it keeps an unrouted expert's normalisation
            statistics moving. Off by default for the same reason.
        options: The same settings as a mapping, which is how a model.yaml carries them. The
            trainer rebuilds the model from that yaml, so a setting applied to the instance
            afterwards is discarded; one written into the config survives every rebuild.
    """

    def __init__(
        self,
        num_experts: int = 4,
        top_k: int = 2,
        channels: int | None = None,
        options: dict | None = None,
        *,
        reduction: int = 8,
        max_kernel_size: int = 15,
        expert_kernel_sizes: Sequence[int] | None = None,
        expert: ExpertFactory = DWExpert,
        balance: BalanceFn | str = gshard_balance,
        out_norm: bool = False,
        dense_training: bool = False,
    ):
        super().__init__()
        if unknown := set(options or {}) - set(SETTINGS):
            raise ValueError(f"unknown ESMoE options {sorted(unknown)}; expected {SETTINGS}")
        settings = dict(zip(SETTINGS, (balance, out_norm, dense_training), strict=True)) | (options or {})
        balance, out_norm, dense_training = (settings[key] for key in SETTINGS)
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        kernels = list(expert_kernel_sizes) if expert_kernel_sizes else odd_kernels(num_experts, max_kernel_size)
        if len(kernels) != num_experts:
            raise ValueError(f"expert_kernel_sizes needs {num_experts} entries, got {len(kernels)}")
        self.num_experts, self.top_k, self.expert_kernel_sizes = num_experts, top_k, kernels
        self.reduction, self.channels = reduction, None
        self.expert_factory = expert
        self.balance = BALANCES[balance] if isinstance(balance, str) else balance
        self.out_norm, self.dense_training = out_norm, dense_training
        self.experts, self.router = nn.ModuleList(), nn.Sequential()
        self.norm: nn.Module = nn.Identity()
        if channels:
            self.build(channels)

    def build(self, channels: int) -> None:
        if self.channels is not None:
            return
        hidden = max(channels // self.reduction, 8)
        self.experts = nn.ModuleList(self.expert_factory(channels, channels, k) for k in self.expert_kernel_sizes)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.num_experts),
        )
        if self.out_norm:
            self.norm = nn.Sequential(nn.BatchNorm2d(channels), nn.SiLU())
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        if self.channels is None:
            self.build(x.shape[1])
            self.to(x.device)  # never x.dtype: autocast feeds half here, params must stay fp32
        # Clamped before the softmax, as upstream does: a router logit that runs away under mixed
        # precision otherwise reaches softmax as inf and takes the whole gate to NaN.
        probs = F.softmax(self.router(x).float().clamp(-30.0, 30.0), dim=1).type_as(x)
        weights, chosen = probs.topk(self.top_k, dim=1)
        gate = torch.zeros_like(probs).scatter(1, chosen, weights)
        gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(x)
        # Skipping an unrouted expert saves work at run time, but the decision depends on the data:
        # a tracer would bake this batch's routing into the graph and the exported model would keep
        # using these experts for every future input. Under tracing, run all of them. `dense_training`
        # extends that to training, where upstream runs every expert so that an unrouted one keeps
        # its normalisation statistics moving instead of freezing.
        every = torch.jit.is_tracing() or torch.onnx.is_in_onnx_export() or (self.dense_training and self.training)
        for index, expert in enumerate(self.experts):
            share = gate[:, index].view(-1, 1, 1, 1)
            if every or torch.count_nonzero(share):
                out = out + share * expert(x)
        registry.publish(self, self.balance(probs, gate))
        return self.norm(out)

    def configure(self, *, out_norm: bool | None = None, dense_training: bool | None = None) -> "ESMoE":
        """Set the upstream-alignment switches after the block exists.

        ``out_norm`` builds a module, so a YAML-constructed block cannot receive it through the
        positional args ``parse_model`` forwards; this is how a caller turns it on afterwards.
        """
        if dense_training is not None:
            self.dense_training = dense_training
        if out_norm is not None and out_norm != self.out_norm:
            self.out_norm = out_norm
            if not out_norm:
                self.norm = nn.Identity()
            elif self.channels is not None:
                self.norm = nn.Sequential(nn.BatchNorm2d(self.channels), nn.SiLU())
        return self

    @property
    def aux_loss(self) -> Tensor:
        value = registry.take(self)
        return registry.zeros() if value is None else value

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, kernels={self.expert_kernel_sizes}, "
            f"balance={self.balance.__name__}, out_norm={self.out_norm}, "
            f"dense_training={self.dense_training}"
        )


def blocks(model: nn.Module) -> Iterator[ESMoE]:
    """Every ESMoE block inside ``model``, in module order."""
    return (m for m in model.modules() if isinstance(m, ESMoE))
