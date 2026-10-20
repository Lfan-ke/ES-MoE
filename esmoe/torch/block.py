"""ES-MoE block: a softmax router, top-k over heterogeneous experts, and a balancing loss.

Structurally a match for YOLO-Master's ES_MOE while staying independent of it; experts, router
width and the balancing objective are all replaceable. The settings are tabulated in docs/API.
"""

from collections.abc import Iterator, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from .. import registry
from ..kernels import Kernel, odd_kernels
from ..naming import named, resolve
from ..spec import SETTINGS, settle
from .balance import BALANCES, BalanceFn
from .experts import EXPERTS, DWExpert, ExpertFactory


class ESMoE(nn.Module):
    """Mixture-of-experts block, channel-preserving unless ``out_channels`` says otherwise.

    Channels are inferred on the first forward pass unless ``channels`` is given, which is what lets
    a stock Ultralytics yaml write ``[-1, 1, ESMoE, [4, 2]]`` and a torch model drop the block into
    an ``nn.Sequential`` without stating a width.

    ``kernels`` decides the experts' receptive fields and, with them, their number: an ``int`` or an
    ``(h, w)`` pair each, even sizes stepping down to odd. ``expert`` decides what a branch is --
    a factory ``(c1, c2, k) -> Module``, a name, or one of those per branch. ``top_k`` decides how
    many run per sample, ``None`` for all of them.

    Positional arguments follow the yaml order (``num_experts``, ``top_k``, ``channels``,
    ``options``), and a mapping among them is read as ``options``. Names other frameworks use are
    accepted; anything unrecognised is ignored with a warning rather than refused.
    """

    balance: BalanceFn
    out_norm: bool
    dense_training: bool
    sparse_inference: bool
    dynamic_threshold: float

    def __init__(
        self,
        num_experts: int | Mapping = 4,
        top_k: int | None = 2,
        channels: int | None = None,
        options: Mapping | None = None,
        *,
        out_channels: int | None = None,
        kernels: Sequence[Kernel] | None = None,
        expert: ExpertFactory | str | Sequence[ExpertFactory | str | nn.Module] = DWExpert,
        reduction: int = 8,
        max_kernel: int = 15,
        **settings,
    ):
        super().__init__()
        if isinstance(num_experts, Mapping):
            num_experts, options = 4, dict(num_experts) | dict(options or {})
        plan = settle(
            {
                "num_experts": num_experts,
                "top_k": top_k,
                "channels": channels,
                "out_channels": out_channels,
                "kernels": kernels,
                "expert": expert,
                "reduction": reduction,
                "max_kernel": max_kernel,
            },
            options,
            settings,
        )
        self.num_experts, self.top_k = plan.num_experts, plan.top_k
        self.kernels, self.max_kernel = plan.kernels, plan.max_kernel
        self.reduction, self.out_channels = plan.reduction, plan.out_channels
        self.listed, self.channels = plan.listed, None
        self.expert_factory = [resolve(e, EXPERTS) if isinstance(e, str) else e for e in plan.expert]
        for key, value in plan.settings.items():
            setattr(self, key, resolve(value, BALANCES) if key == "balance" and isinstance(value, str) else value)
        self.experts, self.router = nn.ModuleList(), nn.Sequential()
        self.norm: nn.Module = nn.Identity()
        if plan.channels:
            self.build(int(plan.channels))

    def build(self, channels: int) -> None:
        if self.channels is not None:
            return
        hidden = max(channels // self.reduction, 8)
        width = self.out_channels or channels
        self.experts = nn.ModuleList(
            maker if isinstance(maker, nn.Module) else maker(channels, width, k)
            for maker, k in zip(self.expert_factory, self.kernels, strict=True)
        )
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.num_experts),
        )
        if self.out_norm:
            self.norm = nn.Sequential(nn.BatchNorm2d(width), nn.SiLU())
        self.channels, self.out_channels = channels, width

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        if self.channels is None:
            self.build(x.shape[1])
            self.to(x.device)  # never x.dtype: autocast feeds half here, parameters must stay fp32
        # Clamped before the softmax, as upstream does: a router logit that runs away under mixed
        # precision reaches softmax as inf and takes the whole gate to NaN.
        probs = F.softmax(self.router(x).float().clamp(-30.0, 30.0), dim=1).type_as(x)
        weights, chosen = probs.topk(self.top_k, dim=1)
        gate = torch.zeros_like(probs).scatter(1, chosen, weights)
        gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-9)
        if self.dynamic_threshold and not self.training and self.sparse_inference and self.top_k < self.num_experts:
            # The share read here is the weight after renormalisation, never the raw probability:
            # read before, 0.4 would drop the runner-up of a top-2 on nearly every input.
            leader = F.one_hot(chosen[:, 0], probs.shape[1]).to(torch.bool)
            gate = gate * ((gate >= self.dynamic_threshold) | leader)
            gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-9)
        out = x.new_zeros(x.shape[0], self.out_channels or x.shape[1], *x.shape[2:])
        # Skipping an unrouted expert depends on the data, so a tracer would bake this batch's
        # routing into the exported graph; under tracing every expert runs.
        every = (
            torch.jit.is_tracing()
            or torch.onnx.is_in_onnx_export()
            or (self.dense_training and self.training)
            or not (self.sparse_inference or self.training)
        )
        # DDP needs every parameter in the graph on every rank, and ultralytics drops
        # find_unused_parameters once compile is on; an unrouted expert joins at zero instead.
        joined = self.training and not every and dist.is_available() and dist.is_initialized()
        for index, expert in enumerate(self.experts):
            share = gate[:, index].view(-1, 1, 1, 1)
            if every or torch.count_nonzero(share):
                out = out + share * expert(x)
            elif joined:
                touched = sum((p.flatten()[0] for p in expert.parameters()), out.new_zeros(()))
                out = out + touched.to(out.dtype) * 0.0
        registry.publish(self, self.balance(probs, gate))
        out = self.norm(out)
        return [out] if self.listed else out

    def configure(self, **settings) -> "ESMoE":
        """Set the alignment switches on a block that already exists.

        For paths that never reach a trainer; a trainer rebuilds the model from its config and
        discards anything set here, so a training run puts them in the config instead.
        """
        if unknown := set(settings) - set(SETTINGS):
            raise ValueError(f"unknown ESMoE options {sorted(unknown)}; expected {sorted(SETTINGS)}")
        if (balance := settings.pop("balance", None)) is not None:
            self.balance = resolve(balance, BALANCES) if isinstance(balance, str) else balance
        out_norm = settings.pop("out_norm", None)
        for key, value in settings.items():
            setattr(self, key, value)
        if out_norm is not None and out_norm != self.out_norm:
            self.out_norm = out_norm
            if not out_norm:
                self.norm = nn.Identity()
            elif self.channels is not None:
                self.norm = nn.Sequential(nn.BatchNorm2d(self.out_channels or self.channels), nn.SiLU())
        return self

    @property
    def aux_loss(self) -> Tensor:
        value = registry.take(self)
        return value if isinstance(value, Tensor) else torch.zeros(())

    def __setstate__(self, state: dict) -> None:
        """Give a block unpickled from an older checkpoint the settings it predates."""
        super().__setstate__(state)
        for key, default in SETTINGS.items():
            if not hasattr(self, key):
                setattr(self, key, resolve(default, BALANCES) if key == "balance" else default)
        if not hasattr(self, "out_channels"):
            self.out_channels = self.channels
        if not hasattr(self, "listed"):
            self.listed = False
        if not hasattr(self, "kernels"):
            self.kernels = getattr(self, "expert_kernel_sizes", odd_kernels(self.num_experts))
        if not hasattr(self, "max_kernel"):
            self.max_kernel = getattr(self, "max_kernel_size", 15)
        makers = getattr(self, "expert_factory", DWExpert)
        self.expert_factory = makers if isinstance(makers, list) else [makers] * self.num_experts
        if not hasattr(self, "norm"):
            # A submodule, so it is missing from _modules rather than from __dict__.
            self.norm = nn.Identity()

    def spec(self) -> dict:
        """The settings this block is holding, in the form a config and a record carry them."""
        found: dict[str, object] = {"balance": named(self.balance, BALANCES)}
        found |= {key: getattr(self, key) for key in SETTINGS if key != "balance"}
        makers = self.expert_factory
        uniform = makers[0] if all(m is makers[0] for m in makers) else None
        if uniform is not DWExpert:
            found["expert"] = named(uniform, EXPERTS) if uniform else [named(m, EXPERTS) for m in makers]
        if self.kernels != odd_kernels(self.num_experts, self.max_kernel):
            found["kernels"] = self.kernels
        return found

    def extra_repr(self) -> str:
        shape = f"channels={self.channels}"
        if self.out_channels not in (None, self.channels):
            shape += f", out_channels={self.out_channels}"
        settings = ", ".join(f"{key}={value}" for key, value in self.spec().items())
        return f"{shape}, num_experts={self.num_experts}, top_k={self.top_k}, kernels={self.kernels}, {settings}"


def blocks(model: nn.Module) -> Iterator[ESMoE]:
    """Every ESMoE block inside ``model``, in module order."""
    return (m for m in model.modules() if isinstance(m, ESMoE))


__all__ = [
    "ESMoE",
    "blocks",
]
