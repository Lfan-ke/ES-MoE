"""ES-MoE block on Flax: a softmax router, top-k over heterogeneous experts, and a balancing loss.

An `nnx.Module`, so it is written the way Flax is written now: state lives on the object, and
`nnx.split` hands it to a pure function when one is wanted. Inputs are NHWC.
"""

from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Self

import jax
import jax.numpy as jnp
from flax import nnx

from ..kernels import (
    Kernel,
    odd_kernels,
)
from ..naming import (
    named,
    resolve,
)
from ..spec import (
    SETTINGS,
    settle,
)
from .balance import BALANCES
from .experts import (
    EXPERTS,
    DWExpert,
    ExpertFactory,
)


class ESMoE(nnx.Module):
    """Mixture-of-experts block, channel-preserving unless ``out_channels`` says otherwise.

    Takes the arguments every backend takes, because they are read by the same code. Flax builds
    its parameters at construction, so ``channels`` is required here rather than inferred.
    """

    def __init__(
        self,
        num_experts: int | Mapping = 4,
        top_k: int | None = 2,
        channels: int | None = None,
        options: Mapping | None = None,
        *,
        rngs: nnx.Rngs,
        out_channels: int | None = None,
        kernels: Sequence[Kernel] | None = None,
        expert: ExpertFactory | str | Sequence[ExpertFactory | str] = DWExpert,
        reduction: int = 8,
        max_kernel: int = 15,
        **settings,
    ):
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
        if not plan.channels:
            raise ValueError("channels is required on Flax: nnx builds its parameters at construction")
        self.num_experts, self.top_k = plan.num_experts, plan.top_k
        self.kernels, self.max_kernel = plan.kernels, plan.max_kernel
        self.reduction = plan.reduction
        self.channels = int(plan.channels)
        self.out_channels = plan.out_channels or self.channels
        self.expert_factory = [resolve(e, EXPERTS) if isinstance(e, str) else e for e in plan.expert]
        for key, value in plan.settings.items():
            setattr(self, key, resolve(value, BALANCES) if key == "balance" and isinstance(value, str) else value)

        hidden = max(self.channels // self.reduction, 8)
        self.experts = nnx.List(
            [
                maker(self.channels, self.out_channels, k, rngs=rngs)
                for maker, k in zip(self.expert_factory, self.kernels, strict=True)
            ]
        )
        self.router = nnx.List(
            [nnx.Linear(self.channels, hidden, rngs=rngs), nnx.Linear(hidden, self.num_experts, rngs=rngs)]
        )
        self.norm = nnx.BatchNorm(self.out_channels, rngs=rngs) if self.out_norm else None
        self.aux = nnx.Intermediate(jnp.zeros(()))

    def route(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        pooled = x.mean(axis=(1, 2))
        # Unclamped, a runaway logit reaches softmax as inf under mixed precision and NaNs the gate.
        logits = jnp.clip(self.router[1](nnx.silu(self.router[0](pooled))).astype(jnp.float32), -30.0, 30.0)
        probs = nnx.softmax(logits, axis=1).astype(x.dtype)
        weights, chosen = jax.lax.top_k(probs, self.top_k)
        gate = (nnx.one_hot(chosen, self.num_experts, dtype=probs.dtype) * weights[..., None]).sum(axis=1)
        return probs, gate / jnp.maximum(gate.sum(axis=1, keepdims=True), 1e-9)

    def __call__(self, x: jax.Array) -> jax.Array:
        probs, gate = self.route(x)
        out = sum(gate[:, index].reshape(-1, 1, 1, 1) * expert(x) for index, expert in enumerate(self.experts))
        self.aux[...] = self.balance(probs, gate)
        return self.norm(out) if self.norm is not None else out

    def configure(self, **settings) -> Self:
        """Set the alignment switches on a block that already exists."""
        if unknown := set(settings) - set(SETTINGS):
            raise ValueError(f"unknown ESMoE options {sorted(unknown)}; expected {sorted(SETTINGS)}")
        if (balance := settings.pop("balance", None)) is not None:
            self.balance = resolve(balance, BALANCES) if isinstance(balance, str) else balance
        for key, value in settings.items():
            setattr(self, key, value)
        return self

    def spec(self) -> dict[str, object]:
        """The settings this block is holding, in the form a config and a record carry them."""
        found: dict[str, object] = {"balance": named(self.balance, BALANCES)}
        found |= {key: getattr(self, key) for key in SETTINGS if key != "balance"}
        makers = self.expert_factory
        uniform = makers[0] if all(m is makers[0] for m in makers) else None
        if uniform is not DWExpert:
            found["expert"] = named(uniform, EXPERTS) if uniform else [named(m, EXPERTS) for m in makers]
        if list(self.kernels) != odd_kernels(self.num_experts, self.max_kernel):
            found["kernels"] = list(self.kernels)
        return found


def blocks(model: nnx.Module):
    """Every ESMoE block inside ``model``, in module order."""
    seen = [model] if isinstance(model, ESMoE) else []
    seen += [value for _, value in nnx.iter_graph(model) if isinstance(value, ESMoE) and value is not model]
    return iter(dict.fromkeys(seen))


__all__ = [
    "ESMoE",
    "blocks",
]
