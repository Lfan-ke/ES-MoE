"""ES-MoE block on TensorFlow: a softmax router, top-k over heterogeneous experts, and a balancing loss.

A `tf.Module`, so the backend stands on TensorFlow alone. Inputs are NHWC.
"""

from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Self

import tensorflow as tf

from .. import registry
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
    GLOROT,
    DWExpert,
    ExpertFactory,
)


class Dense(tf.Module):
    def __init__(self, width: int, units: int):
        super().__init__()
        self.weight = tf.Variable(GLOROT([width, units]), name="weight")
        self.bias = tf.Variable(tf.zeros([units]), name="bias")

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return tf.matmul(x, self.weight) + self.bias


class ESMoE(tf.Module):
    """Mixture-of-experts block, channel-preserving unless ``out_channels`` says otherwise.

    Takes the arguments every backend takes, because they are read by the same code. Channels come
    from the first call.
    """

    balance: object
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
        expert: ExpertFactory | str | Sequence[ExpertFactory | str] = DWExpert,
        reduction: int = 8,
        max_kernel: int = 15,
        name: str | None = None,
        **settings,
    ):
        super().__init__(name=name)
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
        self.reduction, self.out_channels, self.channels = plan.reduction, plan.out_channels, None
        self.expert_factory = [resolve(e, EXPERTS) if isinstance(e, str) else e for e in plan.expert]
        for key, value in plan.settings.items():
            setattr(self, key, resolve(value, BALANCES) if key == "balance" and isinstance(value, str) else value)
        self.experts: list[tf.Module] = []
        self.router: list[Dense] = []
        self.norm = None
        if plan.channels:
            self.build(int(plan.channels))

    def build(self, channels: int) -> None:
        if self.channels is not None:
            return
        from .experts import BatchNorm

        width = self.out_channels or channels
        self.experts = [maker(channels, width, k) for maker, k in zip(self.expert_factory, self.kernels, strict=True)]
        self.router = [
            Dense(channels, max(channels // self.reduction, 8)),
            Dense(max(channels // self.reduction, 8), self.num_experts),
        ]
        self.norm = BatchNorm(width) if self.out_norm else None
        self.channels, self.out_channels = channels, width

    def route(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        pooled = tf.reduce_mean(x, axis=[1, 2])
        # Unclamped, a runaway logit reaches softmax as inf under mixed precision and NaNs the gate.
        logits = tf.clip_by_value(tf.cast(self.router[1](tf.nn.silu(self.router[0](pooled))), tf.float32), -30.0, 30.0)
        probs = tf.cast(tf.nn.softmax(logits, axis=1), x.dtype)
        weights, chosen = tf.math.top_k(probs, self.top_k)
        gate = tf.reduce_sum(tf.one_hot(chosen, self.num_experts, dtype=probs.dtype) * weights[..., None], axis=1)
        return probs, gate / tf.maximum(tf.reduce_sum(gate, axis=1, keepdims=True), 1e-9)

    def __call__(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        if self.channels is None:
            self.build(int(x.shape[-1]))
        probs, gate = self.route(x)
        if self.dynamic_threshold and not training and self.top_k < self.num_experts:
            # The share is the weight after renormalisation; read before, 0.4 drops most runners-up.
            leader = tf.one_hot(
                tf.argmax(gate, axis=1), self.num_experts, dtype=tf.bool, on_value=True, off_value=False
            )
            gate = gate * tf.cast((gate >= self.dynamic_threshold) | leader, gate.dtype)
            gate = gate / tf.maximum(tf.reduce_sum(gate, axis=1, keepdims=True), 1e-9)
        out = tf.add_n(
            [
                tf.reshape(gate[:, index], [-1, 1, 1, 1]) * expert(x, training)
                for index, expert in enumerate(self.experts)
            ]
        )
        registry.publish(self, self.balance(probs, gate))
        return self.norm(out, training) if self.norm is not None else out

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


def blocks(model: tf.Module):
    """Every ESMoE block inside ``model``, in module order."""
    found = model.submodules if isinstance(model, tf.Module) else ()
    return (m for m in (model, *found) if isinstance(m, ESMoE))


__all__ = [
    "Dense",
    "ESMoE",
    "blocks",
]
