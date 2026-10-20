"""ES-MoE block on Keras 3, so the same routing runs on tensorflow, jax or torch.

Every expert is evaluated and weighted by the gate: skipping an unrouted one depends on the data,
which a traced jax or tensorflow graph cannot express. Routing and the balancing term are the
torch backend's, bit for bit; only the inference-time saving is torch-only.
"""

from collections.abc import Mapping, Sequence
from typing import Self

import keras
from keras import layers, ops

from ..kernels import Kernel, odd_kernels
from ..naming import named, resolve
from ..spec import SETTINGS, settle
from .balance import BALANCES
from .experts import EXPERTS, DWExpert, ExpertFactory

# Keras owns these; the rest of the keywords belong to the block.
BASE_ARGS = ("name", "dtype", "trainable", "activity_regularizer", "autocast")


class ESMoE(layers.Layer):
    """Mixture-of-experts layer, channel-preserving unless ``out_channels`` says otherwise.

    Takes the arguments the torch block takes, because both read them through `esmoe.spec.settle`.
    Channels come from the first input shape, and the channel axis follows
    ``keras.config.image_data_format()``.
    """

    def __init__(
        self,
        num_experts: int | Mapping = 4,
        top_k: int | None = 2,
        channels: int | None = None,
        options: Mapping | None = None,
        *,
        out_channels: int | None = None,
        kernels: Sequence[Kernel] | None = None,
        expert: ExpertFactory | str | Sequence[ExpertFactory | str | layers.Layer] = DWExpert,
        reduction: int = 8,
        max_kernel: int = 15,
        **settings,
    ):
        super().__init__(**{key: settings.pop(key) for key in tuple(settings) if key in BASE_ARGS})
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
        self.channels = plan.channels
        self.expert_factory = [resolve(e, EXPERTS) if isinstance(e, str) else e for e in plan.expert]
        for key, value in plan.settings.items():
            setattr(self, key, resolve(value, BALANCES) if key == "balance" and isinstance(value, str) else value)
        self.axis = 1 if keras.config.image_data_format() == "channels_first" else -1

    def build(self, input_shape) -> None:
        channels = self.channels or input_shape[self.axis]
        width = self.out_channels or channels
        self.experts = [
            maker if isinstance(maker, layers.Layer) else maker(channels, width, k)
            for maker, k in zip(self.expert_factory, self.kernels, strict=True)
        ]
        self.router = keras.Sequential(
            [
                layers.GlobalAveragePooling2D(),
                layers.Dense(max(channels // self.reduction, 8), activation="silu"),
                layers.Dense(self.num_experts),
            ]
        )
        self.norm = (
            keras.Sequential([layers.BatchNormalization(), layers.Activation("silu")]) if self.out_norm else None
        )
        self.channels, self.out_channels = channels, width
        super().build(input_shape)

    def call(self, x, training=False):
        # Clamped before the softmax: a router logit that runs away under mixed precision reaches
        # softmax as inf and takes the whole gate to NaN.
        logits = ops.clip(ops.cast(self.router(x, training=training), "float32"), -30.0, 30.0)
        probs = ops.cast(ops.softmax(logits, axis=-1), x.dtype)
        weights, chosen = ops.top_k(probs, self.top_k)
        gate = ops.sum(ops.one_hot(chosen, self.num_experts) * ops.expand_dims(weights, -1), axis=1)
        gate = gate / ops.maximum(ops.sum(gate, axis=-1, keepdims=True), 1e-9)
        if self.dynamic_threshold and not training and self.top_k < self.num_experts:
            # The share read here is the weight after renormalisation, never the raw probability.
            leader = ops.one_hot(chosen[:, 0], self.num_experts)
            gate = gate * ops.cast((gate >= self.dynamic_threshold) | ops.cast(leader, "bool"), gate.dtype)
            gate = gate / ops.maximum(ops.sum(gate, axis=-1, keepdims=True), 1e-9)
        shape = [-1] + [1] * (len(x.shape) - 1)
        out = sum(
            ops.reshape(gate[:, index], shape) * expert(x, training=training)
            for index, expert in enumerate(self.experts)
        )
        self.add_loss(self.balance(probs, gate))
        return self.norm(out, training=training) if self.norm else out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[self.axis] = self.out_channels or shape[self.axis]
        return tuple(shape)

    def configure(self, **settings) -> Self:
        """Set the alignment switches on a layer that already exists."""
        if unknown := set(settings) - set(SETTINGS):
            raise ValueError(f"unknown ESMoE options {sorted(unknown)}; expected {sorted(SETTINGS)}")
        if (balance := settings.pop("balance", None)) is not None:
            self.balance = resolve(balance, BALANCES) if isinstance(balance, str) else balance
        for key, value in settings.items():
            setattr(self, key, value)
        return self

    def spec(self) -> dict[str, object]:
        """The settings this layer is holding, in the form a config and a record carry them."""
        found: dict[str, object] = {"balance": named(self.balance, BALANCES)}
        found |= {key: getattr(self, key) for key in SETTINGS if key != "balance"}
        makers = self.expert_factory
        uniform = makers[0] if all(m is makers[0] for m in makers) else None
        if uniform is not DWExpert:
            found["expert"] = named(uniform, EXPERTS) if uniform else [named(m, EXPERTS) for m in makers]
        if self.kernels != odd_kernels(self.num_experts, self.max_kernel):
            found["kernels"] = self.kernels
        return found

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config |= {"num_experts": self.num_experts, "top_k": self.top_k, "options": self.spec()}
        return config


def blocks(model: layers.Layer):
    """Every ESMoE layer inside ``model``, in layer order."""
    return (layer for layer in model._flatten_layers(include_self=True) if isinstance(layer, ESMoE))


__all__ = [
    "ESMoE",
    "blocks",
]
