"""ES-MoE block on Jax: a softmax router, top-k over heterogeneous experts, and a balancing loss.

Pure functions over a parameter tree, so the block goes under `jax.jit` and `jax.grad` as written.
`init` builds the parameters, `apply` returns the output and the balancing term. Inputs are NHWC.
"""

from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..core.kernels import (
    Kernel,
    odd_kernels,
)
from ..core.naming import (
    named,
    resolve,
)
from ..core.spec import (
    SETTINGS,
    settle,
)
from .balance import BALANCES
from .experts import (
    EXPERTS,
    DWExpert,
    ExpertFactory,
    Params,
    glorot,
)


@dataclass(frozen=True, slots=True)
class ESMoE:
    """Mixture-of-experts block, channel-preserving unless ``out_channels`` says otherwise.

    Takes the arguments every backend takes, because they are read by the same code. Build the
    parameters with `init`, run them with `apply`.
    """

    num_experts: int
    top_k: int
    channels: int | None
    out_channels: int | None
    kernels: list[Kernel]
    expert: list[ExpertFactory]
    reduction: int
    max_kernel: int
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
        made = [resolve(e, EXPERTS) if isinstance(e, str) else e for e in plan.expert]
        for field, value in (
            ("num_experts", plan.num_experts),
            ("top_k", plan.top_k),
            ("channels", plan.channels),
            ("out_channels", plan.out_channels),
            ("kernels", plan.kernels),
            ("expert", made),
            ("reduction", plan.reduction),
            ("max_kernel", plan.max_kernel),
        ):
            object.__setattr__(self, field, value)
        for key, value in plan.settings.items():
            chosen = resolve(value, BALANCES) if key == "balance" and isinstance(value, str) else value
            object.__setattr__(self, key, chosen)

    def init(self, key, channels: int | None = None) -> Params:
        """The parameter tree for a block over ``channels`` inputs."""
        width = int(channels or self.channels or 0)
        if not width:
            raise ValueError("channels must be given to init, or to the block")
        out = self.out_channels or width
        hidden = max(width // self.reduction, 8)
        keys = jax.random.split(key, len(self.kernels) + 2)
        params: Params = {
            "router": {
                "first": glorot(keys[0], (width, hidden)),
                "first_bias": jnp.zeros((hidden,)),
                "second": glorot(keys[1], (hidden, self.num_experts)),
                "second_bias": jnp.zeros((self.num_experts,)),
            },
            "experts": [
                make.init(keys[i + 2], width, out, k)
                for i, (make, k) in enumerate(zip(self.expert, self.kernels, strict=True))
            ],
        }
        if self.out_norm:
            params["norm"] = {"scale": jnp.ones((out,)), "offset": jnp.zeros((out,))}
        return params

    def route(self, params: Params, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        router = params["router"]
        pooled = x.mean(axis=(1, 2))
        hidden = jax.nn.silu(pooled @ router["first"] + router["first_bias"])
        # Unclamped, a runaway logit reaches softmax as inf under mixed precision and NaNs the gate.
        logits = jnp.clip((hidden @ router["second"] + router["second_bias"]).astype(jnp.float32), -30.0, 30.0)
        probs = jax.nn.softmax(logits, axis=1).astype(x.dtype)
        weights, chosen = jax.lax.top_k(probs, self.top_k)
        gate = (jax.nn.one_hot(chosen, self.num_experts, dtype=probs.dtype) * weights[..., None]).sum(axis=1)
        return probs, gate / jnp.maximum(gate.sum(axis=1, keepdims=True), 1e-9)

    def apply(self, params: Params, x: jnp.ndarray, training: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The block's output and this step's balancing term."""
        probs, gate = self.route(params, x)
        if self.dynamic_threshold and not training and self.top_k < self.num_experts:
            # The share is the weight after renormalisation; read before, 0.4 drops most runners-up.
            leader = jax.nn.one_hot(gate.argmax(axis=1), self.num_experts, dtype=bool)
            gate = gate * ((gate >= self.dynamic_threshold) | leader).astype(gate.dtype)
            gate = gate / jnp.maximum(gate.sum(axis=1, keepdims=True), 1e-9)
        out = sum(
            gate[:, index].reshape(-1, 1, 1, 1) * make.apply(held, x)
            for index, (make, held) in enumerate(zip(self.expert, params["experts"], strict=True))
        )
        if self.out_norm:
            norm = params["norm"]
            centred = out - out.mean(axis=(0, 1, 2))
            out = jax.nn.silu(centred * jax.lax.rsqrt(out.var(axis=(0, 1, 2)) + 1e-3) * norm["scale"] + norm["offset"])
        return out, self.balance(probs, gate)

    def spec(self) -> dict[str, object]:
        """The settings this block is holding, in the form a config and a record carry them."""
        found: dict[str, object] = {"balance": named(self.balance, BALANCES)}
        found |= {key: getattr(self, key) for key in SETTINGS if key != "balance"}
        uniform = self.expert[0] if all(m is self.expert[0] for m in self.expert) else None
        if uniform is not DWExpert:
            found["expert"] = named(uniform, EXPERTS) if uniform else [named(m, EXPERTS) for m in self.expert]
        if self.kernels != odd_kernels(self.num_experts, self.max_kernel):
            found["kernels"] = self.kernels
        return found


__all__ = ["ESMoE"]
