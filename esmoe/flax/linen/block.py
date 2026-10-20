"""ES-MoE block on Flax Linen, the other API Flax still ships.

Linen keeps the parameters outside the module, so the block is declared once and run with
``apply``; the balancing term is sown into ``intermediates``, which is how Linen carries a value
out of a forward pass.
"""

from collections.abc import (
    Mapping,
    Sequence,
)

import jax
import jax.numpy as jnp
from flax import linen

from ...core.kernels import (
    Kernel,
    odd_kernels,
)
from ...core.naming import (
    named,
    resolve,
)
from ...core.spec import (
    SETTINGS,
    settle,
)
from ..balance import BALANCES
from .experts import (
    EXPERTS,
    DWExpert,
)


class ESMoE(linen.Module):
    """Mixture-of-experts block, channel-preserving unless ``out_channels`` says otherwise.

    Takes the arguments every backend takes, read by the same code; Linen has no ``**kwargs``, so
    the settings arrive in ``options``.
    """

    num_experts: int = 4
    top_k: int | None = 2
    channels: int | None = None
    options: Mapping | None = None
    out_channels: int | None = None
    kernels: Sequence[Kernel] | None = None
    expert: object = DWExpert
    reduction: int = 8
    max_kernel: int = 15

    def plan(self):
        return settle(
            {
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "channels": self.channels,
                "out_channels": self.out_channels,
                "kernels": self.kernels,
                "expert": self.expert,
                "reduction": self.reduction,
                "max_kernel": self.max_kernel,
            },
            self.options,
            None,
        )

    @linen.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        plan = self.plan()
        channels = int(plan.channels or x.shape[-1])
        width = plan.out_channels or channels
        balance = plan.settings["balance"]
        balance = resolve(balance, BALANCES) if isinstance(balance, str) else balance

        pooled = x.mean(axis=(1, 2))
        hidden = linen.silu(linen.Dense(max(channels // plan.reduction, 8))(pooled))
        # Unclamped, a runaway logit reaches softmax as inf under mixed precision and NaNs the gate.
        logits = jnp.clip(linen.Dense(plan.num_experts)(hidden).astype(jnp.float32), -30.0, 30.0)
        probs = linen.softmax(logits, axis=1).astype(x.dtype)
        weights, chosen = jax.lax.top_k(probs, plan.top_k)
        gate = (linen.one_hot(chosen, plan.num_experts, dtype=probs.dtype) * weights[..., None]).sum(axis=1)
        gate = gate / jnp.maximum(gate.sum(axis=1, keepdims=True), 1e-9)

        made = [
            (resolve(e, EXPERTS) if isinstance(e, str) else e)(channels, width, k)
            for e, k in zip(plan.expert, plan.kernels, strict=True)
        ]
        out = sum(gate[:, index].reshape(-1, 1, 1, 1) * expert(x, training) for index, expert in enumerate(made))
        self.sow("intermediates", "aux", balance(probs, gate))
        if plan.settings["out_norm"]:
            out = linen.silu(linen.BatchNorm(use_running_average=not training)(out))
        return out

    def spec(self) -> dict[str, object]:
        """The settings this block is holding, in the form a config and a record carry them."""
        plan = self.plan()
        found: dict[str, object] = dict(plan.settings)
        if not isinstance(found["balance"], str):
            found["balance"] = named(found["balance"], BALANCES)
        if plan.expert[0] is not DWExpert:
            found["expert"] = named(plan.expert[0], EXPERTS)
        if list(plan.kernels) != odd_kernels(plan.num_experts, plan.max_kernel):
            found["kernels"] = list(plan.kernels)
        return {key: found[key] for key in (*SETTINGS, *(k for k in found if k not in SETTINGS))}


__all__ = ["ESMoE"]
