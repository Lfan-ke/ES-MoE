"""Load-balancing objectives over the router's output."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

type BalanceFn = Callable[[jax.Array, jax.Array], jax.Array]


def __utilisation(weights: jax.Array) -> jax.Array:
    usage = weights.mean(axis=0)
    return usage / jnp.maximum(usage.sum(), 1e-6)


def switch_balance(probs: jax.Array, gate: jax.Array) -> jax.Array:
    """Switch-Transformer balancing: routing mass times realised load, summed over experts."""
    importance = probs.mean(axis=0)
    load = (gate > 0).astype(probs.dtype).mean(axis=0)
    return probs.shape[1] * (importance * load).sum()


def gshard_balance(probs: jax.Array, gate: jax.Array) -> jax.Array:
    """``E · Σ usage²`` over the gated weights, the term YOLO-Master's ES_MOE optimises."""
    usage = __utilisation(gate)
    return gate.shape[1] * (usage * usage).sum()


def master_balance(probs: jax.Array, gate: jax.Array) -> jax.Array:
    """The YOLO-Master paper's form: mean squared deviation from uniform use."""
    usage = __utilisation(gate)
    return ((usage - 1.0 / gate.shape[1]) ** 2).mean()


def gshard_probs_balance(probs: jax.Array, gate: jax.Array) -> jax.Array:
    """`gshard_balance` on the raw probabilities, to isolate what the objective reads."""
    usage = __utilisation(probs)
    return probs.shape[1] * (usage * usage).sum()


BALANCES: dict[str, BalanceFn] = {
    "switch": switch_balance,
    "gshard": gshard_balance,
    "master": master_balance,
    "gshard_probs": gshard_probs_balance,
}


__all__ = [
    "BALANCES",
    "BalanceFn",
    "gshard_balance",
    "gshard_probs_balance",
    "master_balance",
    "switch_balance",
]
