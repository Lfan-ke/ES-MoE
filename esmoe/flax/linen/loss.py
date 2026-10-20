"""Reading the balancing term Linen sowed into ``intermediates``."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp


def collect_aux_loss(state: Mapping) -> jax.Array:
    """Sum the sown balancing terms in what ``apply(..., mutable=["intermediates"])`` returned."""
    found: list = []

    def walk(node):
        if not isinstance(node, Mapping):
            return
        for key, value in node.items():
            if key == "aux":
                found.extend(value if isinstance(value, tuple) else [value])
            else:
                walk(value)

    walk(state)
    return sum(found) if found else jnp.zeros(())


__all__ = ["collect_aux_loss"]
