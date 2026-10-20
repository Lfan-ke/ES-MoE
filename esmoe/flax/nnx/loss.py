"""Reading the router's balancing term off a model."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx

from .block import blocks


def collect_aux_loss(model: nnx.Module) -> jnp.ndarray:
    """Sum this step's router losses over every ESMoE block in ``model``."""
    published = [block.aux[...] for block in blocks(model)]
    return sum(published) if published else jnp.zeros(())


def clear_aux_loss(model: nnx.Module) -> None:
    for block in blocks(model):
        block.aux[...] = jnp.zeros(())


@dataclass(slots=True)
class Routing:
    """What the blocks of one forward pass published."""

    model: nnx.Module

    def loss(self) -> jnp.ndarray:
        return collect_aux_loss(self.model)


@contextmanager
def routing(model: nnx.Module) -> Iterator[Routing]:
    """Run a step with the term cleared, so a block that did not run reports nothing."""
    clear_aux_loss(model)
    yield Routing(model)


__all__ = [
    "Routing",
    "clear_aux_loss",
    "collect_aux_loss",
    "routing",
]
