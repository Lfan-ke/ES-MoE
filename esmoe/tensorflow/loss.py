"""Reading the router's balancing term off a model."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import tensorflow as tf

from .. import registry
from .block import blocks


def collect_aux_loss(model: tf.Module) -> tf.Tensor:
    """Sum this step's router losses over every ESMoE block in ``model``."""
    published = [value for block in blocks(model) if isinstance(value := registry.take(block), tf.Tensor)]
    return tf.add_n(published) if published else tf.constant(0.0)


def clear_aux_loss() -> None:
    registry.clear()


@dataclass(slots=True)
class Routing:
    """What the blocks of one forward pass published."""

    model: tf.Module

    def loss(self) -> tf.Tensor:
        return collect_aux_loss(self.model)


@contextmanager
def routing(model: tf.Module) -> Iterator[Routing]:
    """Run a step with a clean registry, so a block that did not run reports nothing."""
    clear_aux_loss()
    yield Routing(model)


__all__ = [
    "Routing",
    "clear_aux_loss",
    "collect_aux_loss",
    "routing",
]
