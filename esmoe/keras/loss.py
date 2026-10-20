"""Reading the router's balancing term off a model.

Keras gathers what a layer hands to ``add_loss`` into ``model.losses`` and starts each forward pass
afresh, so ``model.fit`` already counts the term and nothing has to be cleared by hand.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from keras import layers, ops


def collect_aux_loss(model: layers.Layer):
    """Sum this step's router losses over every ESMoE layer in ``model``."""
    published = list(model.losses)
    return ops.sum(ops.stack(published)) if published else ops.convert_to_tensor(0.0)


def clear_aux_loss() -> None:
    """Nothing to do: Keras clears a layer's losses itself at the start of each forward pass."""


@dataclass(slots=True)
class Routing:
    """What the layers of one forward pass published."""

    model: layers.Layer

    def loss(self):
        return collect_aux_loss(self.model)


@contextmanager
def routing(model: layers.Layer) -> Iterator[Routing]:
    """The torch backend's reader, for a custom loop that does not go through ``fit``."""
    yield Routing(model)


__all__ = [
    "Routing",
    "clear_aux_loss",
    "collect_aux_loss",
    "routing",
]
