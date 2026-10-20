"""Reading the router's balancing term off a model."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn

from ..core import registry
from .block import blocks


def collect_aux_loss(model: nn.Module, device: torch.device | str | None = None) -> Tensor:
    """Sum this step's router losses over every ESMoE block in ``model``.

    Reading does not consume, so a step that computes the loss twice gets the term both times;
    what keeps an earlier step's value out is the clear before the forward pass.
    """
    published = [value for block in blocks(model) if isinstance(value := registry.take(block), Tensor)]
    if not published:
        return torch.zeros((), device=device)
    total = torch.stack(published).sum()
    return total if device is None else total.to(device)


def clear_aux_loss() -> None:
    registry.clear()


class Routing:
    """What the blocks of one forward pass published."""

    def __init__(self, model: nn.Module):
        self.model = model

    def loss(self, device: torch.device | str | None = None) -> Tensor:
        return collect_aux_loss(self.model, device)


@contextmanager
def routing(model: nn.Module) -> Iterator[Routing]:
    """Run a step with a clean registry, so a block that did not run reports nothing.

    with esmoe.routing(net) as route:
        loss = task(net(x), y) + 0.01 * route.loss()
    """
    clear_aux_loss()
    yield Routing(model)


__all__ = [
    "Routing",
    "clear_aux_loss",
    "collect_aux_loss",
    "routing",
]
