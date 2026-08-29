"""Auxiliary-loss collection over the ESMoE blocks of a model."""

import torch
from torch import Tensor, nn

from . import registry
from .module import blocks


def collect_aux_loss(model: nn.Module, device: torch.device | str | None = None) -> Tensor:
    """Sum this step's router losses over every ESMoE block in ``model``.

    Only values published by the latest forward are summed, so calling this twice without a forward
    in between cannot double-count a stale term.
    """
    published = [value for block in blocks(model) if (value := registry.take(block)) is not None]
    if not published:
        return registry.zeros(device)
    total = torch.stack(published).sum()
    return total if device is None else total.to(device)


def clear_aux_loss() -> None:
    registry.clear()
