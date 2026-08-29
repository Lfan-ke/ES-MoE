import torch

from . import registry
from .module import ESMoE


def collect_aux_loss(model, device=None):
    """Sum this step's router losses over every ESMoE block in ``model``.

    Only graph-connected values published by the latest forward are summed, so calling this
    twice without a forward in between cannot double-count a stale term.
    """
    total = None
    for m in model.modules():
        if not isinstance(m, ESMoE):
            continue
        value = registry.take(m)
        if value is None:
            continue
        total = value if total is None else total + value
    if total is None:
        return registry.zeros(device)
    return total if device is None else total.to(device)


def clear_aux_loss():
    registry.clear()


def is_finite(value):
    return bool(torch.isfinite(value))
