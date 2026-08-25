import torch

from .module import ESMoE


def collect_aux_loss(model):
    total = None
    for m in model.modules():
        if isinstance(m, ESMoE):
            total = m.aux_loss if total is None else total + m.aux_loss
    return total if total is not None else torch.zeros(())
