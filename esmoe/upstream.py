"""How YOLO-Master's trainer trains a model that holds a routed block, for runs compared against it.

Read from the fork's `nn/mixture_loss.py` (`_collect_mixture_aux_loss`, `CompositeCriterion`),
`engine/trainer.py` (`build_optimizer`) and `engine/extensions/mixture.py` (`begin_epoch`). The fork
applies all three to any model with a routed module, whatever its arguments say, so a comparison
with it has to apply them as well. None of this is the package's default: every recorded run
trained without it.
"""

import torch
from torch import Tensor, nn

from .module import blocks

DECAY, FLOOR, CEILING, BUDGET = 0.99, 1e-4, 1e4, 3.0
ROUTER_LR, EXPERT_WARMUP = 0.5, 3


def normalise(owner: nn.Module, aux: Tensor, weight: float) -> Tensor:
    """The routed term over a running mean of its own magnitude, times ``weight``, capped at `BUDGET`.

    A non-finite term counts as zero. The mean starts at 1.0 and takes this step's magnitude,
    clamped to [`FLOOR`, `CEILING`], before it divides. The cap is a detached factor, so it bounds
    the term without a gradient path through the term's size, and a result still non-finite or
    beyond `CEILING` is dropped. The mean lives on the model being trained, as upstream's buffer
    does, but is not saved with it: a resumed run starts it again at 1.0.
    """
    aux = aux if torch.isfinite(aux) else torch.zeros_like(aux)
    magnitude = min(max(float(aux.detach().abs()), FLOOR), CEILING)
    scale = owner._esmoe_aux_scale = DECAY * getattr(owner, "_esmoe_aux_scale", 1.0) + (1.0 - DECAY) * magnitude
    term = aux / scale * weight
    term = term * min(1.0, BUDGET / max(float(term.detach().abs()), FLOOR))
    return term if torch.isfinite(term) and term.detach().abs() <= CEILING else torch.zeros_like(term)


def split_routers(optimizer: torch.optim.Optimizer, model: nn.Module, decay: float) -> int:
    """Move every router parameter into a group of its own at `ROUTER_LR` times the base rate.

    Upstream sorts parameters by name before anything else, so a router weight never reaches Muon
    and a router bias keeps weight decay. The base rate is the one ``optimizer="auto"`` settles on
    inside `build_optimizer`, so it is read back from the groups: MuSGD gives some of them three
    times the rate and none of them less.
    """
    routers = {id(p): p for block in blocks(model) for p in block.router.parameters()}
    if not routers:
        return 0
    template = next(group for group in optimizer.param_groups if not group.get("use_muon"))
    base = min(group["lr"] for group in optimizer.param_groups)
    for group in optimizer.param_groups:
        group["params"] = [p for p in group["params"] if id(p) not in routers]
    settings = {key: value for key, value in template.items() if key != "params"}
    optimizer.add_param_group(
        settings
        | {"params": list(routers.values()), "lr": base * ROUTER_LR, "weight_decay": decay, "param_group": "router"}
    )
    return len(routers)


def experts(model: nn.Module) -> list[nn.Parameter]:
    """The expert parameters upstream holds back during warm-up, of those the trainer left trainable."""
    return [p for block in blocks(model) for p in block.experts.parameters() if p.requires_grad]


def warm(params: list[nn.Parameter], epoch: int) -> None:
    """No expert trains before epoch `EXPERT_WARMUP`; routers, the output norm and the rest train throughout."""
    for p in params:
        p.requires_grad = epoch >= EXPERT_WARMUP
