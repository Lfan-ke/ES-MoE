"""Making the block resolvable in a model.yaml, and making its loss reach the optimiser."""

import os
from collections.abc import Callable

import torch
from torch import nn

from ..torch.block import ESMoE, blocks
from ..torch.loss import clear_aux_loss, collect_aux_loss
from . import upstream

AUX_NAME = "esmoe_aux"
ENV_WEIGHT = "ESMOE_AUX_WEIGHT"
ENV_RECIPE = "ESMOE_RECIPE"
RECIPES = ("esmoe", "upstream")
# Every spelling a config might use for the block, upstream's included.
NAMES = ("ESMoE", "ES_MoE", "ES_MOE")

__PATCHED: dict[type, Callable] = {}
__WEIGHT: float | None = None
__RECIPE = "esmoe"


def register() -> type[ESMoE]:
    """Expose the block where ultralytics resolves layer names, under every spelling."""
    import ultralytics.nn.tasks as tasks

    for name in NAMES:
        setattr(tasks, name, ESMoE)
    return ESMoE


inject_esmoe = register


def holds_block(base: str) -> bool:
    """Whether a config already names a block, in which case nothing has to be grafted in."""
    from ultralytics.nn.tasks import yaml_model_load

    config = yaml_model_load(base)
    return any(layer[2] in NAMES for layer in [*config["backbone"], *config["head"]])


def equip(base: str = "yolov8n.yaml", *, weight: float = 0.01, recipe: str = "esmoe", out: str | None = None, **kwargs):
    """Register, graft, build and wire the aux loss in one call - the usual entry point.

    A config that already names a block is taken as written, as is ``at=None``; otherwise blocks are
    grafted in and the result written to ``out``, or to a temporary file, because a YOLO wrapper
    loads models by path. Arguments `graft` knows go to it, the rest to ``YOLO``.
    """
    import inspect
    import tempfile
    from pathlib import Path

    from ultralytics import YOLO

    from .graft import OPTIONS, graft

    register()
    wanted = set(inspect.signature(graft).parameters) | OPTIONS
    settings = {key: value for key, value in kwargs.items() if key in wanted}
    source = str(base)
    if settings.get("at", "backbone_end") is not None and not holds_block(base):
        target = Path(out) if out else Path(tempfile.mkdtemp(prefix="esmoe-")) / f"{Path(base).stem}-esmoe.yaml"
        graft(base, out=str(target), **settings)
        source = str(target)
    rest = {key: value for key, value in kwargs.items() if key not in wanted}
    return attach_aux_loss(YOLO(source, **rest), weight=weight, recipe=recipe)


def attach_aux_loss(model, weight: float = 0.01, recipe: str = "esmoe"):
    """Make the router load-balancing loss part of the optimised training loss.

    Without this the term exists but never reaches ``backward``. ``recipe`` says how the model
    trains around it: ``esmoe`` adds ``weight`` times the term per image, the way the task loss
    counts; ``upstream`` reproduces what YOLO-Master's trainer does to any model holding a routed
    block, for runs compared against it (`esmoe.yolo.upstream`).
    """
    core = __core(model)
    if next(blocks(core), None) is None:
        raise ValueError("model contains no ESMoE block; nothing to attach")
    arm_process(weight, recipe)
    __patch(__owner(type(core)))
    core._esmoe_aux_weight, core._esmoe_recipe = float(weight), recipe
    if hasattr(model, "add_callback"):
        # The trainer rebuilds the model, so the settings must reach the instance it trains.
        for event, callback in trainer_callbacks(weight, recipe):
            model.add_callback(event, callback)
    if hasattr(model, "_smart_load"):
        __route_trainer(model)
    return model


def arm_process(weight: float, recipe: str = "esmoe") -> None:
    """Remember the weight and recipe for every model this process builds, and patch the loss entry.

    The trainer rebuilds the model and takes the EMA copy before any callback runs, and a DDP worker
    never sees the instance the caller attached to; process scope is what both of them read.
    """
    global __WEIGHT, __RECIPE
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; expected one of {RECIPES}")
    __WEIGHT, __RECIPE = float(weight), recipe
    os.environ[ENV_WEIGHT], os.environ[ENV_RECIPE] = repr(__WEIGHT), recipe
    import ultralytics.nn.tasks as tasks

    __patch(tasks.BaseModel)


def armed_weight() -> float | None:
    return __WEIGHT


def armed_recipe() -> str:
    return __RECIPE


def __owner(cls: type) -> type:
    # Patch where loss() is defined, not the leaf: RT-DETR and YOLO-World keep their own entry.
    return next(c for c in cls.__mro__ if "loss" in vars(c))


def __patch(cls: type) -> None:
    if cls.loss is __loss_with_aux:
        return
    __PATCHED[cls] = cls.loss
    cls.loss = __loss_with_aux


def __original(model: nn.Module) -> Callable:
    return next(__PATCHED[c] for c in type(model).__mro__ if c in __PATCHED)


def __route_trainer(model) -> None:
    """Make ``model.train()`` pick a trainer class that lives in `esmoe.yolo.trainer`.

    Ultralytics spawns DDP workers from a generated file that imports only the trainer's module; if
    that module is ours, the worker registers the block and the aux weight before it builds.
    """
    from . import trainer

    load = model._smart_load

    def _smart_load(key: str):
        loaded = load(key)
        return trainer.wrap(loaded) if key == "trainer" else loaded

    model._smart_load = _smart_load


def __core(model) -> nn.Module:
    # A bare task model's .model is its layer Sequential, so pick whichever level owns loss().
    for candidate in (getattr(model, "model", None), model):
        if callable(getattr(candidate, "loss", None)):
            return candidate
    raise TypeError(f"{type(model).__name__} exposes no loss() to attach to")


def __unwrap(model) -> nn.Module:
    # The helper was renamed across ultralytics releases; both names mean "drop DDP/EMA wrappers".
    import ultralytics.utils.torch_utils as tu

    fn = getattr(tu, "unwrap_model", None) or getattr(tu, "de_parallel", None)
    return fn(model) if fn else model


def __uses_esmoe(model: nn.Module) -> bool:
    present = getattr(model, "_esmoe_present", None)
    if present is None:
        present = next(blocks(model), None) is not None
        model._esmoe_present = present
    return present


def __loss_with_aux(self, batch, preds=None):
    weight = getattr(self, "_esmoe_aux_weight", None)
    weight = __WEIGHT if weight is None else weight
    if not weight or not __uses_esmoe(self):
        return __original(self)(self, batch, preds)
    if preds is None:
        clear_aux_loss()  # the forward below publishes afresh; last step's value would count twice
    total, items = __original(self)(self, batch, preds)
    aux = collect_aux_loss(self, device=total.device).to(total.dtype)
    if not torch.isfinite(aux):
        aux = torch.zeros_like(aux)
    if (getattr(self, "_esmoe_recipe", None) or __RECIPE) == "upstream":
        # Upstream adds the term to a loss vector already scaled by the batch, once per entry.
        aux = upstream.normalise(self, aux, weight) if self.training else torch.zeros_like(aux)
        total = total + aux
    else:
        # Task criteria optimise a batch-scaled loss but log the per-image value; follow both.
        aux = aux * weight
        scaled = (aux * batch["img"].shape[0]).view(1)
        total = total + scaled.squeeze() if total.ndim == 0 else torch.cat([total.reshape(-1), scaled])
    logged = aux.detach().reshape(1)
    # ultralytics >= 8.4.13x reports loss items as a named dict; older releases return a tensor.
    if isinstance(items, dict):
        return total, items | {AUX_NAME: logged.squeeze()}
    return total, torch.cat([items, logged.to(items.dtype)])


def trainer_callbacks(weight: float, recipe: str = "esmoe") -> list[tuple[str, Callable]]:
    """What a trainer has to run for an armed model it rebuilt from the config."""
    found = [("on_train_start", arm_trainer(weight, recipe))]
    if recipe == "upstream":
        found.append(("on_train_epoch_start", warm_experts))
    return found


def arm_trainer(weight: float, recipe: str = "esmoe") -> Callable:
    def on_train_start(trainer) -> None:
        core = __core(__unwrap(trainer.model))
        core._esmoe_aux_weight, core._esmoe_recipe = float(weight), recipe
        if recipe == "upstream" and getattr(trainer, "_esmoe_routers", None) is None:
            raise RuntimeError(
                "recipe 'upstream' sets the router learning rate in build_optimizer, which only the trainer "
                "attach_aux_loss routes model.train() to overrides; this trainer never split the routers"
            )
        # Older releases fix the loss names up front; newer ones derive them from the loss dict.
        names = tuple(getattr(trainer, "loss_names", ()) or ())
        if names and AUX_NAME not in names:
            trainer.loss_names = (*names, AUX_NAME)

    return on_train_start


def warm_experts(trainer) -> None:
    held = getattr(trainer, "_esmoe_experts", None)
    if held is None:
        # Collected once, so a layer the run froze stays frozen.
        held = trainer._esmoe_experts = upstream.experts(__unwrap(trainer.model))
    upstream.warm(held, trainer.epoch)


__all__ = [
    "AUX_NAME",
    "ENV_RECIPE",
    "ENV_WEIGHT",
    "NAMES",
    "RECIPES",
    "arm_process",
    "arm_trainer",
    "armed_recipe",
    "armed_weight",
    "attach_aux_loss",
    "equip",
    "holds_block",
    "inject_esmoe",
    "register",
    "trainer_callbacks",
    "warm_experts",
]
