"""Runtime injection: make ESMoE resolvable in configs and make its loss reach the optimiser."""

import os
from collections.abc import Callable

import torch
from torch import nn

from . import upstream
from .aux_loss import clear_aux_loss, collect_aux_loss
from .module import ESMoE, blocks

AUX_NAME = "esmoe_aux"
ENV_WEIGHT = "ESMOE_AUX_WEIGHT"
ENV_RECIPE = "ESMOE_RECIPE"
RECIPES = ("esmoe", "upstream")
_PATCHED: dict[type, Callable] = {}
_WEIGHT: float | None = None
_RECIPE = "esmoe"


def inject_esmoe() -> type[ESMoE]:
    """Expose ESMoE where ultralytics parse_model resolves layer names, so any model.yaml can
    reference ``ESMoE`` across YOLOv8 / YOLO11 / YOLO12 and every task."""
    import ultralytics.nn.tasks as tasks

    tasks.ESMoE = ESMoE
    return ESMoE


def equip(
    base: str = "yolov8n.yaml", *, weight: float = 0.01, recipe: str = "esmoe", out: str | None = None, **graft_kwargs
):
    """Register, graft, build and wire the aux loss in one call - the usual entry point.

    ``out`` names the grafted config to keep; without it the config still has to reach disk, because
    a YOLO wrapper loads models by path, so it goes to a temporary directory.
    """
    import tempfile
    from pathlib import Path

    from ultralytics import YOLO

    from .graft import graft

    inject_esmoe()
    target = Path(out) if out else Path(tempfile.mkdtemp(prefix="esmoe-")) / f"{Path(base).stem}-esmoe.yaml"
    graft(base, out=str(target), **graft_kwargs)
    return attach_aux_loss(YOLO(str(target)), weight=weight, recipe=recipe)


def attach_aux_loss(model, weight: float = 0.01, recipe: str = "esmoe"):
    """Make the router load-balancing loss part of the optimised training loss.

    Without this the aux term exists but never reaches ``backward``: a config key and a printed
    number prove nothing on their own.

    ``recipe`` says how the model trains around the term. ``"esmoe"``, which every recorded run
    used, adds ``weight`` times the term per image, the way the task loss counts. ``"upstream"`` is
    what YOLO-Master's trainer does to any model with a routed block, for runs compared against it:
    the term over a running mean of its magnitude, times ``weight``, capped at 3.0 and added to each
    native loss term; router parameters at half the learning rate and outside Muon; experts frozen
    for the first three epochs (`esmoe.upstream`). The last two live in the trainer, so they need
    the one ``model.train()`` picks here.
    """
    core = _core(model)
    if next(blocks(core), None) is None:
        raise ValueError("model contains no ESMoE block; nothing to attach")
    arm_process(weight, recipe)
    _patch(_owner(type(core)))
    core._esmoe_aux_weight, core._esmoe_recipe = float(weight), recipe
    if hasattr(model, "add_callback"):
        # The trainer rebuilds the model from yaml, so the settings have to reach the instance it
        # actually trains, not only the one handed to us here.
        for event, callback in trainer_callbacks(weight, recipe):
            model.add_callback(event, callback)
    if hasattr(model, "_smart_load"):
        _route_trainer(model)
    return model


def arm_process(weight: float, recipe: str = "esmoe") -> None:
    """Remember the weight and recipe for every model this process builds and patch the shared loss entry.

    The trainer rebuilds the model and takes the EMA copy before any callback runs, and a DDP worker
    never sees the instance the caller attached to; process scope is what both of them read.
    """
    global _WEIGHT, _RECIPE
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; expected one of {RECIPES}")
    _WEIGHT, _RECIPE = float(weight), recipe
    os.environ[ENV_WEIGHT], os.environ[ENV_RECIPE] = repr(_WEIGHT), recipe
    import ultralytics.nn.tasks as tasks

    _patch(tasks.BaseModel)


def weight() -> float | None:
    return _WEIGHT


def recipe() -> str:
    return _RECIPE


def _owner(cls: type) -> type:
    # Patch where loss() is defined, not the leaf: models that override it (RT-DETR, YOLO-World)
    # keep their own entry, everything that inherits BaseModel.loss shares one.
    return next(c for c in cls.__mro__ if "loss" in vars(c))


def _patch(cls: type) -> None:
    if cls.loss is _loss_with_aux:
        return
    _PATCHED[cls] = cls.loss
    cls.loss = _loss_with_aux


def _original(model: nn.Module) -> Callable:
    return next(_PATCHED[c] for c in type(model).__mro__ if c in _PATCHED)


def _route_trainer(model) -> None:
    """Make ``model.train()`` pick a trainer class that lives in ``esmoe.trainer``.

    Ultralytics spawns DDP workers from a generated file that imports only the trainer's module;
    if that module is ours, the worker registers the block and the aux weight before it builds.
    """
    from . import trainer

    load = model._smart_load

    def _smart_load(key: str):
        loaded = load(key)
        return trainer.wrap(loaded) if key == "trainer" else loaded

    model._smart_load = _smart_load


def _core(model) -> nn.Module:
    # A YOLO wrapper hides the task model under .model, but a bare task model's .model is its layer
    # Sequential - pick whichever level actually owns loss().
    for candidate in (getattr(model, "model", None), model):
        if callable(getattr(candidate, "loss", None)):
            return candidate
    raise TypeError(f"{type(model).__name__} exposes no loss() to attach to")


def _unwrap(model) -> nn.Module:
    # The helper was renamed across ultralytics releases; both names mean "drop DDP/EMA wrappers".
    import ultralytics.utils.torch_utils as tu

    fn = getattr(tu, "unwrap_model", None) or getattr(tu, "de_parallel", None)
    return fn(model) if fn else model


def _uses_esmoe(model: nn.Module) -> bool:
    present = getattr(model, "_esmoe_present", None)
    if present is None:
        present = next(blocks(model), None) is not None
        model._esmoe_present = present
    return present


def _loss_with_aux(self, batch, preds=None):
    # The settings also live at process scope because the trainer rebuilds the model and the EMA copy
    # is taken before any callback runs; an instance-only flag makes those copies report a
    # differently shaped loss than the trainer expects.
    weight = getattr(self, "_esmoe_aux_weight", None)
    weight = _WEIGHT if weight is None else weight
    if not weight or not _uses_esmoe(self):
        return _original(self)(self, batch, preds)
    if preds is None:
        clear_aux_loss()  # about to forward; drop anything left from an earlier step
    total, items = _original(self)(self, batch, preds)
    aux = collect_aux_loss(self, device=total.device).to(total.dtype)
    if not torch.isfinite(aux):
        aux = torch.zeros_like(aux)
    if (getattr(self, "_esmoe_recipe", None) or _RECIPE) == "upstream":
        # Upstream adds the term to the native loss as it comes, a box/cls/dfl vector already scaled
        # by the batch, so every entry carries it once. Outside training it adds nothing.
        aux = upstream.normalise(self, aux, weight) if self.training else torch.zeros_like(aux)
        total = total + aux
    else:
        # Task criteria scale the optimised loss by batch size but log the per-image value, so the
        # aux term follows both conventions rather than showing up 'batch' times too large.
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
        core = _core(_unwrap(trainer.model))
        core._esmoe_aux_weight, core._esmoe_recipe = float(weight), recipe
        if recipe == "upstream" and getattr(trainer, "_esmoe_routers", None) is None:
            raise RuntimeError(
                "recipe 'upstream' sets the router learning rate in build_optimizer, which only the trainer "
                "attach_aux_loss routes model.train() to overrides; this trainer never split the routers"
            )
        # Older releases fix the loss names before training and need the extra one appended; newer
        # ones derive them from the loss dict, and appending to the empty tuple would leave the
        # progress header claiming the run has a single loss term.
        names = tuple(getattr(trainer, "loss_names", ()) or ())
        if names and AUX_NAME not in names:
            trainer.loss_names = (*names, AUX_NAME)

    return on_train_start


def warm_experts(trainer) -> None:
    held = getattr(trainer, "_esmoe_experts", None)
    if held is None:
        # Collected once, from what the trainer left trainable, so a layer the run froze stays frozen.
        held = trainer._esmoe_experts = upstream.experts(_unwrap(trainer.model))
    upstream.warm(held, trainer.epoch)
