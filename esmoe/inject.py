"""Runtime injection: make ESMoE resolvable in configs and make its loss reach the optimiser."""

import os
from collections.abc import Callable

import torch
from torch import nn

from .aux_loss import clear_aux_loss, collect_aux_loss
from .module import ESMoE, blocks

AUX_NAME = "esmoe_aux"
ENV_WEIGHT = "ESMOE_AUX_WEIGHT"
_PATCHED: dict[type, Callable] = {}
_WEIGHT: float | None = None


def inject_esmoe() -> type[ESMoE]:
    """Expose ESMoE where ultralytics parse_model resolves layer names, so any model.yaml can
    reference ``ESMoE`` across YOLOv8 / YOLO11 / YOLO12 and every task."""
    import ultralytics.nn.tasks as tasks

    tasks.ESMoE = ESMoE
    return ESMoE


def equip(base: str = "yolov8n.yaml", *, weight: float = 0.01, out: str | None = None, **graft_kwargs):
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
    return attach_aux_loss(YOLO(str(target)), weight=weight)


def attach_aux_loss(model, weight: float = 0.01):
    """Make the router load-balancing loss part of the optimised training loss.

    Without this the aux term exists but never reaches ``backward``, which is the exact failure the
    task book's red line asks to rule out.
    """
    core = _core(model)
    if next(blocks(core), None) is None:
        raise ValueError("model contains no ESMoE block; nothing to attach")
    arm_process(weight)
    _patch(_owner(type(core)))
    core._esmoe_aux_weight = float(weight)
    if hasattr(model, "add_callback"):
        # The trainer rebuilds the model from yaml, so the weight has to reach the instance it
        # actually trains, not only the one handed to us here.
        model.add_callback("on_train_start", arm_trainer(weight))
    if hasattr(model, "_smart_load"):
        _route_trainer(model)
    return model


def arm_process(weight: float) -> None:
    """Remember the weight for every model this process builds and patch the shared loss entry.

    The trainer rebuilds the model and takes the EMA copy before any callback runs, and a DDP worker
    never sees the instance the caller attached to; process scope is what both of them read.
    """
    global _WEIGHT
    _WEIGHT = float(weight)
    os.environ[ENV_WEIGHT] = repr(_WEIGHT)
    import ultralytics.nn.tasks as tasks

    _patch(tasks.BaseModel)


def weight() -> float | None:
    return _WEIGHT


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
    # The weight also lives at process scope because the trainer rebuilds the model and the EMA copy
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
    # Task criteria scale the optimised loss by batch size but log the per-image value, so the aux
    # term follows both conventions rather than showing up 'batch' times too large.
    aux = (aux * weight).view(1)
    scaled = aux * batch["img"].shape[0]
    total = total + scaled.squeeze() if total.ndim == 0 else torch.cat([total.reshape(-1), scaled])
    # ultralytics >= 8.4.13x reports loss items as a named dict; older releases return a tensor.
    if isinstance(items, dict):
        return total, items | {AUX_NAME: aux.detach().squeeze()}
    return total, torch.cat([items, aux.detach().to(items.dtype)])


def arm_trainer(weight: float) -> Callable:
    def on_train_start(trainer) -> None:
        _core(_unwrap(trainer.model))._esmoe_aux_weight = float(weight)
        # Older releases fix the loss names before training and need the extra one appended; newer
        # ones derive them from the loss dict, and appending to the empty tuple would leave the
        # progress header claiming the run has a single loss term.
        names = tuple(getattr(trainer, "loss_names", ()) or ())
        if names and AUX_NAME not in names:
            trainer.loss_names = (*names, AUX_NAME)

    return on_train_start
