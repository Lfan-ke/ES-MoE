import torch

from .aux_loss import clear_aux_loss, collect_aux_loss
from .module import ESMoE

AUX_NAME = "esmoe_aux"
_PATCHED = {}
_WEIGHT = None


def inject_esmoe():
    """Expose ESMoE where ultralytics parse_model resolves layer names, so any model.yaml can
    reference ``ESMoE`` across YOLOv8 / YOLO11 / YOLOv12 backbones."""
    import ultralytics.nn.tasks as tasks

    setattr(tasks, "ESMoE", ESMoE)
    return ESMoE


def _core(model):
    # A YOLO wrapper hides the task model under .model, but a bare task model's .model is its
    # layer Sequential - pick whichever level actually owns loss().
    for candidate in (getattr(model, "model", None), model):
        if callable(getattr(candidate, "loss", None)):
            return candidate
    raise TypeError(f"{type(model).__name__} exposes no loss() to attach to")


def _unwrap(model):
    # The helper was renamed across ultralytics releases; both names mean "drop DDP/EMA wrappers".
    import ultralytics.utils.torch_utils as tu

    fn = getattr(tu, "unwrap_model", None) or getattr(tu, "de_parallel", None)
    return fn(model) if fn else model


def _uses_esmoe(model):
    present = getattr(model, "_esmoe_present", None)
    if present is None:
        present = any(isinstance(m, ESMoE) for m in model.modules())
        model._esmoe_present = present
    return present


def _loss_with_aux(self, batch, preds=None):
    # The weight also lives at process scope because the trainer rebuilds the model and the EMA
    # copy is taken before any callback runs; an instance-only flag makes those copies log a
    # differently shaped loss than the trainer expects.
    weight = getattr(self, "_esmoe_aux_weight", None)
    weight = _WEIGHT if weight is None else weight
    if not weight or not _uses_esmoe(self):
        return _PATCHED[type(self)](self, batch, preds)
    if preds is None:
        clear_aux_loss()  # we are about to forward; drop anything left from an earlier step
    total, items = _PATCHED[type(self)](self, batch, preds)
    aux = collect_aux_loss(self, device=total.device).to(total.dtype)
    if not torch.isfinite(aux):
        aux = torch.zeros_like(aux)
    # Detection criteria scale the optimised loss by batch size but log the per-image value, so
    # the aux term follows both conventions rather than showing up 'batch' times too large.
    aux = (aux * weight).view(1)
    total = torch.cat([total.reshape(-1), aux * batch["img"].shape[0]])
    return total, torch.cat([items, aux.detach().to(items.dtype)])


def attach_aux_loss(model, weight=0.01):
    """Make the router load-balancing loss part of the optimised training loss.

    Without this the aux term exists but never reaches ``backward``, which is the exact failure
    the task book's red line asks to rule out.
    """
    core = _core(model)
    if not any(isinstance(m, ESMoE) for m in core.modules()):
        raise ValueError("model contains no ESMoE block; nothing to attach")
    cls = type(core)
    if cls not in _PATCHED:
        _PATCHED[cls] = cls.loss
        cls.loss = _loss_with_aux
    global _WEIGHT
    _WEIGHT = float(weight)
    core._esmoe_aux_weight = float(weight)
    if hasattr(model, "add_callback"):
        # The trainer rebuilds the model from yaml, so the weight has to be re-attached to the
        # instance it actually trains, not only to the one handed to us here.
        model.add_callback("on_train_start", _arm(weight))
    return model


def _arm(weight):
    def on_train_start(trainer):
        _core(_unwrap(trainer.model))._esmoe_aux_weight = float(weight)
        if AUX_NAME not in trainer.loss_names:
            trainer.loss_names = (*trainer.loss_names, AUX_NAME)

    return on_train_start
