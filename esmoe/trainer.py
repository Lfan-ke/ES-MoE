"""DDP workers are fresh interpreters that import nothing but the trainer's own module.

Ultralytics launches them from a generated file that does ``from <trainer module> import
<TrainerClass>``, so a trainer class that lives here is the one hook through which a worker can
learn about ESMoE at all. Importing this module registers the block; the subclass restores the
auxiliary-loss weight and recipe the parent process chose.
"""

import importlib
import inspect
import os

from . import inject, upstream

ENV_WEIGHT = inject.ENV_WEIGHT
ENV_RECIPE = inject.ENV_RECIPE
ENV_BASE = "ESMOE_TRAINER_BASE"

inject.inject_esmoe()
if weight := float(os.environ.get(ENV_WEIGHT, 0) or 0):
    inject.arm_process(weight, os.environ.get(ENV_RECIPE) or "esmoe")


def wrap(base: type) -> type:
    """A subclass of ``base`` that lives in this module, named ``ESMoE<Base>``."""
    name = f"ESMoE{base.__name__}"
    if (existing := globals().get(name)) is not None:
        return existing
    members = {
        "__module__": __name__,
        "__init__": _init_for(base),
        "build_optimizer": _build_optimizer_for(base),
        "__doc__": base.__doc__,
    }
    cls = type(name, (base,), members)
    globals()[name] = cls
    # The worker resolves the same name through __getattr__ and needs to know which base to wrap.
    os.environ[ENV_BASE] = f"{base.__module__}:{base.__qualname__}"
    return cls


def _init_for(base: type):
    def __init__(self, *args, **kwargs):
        base.__init__(self, *args, **kwargs)
        weight = float(os.environ.get(ENV_WEIGHT, 0) or inject.armed_weight() or 0)
        if weight:
            recipe = os.environ.get(ENV_RECIPE) or inject.armed_recipe()
            inject.arm_process(weight, recipe)
            for event, callback in inject.trainer_callbacks(weight, recipe):
                self.add_callback(event, callback)

    return __init__


def _build_optimizer_for(base: type):
    """Wrap the base trainer's ``build_optimizer`` whatever arguments it takes.

    The official trainer and YOLO-Master's differ here, and both keep adding keywords (the fork
    now passes ``router_decay``). Forwarding the call untouched and reading `model` and `decay`
    off the base's own signature keeps the wrapper working across those changes.
    """
    signature = inspect.signature(base.build_optimizer)

    def build_optimizer(self, *args, **kwargs):
        optimizer = base.build_optimizer(self, *args, **kwargs)
        if inject.armed_recipe() == "upstream":
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            model = bound.arguments.get("model", args[0] if args else None)
            self._esmoe_routers = upstream.split_routers(optimizer, model, bound.arguments.get("decay", 1e-5))
        return optimizer

    return build_optimizer


def __getattr__(name: str) -> type:
    if not name.startswith("ESMoE"):
        raise AttributeError(name)
    spec = os.environ.get(ENV_BASE)
    if not spec:
        raise AttributeError(f"{name}: set {ENV_BASE}=module:Class or call esmoe.trainer.wrap(...) first")
    module, _, qualname = spec.partition(":")
    base = importlib.import_module(module)
    for part in qualname.split("."):
        base = getattr(base, part)
    if f"ESMoE{base.__name__}" != name:
        raise AttributeError(f"{name} does not match {ENV_BASE}={spec}")
    return wrap(base)
