"""DDP workers are fresh interpreters that import nothing but the trainer's own module.

Ultralytics launches them from a generated file that does ``from <trainer module> import
<TrainerClass>``, so a trainer class that lives here is the one hook through which a worker can
learn about ESMoE at all. Importing this module registers the block; the subclass restores the
auxiliary-loss weight and recipe the parent process chose.
"""

import importlib
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
        weight = float(os.environ.get(ENV_WEIGHT, 0) or inject.weight() or 0)
        if weight:
            recipe = os.environ.get(ENV_RECIPE) or inject.recipe()
            inject.arm_process(weight, recipe)
            for event, callback in inject.trainer_callbacks(weight, recipe):
                self.add_callback(event, callback)

    return __init__


def _build_optimizer_for(base: type):
    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        optimizer = base.build_optimizer(self, model, name, lr, momentum, decay, iterations)
        if inject.recipe() == "upstream":
            self._esmoe_routers = upstream.split_routers(optimizer, model, decay)
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
