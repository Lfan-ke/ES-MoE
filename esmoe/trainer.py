"""DDP workers are fresh interpreters that import nothing but the trainer's own module.

Ultralytics launches them from a generated file that does ``from <trainer module> import
<TrainerClass>``, so a trainer class that lives here is the one hook through which a worker can
learn about ESMoE at all. Importing this module registers the block; the subclass restores the
auxiliary-loss weight the parent process chose.
"""

import importlib
import os

from . import inject

ENV_WEIGHT = "ESMOE_AUX_WEIGHT"
ENV_BASE = "ESMOE_TRAINER_BASE"

inject.inject_esmoe()
if weight := float(os.environ.get(ENV_WEIGHT, 0) or 0):
    inject.arm_process(weight)


def wrap(base: type) -> type:
    """A subclass of ``base`` that lives in this module, named ``ESMoE<Base>``."""
    name = f"ESMoE{base.__name__}"
    if (existing := globals().get(name)) is not None:
        return existing
    cls = type(name, (base,), {"__module__": __name__, "__init__": _init_for(base), "__doc__": base.__doc__})
    globals()[name] = cls
    # The worker resolves the same name through __getattr__ and needs to know which base to wrap.
    os.environ[ENV_BASE] = f"{base.__module__}:{base.__qualname__}"
    return cls


def _init_for(base: type):
    def __init__(self, *args, **kwargs):
        base.__init__(self, *args, **kwargs)
        weight = float(os.environ.get(ENV_WEIGHT, 0) or inject.weight() or 0)
        if weight:
            inject.arm_process(weight)
            self.add_callback("on_train_start", inject.arm_trainer(weight))

    return __init__


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
