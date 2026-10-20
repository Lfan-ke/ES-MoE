"""ES-MoE: an expert-sparse mixture-of-experts block, for a torch model or an Ultralytics YOLO.

Backends implement the block, integrations wire it into a training stack built on one. The torch
backend is the default and lives here at the top; the rest answer to their own names.
"""

from importlib import import_module
from types import ModuleType

from torch import nn

from .core import *
from .torch import *
from .torch.graft import graft as __into_model

__version__ = "2.2.1"

BACKENDS = {"torch": "torch", "tensorflow": "tensorflow", "jax": "jax", "flax": "flax", "keras": "keras"}
INTEGRATIONS = {"yolo": "yolo", "ultralytics": "yolo"}

# Reached through __getattr__ rather than imported, so a torch-only install never loads them.
__LAZY = {"attach_aux_loss": "yolo", "equip": "yolo", "inject_esmoe": "yolo", "register": "yolo"}


def backend(name: str = "torch") -> ModuleType:
    """The package implementing the block on ``name``."""
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; expected one of {sorted(BACKENDS)}")
    return import_module(f".{BACKENDS[name]}", __name__)


def integration(name: str) -> ModuleType:
    """The package wiring the block into ``name``."""
    if name not in INTEGRATIONS:
        raise ValueError(f"unknown integration {name!r}; expected one of {sorted(INTEGRATIONS)}")
    return import_module(f".{INTEGRATIONS[name]}", __name__)


def graft(target, *args, using: str = "yolo", **kwargs):
    """Insert blocks: a model goes to its backend, a config to the integration that reads it."""
    if isinstance(target, nn.Module):
        return __into_model(target, *args, **kwargs)
    return integration(using).graft(target, *args, **kwargs)


def __getattr__(name: str) -> object:
    if name in __LAZY:
        return getattr(import_module(f".{__LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALIASES",
    "ARGUMENTS",
    "BACKENDS",
    "BALANCES",
    "EXPERTS",
    "INTEGRATIONS",
    "BalanceFn",
    "DWExpert",
    "ESMoE",
    "ExpertFactory",
    "Kernel",
    "Plan",
    "Routing",
    "SETTINGS",
    "backend",
    "blocks",
    "clear_aux_loss",
    "collect_aux_loss",
    "fit",
    "graft",
    "integration",
    "odd",
    "odd_kernels",
    "routing",
    "settle",
    "gshard_balance",
    "gshard_probs_balance",
    "master_balance",
    "switch_balance",
]
