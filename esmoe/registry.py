# Aux losses live in a weak registry rather than on the module, so checkpointing, EMA and
# deepcopy never drag a non-leaf graph tensor along with the model.
import threading
import weakref

import torch

_REGISTRY = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()


def publish(module, value):
    with _LOCK:
        _REGISTRY[module] = value


def take(module):
    with _LOCK:
        return _REGISTRY.get(module)


def clear():
    with _LOCK:
        _REGISTRY.clear()


def zeros(device=None, dtype=None):
    return torch.zeros((), device=device, dtype=dtype or torch.float32)
