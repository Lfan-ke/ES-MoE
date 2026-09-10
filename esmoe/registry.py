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
    """The value this module published on its last forward, or None.

    Reading does not remove, which is what upstream's registry does too: a step may compute the
    loss more than once and each computation needs the term. Upstream rejects a value from an
    earlier step by stamping it; here the loss patch clears the registry before each forward,
    which covers the same case as long as every block runs in that forward.
    """
    with _LOCK:
        return _REGISTRY.get(module)


def clear():
    with _LOCK:
        _REGISTRY.clear()


def zeros(device=None, dtype=None):
    return torch.zeros((), device=device, dtype=dtype or torch.float32)
