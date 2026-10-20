"""Where a block leaves its balancing term for the loss to pick up.

Not on the module itself: a non-leaf graph tensor there would follow the model through deepcopy,
EMA and checkpointing.
"""

import threading
import weakref

__REGISTRY: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
__LOCK = threading.Lock()


def publish(module: object, value: object) -> None:
    with __LOCK:
        __REGISTRY[module] = value


def take(module: object) -> object | None:
    """What this module published on its last forward pass, or None.

    Reading does not remove: one step may compute the loss twice and each needs the term.
    """
    with __LOCK:
        return __REGISTRY.get(module)


def clear() -> None:
    with __LOCK:
        __REGISTRY.clear()


__all__ = [
    "clear",
    "publish",
    "take",
]
