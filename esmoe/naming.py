"""How a config names a callable, and how that name leads back to it."""

import importlib
from collections.abc import Callable, Mapping


def reference(obj: Callable, shipped: Mapping[str, Callable]) -> str:
    """A shipped callable by its short name, any other by ``module:qualname``.

    A config holds text, and the trainer rebuilds the model from that text in this process and in
    every DDP worker, so a lambda, a nested function or anything defined in ``__main__`` is refused
    here rather than lost without a trace later.
    """
    for name, known in shipped.items():
        if known is obj:
            return name
    module, qualname = getattr(obj, "__module__", None), getattr(obj, "__qualname__", "")
    if not module or module == "__main__" or "<" in qualname:
        raise ValueError(
            f"{qualname or obj!r} cannot be named in a config; define it at module level in an importable "
            "module, because the trainer and every DDP worker rebuild the model from that name"
        )
    name = f"{module}:{qualname}"
    if resolve(name, shipped) is not obj:
        raise ValueError(f"{name} does not import back to the object it names")
    return name


def resolve(name: str, shipped: Mapping[str, Callable]) -> Callable:
    """The callable a config names: a key of ``shipped``, or ``module:qualname``."""
    if name in shipped:
        return shipped[name]
    module, colon, qualname = name.partition(":")
    if not colon:
        raise ValueError(f"unknown name {name!r}; use one of {sorted(shipped)} or 'module:qualname'")
    target = importlib.import_module(module)
    for part in qualname.split(".") if qualname else ():
        target = getattr(target, part)
    if not callable(target):
        raise ValueError(f"{name} names {type(target).__name__}, not a callable")
    return target


def named(obj: Callable, shipped: Mapping[str, Callable]) -> str:
    """`reference`, falling back to a readable repr for a callable no config has to carry."""
    try:
        return reference(obj, shipped)
    except ValueError:
        return getattr(obj, "__qualname__", repr(obj))


__all__ = [
    "named",
    "reference",
    "resolve",
]
