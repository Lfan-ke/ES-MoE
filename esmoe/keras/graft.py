"""Putting layers into a model that already exists."""

from collections.abc import Callable, Iterable

import keras
from keras import layers

from .block import ESMoE

type Where = str | Iterable[str] | Callable[[str, layers.Layer], bool]

MODES = ("after", "before", "replace")


def __picked(at: Where) -> Callable[[str, layers.Layer], bool]:
    if not isinstance(at, str) and callable(at):
        return at
    wanted = {at} if isinstance(at, str) else set(at)
    return lambda name, _: name in wanted


def graft(model: keras.Sequential, at: Where, *, mode: str = "after", **block) -> keras.Sequential:
    """Rebuild ``model`` with an ESMoE layer at every layer ``at`` picks out.

    A functional model holds a graph rather than a list, so this takes a ``Sequential`` and hands
    back a new one.
    """
    if not isinstance(model, keras.Sequential):
        raise TypeError("the Keras backend grafts into a Sequential; rebuild a functional model yourself")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    picked = __picked(at)
    grafted: list[layers.Layer] = []
    for layer in model.layers:
        if not picked(layer.name, layer):
            grafted.append(layer)
            continue
        match mode:
            case "after":
                grafted += [layer, ESMoE(**block)]
            case "before":
                grafted += [ESMoE(**block), layer]
            case _:
                grafted.append(ESMoE(**block))
    if len(grafted) == len(model.layers):
        raise ValueError(f"nothing in the model answers to {at!r}")
    return keras.Sequential(grafted, name=model.name)


__all__ = [
    "Where",
    "graft",
]
