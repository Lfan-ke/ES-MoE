"""Putting blocks into a model that already exists."""

from collections.abc import (
    Callable,
    Iterable,
)

import tensorflow as tf

from .block import ESMoE

type Where = str | Iterable[str] | Callable[[str, tf.Module], bool]

MODES = ("after", "before", "replace")


class Sequential(tf.Module):
    """A list of callables, called in order: the smallest container a graft can edit."""

    def __init__(self, layers: Iterable[Callable], name: str | None = None):
        super().__init__(name=name)
        self.layers = list(layers)

    def __call__(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        for layer in self.layers:
            x = layer(x, training) if isinstance(layer, tf.Module) else layer(x)
        return x


def __picked(at: Where) -> Callable[[str, tf.Module], bool]:
    if not isinstance(at, str) and callable(at):
        return at
    wanted = {at} if isinstance(at, str) else set(at)
    return lambda name, _: name in wanted


def graft(model: tf.Module, at: Where, *, mode: str = "after", **block) -> tf.Module:
    """Put an ESMoE block at every submodule ``at`` picks out, in place.

    A `Sequential` is addressed by position (``"0"``, ``"1"``); any other module by attribute name.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    picked, grafted = __picked(at), 0

    def place(found: tf.Module) -> tf.Module:
        made = ESMoE(**block)
        return made if mode == "replace" else Sequential([found, made] if mode == "after" else [made, found])

    def walk(owner: tf.Module, prefix: str) -> None:
        nonlocal grafted
        if isinstance(owner, Sequential):
            for index, layer in enumerate(list(owner.layers)):
                name = f"{prefix}{index}"
                if isinstance(layer, tf.Module) and picked(name, layer):
                    owner.layers[index] = place(layer)
                    grafted += 1
                elif isinstance(layer, tf.Module):
                    walk(layer, f"{name}.")
            return
        for key, value in list(vars(owner).items()):
            if not isinstance(value, tf.Module):
                continue
            name = f"{prefix}{key}"
            if picked(name, value):
                setattr(owner, key, place(value))
                grafted += 1
            else:
                walk(value, f"{name}.")

    walk(model, "")
    if not grafted:
        raise ValueError(f"nothing in the model answers to {at!r}")
    return model


__all__ = [
    "Sequential",
    "Where",
    "graft",
]
