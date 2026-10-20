"""Putting blocks into a model that already exists."""

from collections.abc import Callable, Iterable

from torch import nn

from .block import ESMoE

type Where = str | Iterable[str] | Callable[[str, nn.Module], bool]

MODES = ("after", "before", "replace")


def __targets(model: nn.Module, at: Where) -> list[tuple[str, nn.Module]]:
    if not isinstance(at, str) and callable(at):
        return [(name, module) for name, module in model.named_modules() if name and at(name, module)]
    wanted = {at} if isinstance(at, str) else set(at)
    found = [(name, module) for name, module in model.named_modules() if name in wanted]
    if missing := wanted - {name for name, _ in found}:
        raise ValueError(f"no submodule named {sorted(missing)}")
    return found


def graft(model: nn.Module, at: Where, *, mode: str = "after", **block) -> nn.Module:
    """Put an ESMoE block at every submodule ``at`` picks out, in place.

    ``at`` is a submodule name, several names, or a ``(name, module) -> bool`` test. ``mode`` puts
    the block ``after`` or ``before`` what it found, or ``replace``s it. The rest is handed to
    `ESMoE`, so the block sizes itself on the first forward pass.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    for name, found in __targets(model, at):
        parent, _, leaf = name.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        made = ESMoE(**block)
        match mode:
            case "after":
                grafted: nn.Module = nn.Sequential(found, made)
            case "before":
                grafted = nn.Sequential(made, found)
            case _:
                grafted = made
        setattr(owner, leaf, grafted)
    return model


__all__ = [
    "Where",
    "graft",
]
