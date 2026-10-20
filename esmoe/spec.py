"""What a block is asked for, before any backend builds it.

`settle` is the first half of every backend's constructor: it reads however the caller spelled the
arguments and hands back one settled plan. A backend adds only the tensor work.
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .kernels import Kernel, fit, odd_kernels

# What a config can carry. A trainer rebuilds the model from its config, so a setting applied to
# the instance afterwards is discarded; upstream's own defaults are noted where they differ.
SETTINGS = MappingProxyType(
    {
        # Switch reads the full softmax; gshard reads the gate, which carries no gradient for an
        # expert outside the top-k and let experts die (docs/JUDGMENT, round six).
        "balance": "switch",
        "out_norm": False,  # upstream: always on
        "dense_training": False,  # upstream: always on
        "sparse_inference": True,
        "dynamic_threshold": 0.0,  # upstream: 0.4
    }
)

ARGUMENTS = ("num_experts", "top_k", "channels", "out_channels", "kernels", "expert", "reduction", "max_kernel")

# What other callers name the same thing. Anything else is ignored with a warning rather than
# refused, so a stray framework argument never costs a run.
ALIASES = {
    "c": "channels",
    "c1": "channels",
    "ch": "channels",
    "in_channels": "channels",
    "c2": "out_channels",
    "e": "num_experts",
    "n_experts": "num_experts",
    "k": "top_k",
    "topk": "top_k",
    "kernel_sizes": "kernels",
    "expert_kernel_sizes": "kernels",
    "max_kernel_size": "max_kernel",
}


@dataclass(frozen=True, slots=True)
class Plan:
    """Everything a backend needs to build one block."""

    num_experts: int
    top_k: int
    channels: int | None
    out_channels: int | None
    kernels: list[Kernel]
    expert: list[object]
    reduction: int
    max_kernel: int
    listed: bool
    settings: dict[str, object]


def __slot(key: str, value: object) -> str:
    """``experts`` stands for whichever argument its value fits; the rest is a lookup."""
    if key != "experts":
        return ALIASES.get(key, key)
    if isinstance(value, int):
        return "num_experts"
    first = value[0] if isinstance(value, Sequence) and value else None
    return "kernels" if isinstance(first, int | tuple | list) else "expert"


def __adapt(given: Mapping, known: set[str]) -> dict[str, object]:
    found = {}
    for key, value in given.items():
        if (name := __slot(key, value)) in known:
            found[name] = value
        else:
            warnings.warn(f"ESMoE ignores unknown option {key!r}", stacklevel=4)
    return found


def settle(local: Mapping, options: Mapping | None = None, settings: Mapping | None = None) -> Plan:
    """One plan out of the arguments, the config options and the loose keywords."""
    known = set(ARGUMENTS) | set(SETTINGS)
    carried = __adapt(dict(options or {}), known)
    given = carried | __adapt(dict(settings or {}), known)
    asked = dict(local) | {key: value for key, value in given.items() if key in ARGUMENTS}
    chosen = dict(SETTINGS) | {key: value for key, value in given.items() if key in SETTINGS}

    kernels, expert = asked["kernels"], asked["expert"]
    branches = list(expert) if not isinstance(expert, str) and isinstance(expert, Sequence) else None
    count = len(kernels) if kernels else len(branches) if branches else asked["num_experts"]
    if kernels and branches and len(kernels) != len(branches):
        raise ValueError(f"kernels and expert disagree on the count: {len(kernels)} against {len(branches)}")
    if not isinstance(count, int) or count < 1:
        raise ValueError(f"num_experts must be a positive integer, got {count!r}")
    if asked["reduction"] < 1:
        raise ValueError(f"reduction must be positive, got {asked['reduction']}")
    if asked["max_kernel"] < 1:
        raise ValueError(f"max_kernel must be positive, got {asked['max_kernel']}")
    threshold = chosen["dynamic_threshold"]
    if not isinstance(threshold, int | float) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"dynamic_threshold must be a number in [0, 1], got {threshold!r}")
    chosen["dynamic_threshold"] = float(threshold)
    top_k = count if asked["top_k"] is None else int(asked["top_k"])
    if top_k < 1:
        raise ValueError(f"top_k must be positive or None, got {top_k}")
    if top_k > count:
        warnings.warn(f"ESMoE has {count} experts, so top_k {top_k} activates all of them", stacklevel=3)
        top_k = count

    cap = int(asked["max_kernel"])
    return Plan(
        num_experts=count,
        top_k=top_k,
        channels=asked["channels"],
        out_channels=asked["out_channels"],
        kernels=[fit(k, cap) for k in kernels] if kernels else odd_kernels(count, cap),
        expert=branches if branches else [expert] * count,
        reduction=int(asked["reduction"]),
        max_kernel=cap,
        # A widened block hands a one-element list to the official Index layer that follows it,
        # which only a yaml asks for; set directly, out_channels just returns a wider tensor.
        listed=carried.get("out_channels") is not None,
        settings=chosen,
    )


__all__ = [
    "ALIASES",
    "ARGUMENTS",
    "SETTINGS",
    "Plan",
    "settle",
]
