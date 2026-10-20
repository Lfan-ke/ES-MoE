"""Expert branches as `flax.nnx` modules."""

from collections.abc import Callable

import jax
from flax import nnx

from ...core.kernels import Kernel

type ExpertFactory = Callable[..., nnx.Module]


def pair(kernel: Kernel) -> tuple[int, int]:
    return kernel if isinstance(kernel, tuple) else (kernel, kernel)


class DWExpert(nnx.Module):
    """Depthwise-separable expert: the default branch, and a template for custom ones."""

    def __init__(self, c1: int, c2: int, k: Kernel, *, rngs: nnx.Rngs):
        self.depthwise = nnx.Conv(c1, c1, pair(k), padding="SAME", feature_group_count=c1, use_bias=False, rngs=rngs)
        self.pointwise = nnx.Conv(c1, c2, (1, 1), use_bias=False, rngs=rngs)
        self.norm = nnx.BatchNorm(c2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.silu(self.norm(self.pointwise(self.depthwise(x))))


EXPERTS: dict[str, ExpertFactory] = {"dw": DWExpert}


__all__ = [
    "EXPERTS",
    "DWExpert",
    "ExpertFactory",
    "pair",
]
