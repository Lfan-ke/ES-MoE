"""Expert branches as Linen modules."""

import jax
from flax import linen

from ...core.kernels import Kernel


def pair(kernel: Kernel) -> tuple[int, int]:
    return kernel if isinstance(kernel, tuple) else (kernel, kernel)


class DWExpert(linen.Module):
    """Depthwise-separable expert: the default branch, and a template for custom ones."""

    channels: int
    out_channels: int
    kernel: Kernel

    @linen.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        spread = linen.Conv(
            self.channels, pair(self.kernel), padding="SAME", feature_group_count=self.channels, use_bias=False
        )(x)
        mixed = linen.Conv(self.out_channels, (1, 1), use_bias=False)(spread)
        return linen.silu(linen.BatchNorm(use_running_average=not training)(mixed))


EXPERTS: dict[str, type[linen.Module]] = {"dw": DWExpert}


__all__ = [
    "EXPERTS",
    "DWExpert",
    "pair",
]
