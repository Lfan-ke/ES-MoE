"""Expert branches as pure functions over a parameter tree."""

import jax
import jax.numpy as jnp

from ..kernels import Kernel

type Params = dict


def pair(kernel: Kernel) -> tuple[int, int]:
    return kernel if isinstance(kernel, tuple) else (kernel, kernel)


def glorot(key, shape: tuple[int, ...]) -> jnp.ndarray:
    fan_in, fan_out = shape[-2], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


class DWExpert:
    """Depthwise-separable expert: the default branch, and a template for custom ones.

    An expert is a pair of pure functions, `init` and `apply`, carried by a class so that a config
    can name it and the shared core can tell one factory from a list of them.
    """

    @staticmethod
    def init(key, c1: int, c2: int, k: Kernel) -> Params:
        height, width = pair(k)
        depth, point = jax.random.split(key)
        return {
            "depthwise": glorot(depth, (height, width, 1, c1)),
            "pointwise": glorot(point, (1, 1, c1, c2)),
            "scale": jnp.ones((c2,)),
            "offset": jnp.zeros((c2,)),
        }

    @staticmethod
    def apply(params: Params, x: jnp.ndarray) -> jnp.ndarray:
        spread = jax.lax.conv_general_dilated(
            x,
            params["depthwise"],
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=x.shape[-1],
        )
        mixed = jax.lax.conv_general_dilated(
            spread,
            params["pointwise"],
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        centred = mixed - mixed.mean(axis=(0, 1, 2))
        normed = centred * jax.lax.rsqrt(mixed.var(axis=(0, 1, 2)) + 1e-3) * params["scale"] + params["offset"]
        return jax.nn.silu(normed)


type ExpertFactory = type[DWExpert]

EXPERTS: dict[str, ExpertFactory] = {"dw": DWExpert}


__all__ = [
    "EXPERTS",
    "DWExpert",
    "ExpertFactory",
    "Params",
    "glorot",
    "pair",
]
