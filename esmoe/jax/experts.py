"""Expert branches as pure functions over a parameter tree."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from ..kernels import Kernel

type Params = dict
type ExpertFactory = tuple[Callable[..., Params], Callable[..., jnp.ndarray]]


def pair(kernel: Kernel) -> tuple[int, int]:
    return kernel if isinstance(kernel, tuple) else (kernel, kernel)


def glorot(key, shape: tuple[int, ...]) -> jnp.ndarray:
    fan_in, fan_out = shape[-2], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def dw_init(key, c1: int, c2: int, k: Kernel) -> Params:
    """Depthwise-separable expert: the default branch, and a template for custom ones."""
    height, width = pair(k)
    depth, point = jax.random.split(key)
    return {
        "depthwise": glorot(depth, (height, width, 1, c1)),
        "pointwise": glorot(point, (1, 1, c1, c2)),
        "scale": jnp.ones((c2,)),
        "offset": jnp.zeros((c2,)),
    }


def dw_apply(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    channels = x.shape[-1]
    spread = jax.lax.conv_general_dilated(
        x,
        params["depthwise"],
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=channels,
    )
    mixed = jax.lax.conv_general_dilated(
        spread,
        params["pointwise"],
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    mean = mixed.mean(axis=(0, 1, 2))
    variance = mixed.var(axis=(0, 1, 2))
    normed = (mixed - mean) * jax.lax.rsqrt(variance + 1e-3) * params["scale"] + params["offset"]
    return jax.nn.silu(normed)


DWExpert: ExpertFactory = (dw_init, dw_apply)

EXPERTS: dict[str, ExpertFactory] = {"dw": DWExpert}


__all__ = [
    "EXPERTS",
    "DWExpert",
    "ExpertFactory",
    "Params",
    "dw_apply",
    "dw_init",
    "glorot",
    "pair",
]
