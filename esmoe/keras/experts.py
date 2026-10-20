"""Expert branches."""

from collections.abc import Callable

from keras import layers

from ..core.kernels import Kernel

type ExpertFactory = Callable[[int, int, Kernel], layers.Layer]


class DWExpert(layers.Layer):
    """Depthwise-separable expert: the default branch, and a template for custom ones."""

    def __init__(self, c1: int, c2: int, k: Kernel, **kwargs):
        super().__init__(**kwargs)
        self.dw = layers.DepthwiseConv2D(k, padding="same", use_bias=False)
        self.pw = layers.Conv2D(c2, 1, use_bias=False)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation("silu")

    def call(self, x, training=False):
        return self.act(self.bn(self.pw(self.dw(x)), training=training))


EXPERTS: dict[str, ExpertFactory] = {"dw": DWExpert}


__all__ = [
    "EXPERTS",
    "DWExpert",
    "ExpertFactory",
]
