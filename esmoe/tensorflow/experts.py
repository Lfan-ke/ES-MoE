"""Expert branches, built on tf.Module so the backend owes nothing to Keras."""

from collections.abc import Callable

import tensorflow as tf

from ..core.kernels import Kernel

type ExpertFactory = Callable[[int, int, Kernel], tf.Module]

GLOROT = tf.initializers.GlorotUniform()


def pair(kernel: Kernel) -> tuple[int, int]:
    return kernel if isinstance(kernel, tuple) else (kernel, kernel)


class BatchNorm(tf.Module):
    """Batch normalisation over NHWC, with the moving statistics kept as untrained variables."""

    def __init__(self, width: int, momentum: float = 0.9, epsilon: float = 1e-3):
        super().__init__()
        self.momentum, self.epsilon = momentum, epsilon
        self.scale = tf.Variable(tf.ones([width]), name="scale")
        self.offset = tf.Variable(tf.zeros([width]), name="offset")
        self.mean = tf.Variable(tf.zeros([width]), trainable=False, name="mean")
        self.variance = tf.Variable(tf.ones([width]), trainable=False, name="variance")

    def __call__(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        mean, variance = tf.nn.moments(x, axes=[0, 1, 2]) if training else (self.mean, self.variance)
        if training:
            self.mean.assign(self.momentum * self.mean + (1.0 - self.momentum) * mean)
            self.variance.assign(self.momentum * self.variance + (1.0 - self.momentum) * variance)
        return tf.nn.batch_normalization(x, mean, variance, self.offset, self.scale, self.epsilon)


class DWExpert(tf.Module):
    """Depthwise-separable expert: the default branch, and a template for custom ones."""

    def __init__(self, c1: int, c2: int, k: Kernel):
        super().__init__()
        height, width = pair(k)
        self.depthwise = tf.Variable(GLOROT([height, width, c1, 1]), name="depthwise")
        self.pointwise = tf.Variable(GLOROT([1, 1, c1, c2]), name="pointwise")
        self.norm = BatchNorm(c2)

    def __call__(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        spread = tf.nn.depthwise_conv2d(x, self.depthwise, strides=[1, 1, 1, 1], padding="SAME")
        mixed = tf.nn.conv2d(spread, self.pointwise, strides=1, padding="SAME")
        return tf.nn.silu(self.norm(mixed, training))


EXPERTS: dict[str, ExpertFactory] = {"dw": DWExpert}


__all__ = [
    "EXPERTS",
    "BatchNorm",
    "DWExpert",
    "ExpertFactory",
    "pair",
]
