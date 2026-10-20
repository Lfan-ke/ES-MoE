"""The Keras layer used the way each backend's own users write code, not only through Keras."""

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
keras = pytest.importorskip("keras")

from esmoe.keras import ESMoE, collect_aux_loss

BACKEND = keras.backend.backend()
SHAPE = (2, 16, 16, 4) if keras.config.image_data_format() == "channels_last" else (2, 4, 16, 16)


def block() -> keras.Sequential:
    net = keras.Sequential([ESMoE(kernels=[3, 5, 7], top_k=1)])
    net.build((None, *SHAPE[1:]))
    return net


def sample() -> np.ndarray:
    return np.random.rand(*SHAPE).astype("float32")


@pytest.mark.skipif(BACKEND != "torch", reason="needs the torch backend")
def test_torch_autograd_runs_through_the_layer():
    import torch

    net = block()
    out = net(torch.as_tensor(sample()))
    (out.sum() + collect_aux_loss(net)).backward()
    assert any(variable.value.grad is not None for variable in net.trainable_variables)


@pytest.mark.skipif(BACKEND != "tensorflow", reason="needs the tensorflow backend")
def test_tensorflow_traces_and_differentiates_the_layer():
    import tensorflow as tf

    net = block()
    traced = tf.function(lambda x: net(x))
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(traced(tf.constant(sample()))) + collect_aux_loss(net)
    grads = tape.gradient(total, net.trainable_variables)
    assert grads and all(g is not None for g in grads)


@pytest.mark.skipif(BACKEND != "jax", reason="needs the jax backend")
def test_jax_jits_and_differentiates_the_layer():
    import jax

    net = block()
    layer = net.layers[0]
    trainable = [v.value for v in layer.trainable_variables]
    other = [v.value for v in layer.non_trainable_variables]

    def total(weights, x):
        out, _ = layer.stateless_call(weights, other, x)
        return out.sum()

    x = sample()
    assert float(jax.jit(total)(trainable, x)) == pytest.approx(float(total(trainable, x)), rel=1e-4)
    grads = jax.grad(total)(trainable, x)
    assert any(float(abs(g).sum()) > 0 for g in grads)
