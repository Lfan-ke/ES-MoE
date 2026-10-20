"""The native TensorFlow backend: tf.Module all the way down, no Keras anywhere."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

import esmoe
from esmoe.tensorflow import (
    ESMoE,
    Sequential,
    blocks,
    collect_aux_loss,
    graft,
    routing,
)

SHAPE = (2, 16, 16, 8)


def sample() -> tf.Tensor:
    return tf.constant(np.random.rand(*SHAPE).astype("float32"))


def test_the_backend_table_points_here():
    assert esmoe.backend("tensorflow").ESMoE is ESMoE


def test_it_owes_nothing_to_keras():
    import esmoe.tensorflow.block as block

    assert all("keras" not in str(value) for value in vars(block).values() if isinstance(value, str))
    assert ESMoE.__mro__[1] is tf.Module


def test_kernels_decide_the_experts():
    held = ESMoE(kernels=[(3, 3), (1, 9)], top_k=1)
    assert (held.num_experts, held.top_k) == (2, 1)
    assert list(held.kernels) == [(3, 3), (1, 9)]


def test_forward_keeps_the_channels():
    net = ESMoE(kernels=[3, 5, 7], top_k=1)
    assert tuple(net(sample()).shape) == SHAPE


def test_the_balancing_term_reaches_the_gradients():
    net = Sequential([ESMoE(kernels=[3, 5])])
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(net(sample(), training=True)) + collect_aux_loss(net)
    grads = tape.gradient(total, net.trainable_variables)
    assert len(net.trainable_variables) > 0
    assert all(g is not None for g in grads)


def test_routing_forgets_the_step_before():
    net = Sequential([ESMoE(kernels=[3, 5])])
    net(sample())
    with routing(net) as route:
        assert float(route.loss()) == 0.0


def test_graft_puts_blocks_into_a_model_that_exists():
    class Tiny(tf.Module):
        def __init__(self):
            super().__init__()
            self.stem = Sequential([lambda x, training=False: x])

        def __call__(self, x, training=False):
            return self.stem(x, training)

    model = Tiny()
    graft(model, at="stem", kernels=[3, 5])
    assert len(list(blocks(model))) == 1
    assert tuple(model(sample()).shape) == SHAPE


def test_an_unknown_argument_warns_and_is_ignored():
    with pytest.warns(UserWarning, match="nonsense"):
        assert ESMoE(kernels=[3, 5], nonsense=True).num_experts == 2
