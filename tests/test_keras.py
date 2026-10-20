"""The Keras backend, on whichever of tensorflow, jax or torch Keras is configured for."""

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
keras = pytest.importorskip("keras")

import esmoe
from esmoe.keras import ESMoE, blocks, collect_aux_loss, graft

SHAPE = (2, 16, 16, 3) if keras.config.image_data_format() == "channels_last" else (2, 3, 16, 16)


def stack(*layers) -> keras.Sequential:
    return keras.Sequential(list(layers))


def test_the_backend_table_points_here():
    assert esmoe.backend("keras").ESMoE is ESMoE
    assert esmoe.backend("jax").ESMoE is ESMoE
    assert esmoe.backend("tensorflow").ESMoE is ESMoE


def test_it_drops_into_a_sequential():
    net = stack(keras.layers.Conv2D(8, 3, padding="same"), ESMoE(kernels=[3, 5, 7], top_k=1))
    out = net(np.random.rand(*SHAPE).astype("float32"))
    assert tuple(out.shape) == (2, *SHAPE[1:-1], 8) if SHAPE[-1] == 3 else out.shape[1] == 8


def test_the_arguments_are_the_torch_backend_s():
    block = ESMoE(kernels=[(3, 3), (1, 9)], top_k=1)
    assert block.kernels == [(3, 3), (1, 9)]
    assert block.num_experts == 2
    assert block.spec()["balance"] == "switch"


def test_the_balancing_term_reaches_the_model_losses():
    net = stack(keras.layers.Conv2D(8, 3, padding="same"), ESMoE(kernels=[3, 5]))
    net(np.random.rand(*SHAPE).astype("float32"))
    assert float(collect_aux_loss(net)) > 0
    assert len(list(blocks(net))) == 1


def test_it_trains_through_fit():
    net = stack(keras.layers.Conv2D(4, 3, padding="same"), ESMoE(kernels=[3, 5]), keras.layers.GlobalAveragePooling2D())
    net.compile(optimizer="adam", loss="mse")
    history = net.fit(np.random.rand(*SHAPE).astype("float32"), np.random.rand(2, 4).astype("float32"), verbose=0)
    assert np.isfinite(history.history["loss"][0])


def test_graft_rebuilds_a_sequential():
    net = graft(stack(keras.layers.Conv2D(8, 3, padding="same", name="stem")), at="stem", kernels=[3, 5])
    assert len(list(blocks(net))) == 1
    net(np.random.rand(*SHAPE).astype("float32"))


def test_graft_refuses_a_functional_model():
    inputs = keras.Input(shape=SHAPE[1:])
    model = keras.Model(inputs, keras.layers.Conv2D(4, 1, name="head")(inputs))
    with pytest.raises(TypeError, match="Sequential"):
        graft(model, at="head")
