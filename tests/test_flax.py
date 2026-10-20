"""The Flax backend, written against flax.nnx."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")

import esmoe
from esmoe.flax import (
    ESMoE,
    blocks,
    collect_aux_loss,
    routing,
)

SHAPE = (2, 16, 16, 8)


def sample():
    return jax.numpy.asarray(np.random.rand(*SHAPE).astype("float32"))


def block(**held):
    return ESMoE(channels=SHAPE[-1], rngs=nnx.Rngs(0), **held)


def test_the_backend_table_points_here():
    assert esmoe.backend("flax").ESMoE is ESMoE


def test_it_owes_nothing_to_keras():
    import pathlib

    import esmoe.flax

    source = "".join(p.read_text(encoding="utf-8") for p in pathlib.Path(esmoe.flax.__path__[0]).glob("*.py"))
    assert "keras" not in source


def test_kernels_decide_the_experts():
    held = block(kernels=[(3, 3), (1, 9)], top_k=1)
    assert (held.num_experts, held.top_k) == (2, 1)
    assert list(held.kernels) == [(3, 3), (1, 9)]
    assert len(held.experts) == 2


def test_forward_keeps_the_channels():
    assert tuple(block(kernels=[3, 5, 7], top_k=1)(sample()).shape) == SHAPE


def test_channels_are_required():
    with pytest.raises(ValueError, match="channels is required"):
        ESMoE(kernels=[3, 5], rngs=nnx.Rngs(0))


def test_routing_reads_this_step_only():
    held = block(kernels=[3, 5])
    held(sample())
    with routing(held) as route:
        assert float(route.loss()) == 0.0
        held(sample())
        assert float(route.loss()) > 0


def test_it_jits_and_differentiates():
    model = nnx.Sequential(nnx.Conv(3, 8, (3, 3), rngs=nnx.Rngs(0)), block(kernels=[3, 5]))
    x = jax.numpy.asarray(np.random.rand(2, 16, 16, 3).astype("float32"))

    @nnx.jit
    def step(net, inputs):
        def objective(net):
            return net(inputs).sum() + 0.01 * collect_aux_loss(net)

        return nnx.grad(objective)(net)

    leaves = jax.tree.leaves(step(model, x))
    assert leaves and any(float(abs(leaf).sum()) > 0 for leaf in leaves)


def test_blocks_finds_them_inside_a_model():
    model = nnx.Sequential(nnx.Conv(3, 8, (3, 3), rngs=nnx.Rngs(0)), block(kernels=[3, 5]))
    assert len(list(blocks(model))) == 1


def test_an_unknown_argument_warns_and_is_ignored():
    with pytest.warns(UserWarning, match="nonsense"):
        assert block(kernels=[3, 5], nonsense=True).num_experts == 2
