"""The Flax backend's Linen variant, the other API Flax still ships."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax.linen")

from esmoe.flax.linen import (
    ESMoE,
    collect_aux_loss,
)

SHAPE = (2, 16, 16, 8)


def sample():
    return jax.numpy.asarray(np.random.rand(*SHAPE).astype("float32"))


def built(**held):
    block = ESMoE(**held)
    x = sample()
    return block, block.init(jax.random.PRNGKey(0), x, training=True), x


def test_kernels_decide_the_experts():
    plan = ESMoE(kernels=[(3, 3), (1, 9)], top_k=1).plan()
    assert (plan.num_experts, plan.top_k) == (2, 1)
    assert plan.kernels == [(3, 3), (1, 9)]


def test_apply_keeps_the_channels_and_sows_the_term():
    block, variables, x = built(kernels=[3, 5, 7], top_k=1)
    out, state = block.apply(variables, x, training=True, mutable=["intermediates", "batch_stats"])
    assert tuple(out.shape) == SHAPE
    assert float(collect_aux_loss(state)) > 0


def test_it_jits_and_differentiates():
    block, variables, x = built(kernels=[3, 5])

    def objective(params):
        out, state = block.apply(params, x, training=True, mutable=["intermediates", "batch_stats"])
        return out.sum() + 0.01 * collect_aux_loss(state)

    assert float(jax.jit(objective)(variables)) == pytest.approx(float(objective(variables)), rel=1e-4)
    grads = jax.grad(objective)(variables)
    assert any(float(abs(leaf).sum()) > 0 for leaf in jax.tree.leaves(grads))


def test_the_channels_come_from_the_input():
    block, variables, x = built(kernels=[3, 5])
    out, _ = block.apply(variables, x, training=True, mutable=["intermediates", "batch_stats"])
    assert out.shape[-1] == SHAPE[-1]


def test_an_unknown_argument_warns_and_is_ignored():
    with pytest.warns(UserWarning, match="nonsense"):
        assert ESMoE(kernels=[3, 5], options={"nonsense": True}).plan().num_experts == 2
