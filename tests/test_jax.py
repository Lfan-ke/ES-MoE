"""The native Jax backend: pure functions over a parameter tree, no Keras and no Flax."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import numpy as np

import esmoe
from esmoe.jax import ESMoE

SHAPE = (2, 16, 16, 8)


def sample() -> jnp.ndarray:
    return jnp.asarray(np.random.rand(*SHAPE).astype("float32"))


def built(**block):
    held = ESMoE(**block)
    return held, held.init(jax.random.PRNGKey(0), SHAPE[-1])


def test_the_backend_table_points_here():
    assert esmoe.backend("jax").ESMoE is ESMoE


def test_it_owes_nothing_to_keras_or_flax():
    import pathlib

    import esmoe.jax

    source = "".join(p.read_text(encoding="utf-8") for p in pathlib.Path(esmoe.jax.__path__[0]).glob("*.py"))
    assert "keras" not in source and "flax" not in source


def test_kernels_decide_the_experts():
    held = ESMoE(kernels=[(3, 3), (1, 9)], top_k=1)
    assert (held.num_experts, held.top_k) == (2, 1)
    assert held.kernels == [(3, 3), (1, 9)]


def test_apply_returns_the_output_and_the_term():
    held, params = built(kernels=[3, 5, 7], top_k=1)
    out, aux = held.apply(params, sample())
    assert out.shape == SHAPE
    assert float(aux) > 0


def test_it_jits_and_differentiates():
    held, params = built(kernels=[3, 5])
    x = sample()

    def total(weights, inputs):
        out, aux = held.apply(weights, inputs, training=True)
        return out.sum() + 0.01 * aux

    assert float(jax.jit(total)(params, x)) == pytest.approx(float(total(params, x)), rel=1e-4)
    grads = jax.grad(total)(params, x)
    assert float(abs(grads["router"]["second"]).sum()) > 0
    assert all(float(abs(expert["pointwise"]).sum()) > 0 for expert in grads["experts"])


def test_an_unknown_argument_warns_and_is_ignored():
    with pytest.warns(UserWarning, match="nonsense"):
        assert ESMoE(kernels=[3, 5], nonsense=True).num_experts == 2


def test_the_output_norm_is_optional():
    held, params = built(kernels=[3, 5], out_norm=True)
    assert "norm" in params
    out, _ = held.apply(params, sample())
    assert out.shape == SHAPE
