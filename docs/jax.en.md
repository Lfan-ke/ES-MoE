# Jax integration

```bash
uv add esmoe[jax]
```

On Jax the block is a Keras 3 layer. Install the extra, set `KERAS_BACKEND` to `jax`, and every
forward pass runs on Jax.

```bash
KERAS_BACKEND=jax python train.py
```

## As a pure function

On Jax a Keras layer presents itself as a pure function: `stateless_call` takes the variables in
and hands the outputs and the updated variables back, so it goes straight under `jax.jit` and
`jax.grad` with no state hidden anywhere.

```python
import jax
import keras
from esmoe.keras import ESMoE

net = keras.Sequential([ESMoE(kernels=[3, 5, 7], top_k=1)])
net(batch)  # the first pass builds the experts and the router
layer = net.layers[0]

trainable = [v.value for v in layer.trainable_variables]
other = [v.value for v in layer.non_trainable_variables]


def total(weights, x):
    out, _ = layer.stateless_call(weights, other, x)
    return out.sum()


jax.jit(total)(trainable, batch)
grads = jax.grad(total)(trainable, batch)
```

That is what `tests/test_backends.py` runs on the Jax leg of CI: the jitted value matches the eager
one, and the gradients are non-zero.

## Training

Through Keras it is like any other backend: `compile`, then `fit`, with the balancing term counted
by `add_loss`.

## Boundaries

Every expert is evaluated. Skipping an unrouted one depends on the batch, which a jitted graph
cannot express; the routing and the balancing term match the torch block bit for bit.

Grafting takes a `keras.Sequential`; a functional model has to be rebuilt by hand.
