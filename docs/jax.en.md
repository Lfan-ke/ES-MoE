# Jax integration

```bash
uv add esmoe[jax]
```

`esmoe.jax.ESMoE` is a set of pure functions. It needs Jax and nothing else, Keras and Flax
included. `init` builds the parameter tree, `apply` returns the output and the balancing term.
Inputs are NHWC.

```python
import jax
from esmoe.jax import ESMoE

block = ESMoE(kernels=[3, 5, 7], top_k=1)
params = block.init(jax.random.PRNGKey(0), channels=8)
out, aux = block.apply(params, x)
```

`kernels` gives each expert a receptive field, an `int` or an `(h, w)` pair, and sets how many
there are; `top_k` says how many run per sample.

## Training

`apply` hands the balancing term back, so the gradient is yours to take:

```python
def loss(params, x, y):
    out, aux = block.apply(params, x, training=True)
    return task_loss(out, y) + 0.01 * aux


grads = jax.grad(loss)(params, x, y)
```

All of it goes under `jax.jit`: no hidden state, and no registry to clear.

## The parameter tree

```
{"router": {"first", "first_bias", "second", "second_bias"},
 "experts": [{"depthwise", "pointwise", "scale", "offset"}, ...],
 "norm": {"scale", "offset"}}        # only when out_norm is on
```

Plain dictionaries and arrays, which `optax` and `jax.tree` take as they are.

## Boundaries

Every expert is evaluated. Skipping an unrouted one depends on the batch, which a jitted graph
cannot express.

Normalisation uses the batch's own statistics and keeps no moving average: in a functional
interface the state is either carried by the caller or absent, and this one makes it absent.
