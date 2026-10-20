# Flax integration

```bash
uv add esmoe[flax]
```

Flax ships two APIs side by side and both are supported: `esmoe.flax` is `nnx`, the one Flax
recommends now, and `esmoe.flax.linen` is Linen. Inputs are NHWC either way.

## nnx

State lives on the object, and `nnx.split` hands it to a pure function when one is wanted.

```python
from flax import nnx
from esmoe.flax import ESMoE

net = nnx.Sequential(
    nnx.Conv(3, 32, (3, 3), rngs=nnx.Rngs(0)),
    ESMoE(kernels=[3, 5, 7], top_k=1, channels=32, rngs=nnx.Rngs(1)),
)
out = net(x)
```

`kernels` gives each expert a receptive field, an `int` or an `(h, w)` pair, and sets how many
there are; `top_k` says how many run per sample. `channels` is required here: nnx builds its
parameters at construction, so there is no first forward pass to infer a width from.

## The auxiliary loss

The balancing term lives in the block's `nnx.Intermediate` variable; `routing` reads and clears it:

```python
from esmoe.flax import collect_aux_loss


@nnx.jit
def step(net, x):
    def objective(net):
        return task_loss(net(x)) + 0.01 * collect_aux_loss(net)

    return nnx.grad(objective)(net)
```

`nnx.jit` and `nnx.grad` work as they are, with nnx threading the state through the transform.

## linen

Parameters live outside the module: declare once, run with `apply`, and the balancing term is sown
into `intermediates`.

```python
from flax import linen
from esmoe.flax.linen import ESMoE, collect_aux_loss

block = ESMoE(kernels=[3, 5, 7], top_k=1)
variables = block.init(key, x, training=True)
out, state = block.apply(variables, x, training=True, mutable=["intermediates", "batch_stats"])
aux = collect_aux_loss(state)
```

Linen takes no `**kwargs`, so settings arrive in `options`:
`ESMoE(kernels=[3, 5], options={"out_norm": True})`. Channels come from the input `init` sees, so
they need not be stated.

## Against the other two ways onto jax

| route | shape | when |
|:--:|:--:|:--:|
| `esmoe.flax` | `nnx.Module`, state on the object | the project uses Flax's current API |
| `esmoe.flax.linen` | `linen.Module`, parameters outside | the project uses Linen |
| `esmoe.jax` | pure `init` / `apply` | no Flax wanted, the parameter tree is yours |
| `esmoe.keras` | `keras.layers.Layer` with the jax backend | the project uses Keras across backends |

The routing and the balancing term are the same on all three, and so are the arguments.

## Boundaries

Every expert is evaluated. Skipping an unrouted one depends on the batch, which a jitted graph
cannot express.
