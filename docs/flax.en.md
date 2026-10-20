# Flax integration

```bash
uv add esmoe[flax]
```

`esmoe.flax.ESMoE` is an `nnx.Module`, written the way Flax is written now: state lives on the
object, and `nnx.split` hands it to a pure function when one is wanted. Inputs are NHWC.

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

## Against the other two ways onto jax

| route | shape | when |
|:--:|:--:|:--:|
| `esmoe.flax` | `nnx.Module`, state on the object | the project already uses Flax |
| `esmoe.jax` | pure `init` / `apply` | no Flax wanted, the parameter tree is yours |
| `esmoe.keras` | `keras.layers.Layer` with the jax backend | the project uses Keras across backends |

The routing and the balancing term are the same on all three, and so are the arguments.

## Boundaries

Every expert is evaluated. Skipping an unrouted one depends on the batch, which a jitted graph
cannot express.
