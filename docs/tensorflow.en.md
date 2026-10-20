# TensorFlow integration

```bash
uv add esmoe[tensorflow]
```

`esmoe.tensorflow.ESMoE` is a `tf.Module`. It needs TensorFlow and nothing else, Keras included.
Inputs are NHWC.

```python
import tensorflow as tf
from esmoe.tensorflow import ESMoE, Sequential

net = Sequential([ESMoE(kernels=[3, 5, 7], top_k=1)])
out = net(tf.random.normal([2, 16, 16, 8]), training=True)
```

`kernels` gives each expert a receptive field, an `int` or an `(h, w)` pair, and sets how many
there are; `top_k` says how many run per sample. Channels come from the first call.

## The auxiliary loss

The balancing term joins the total loss yourself:

```python
from esmoe.tensorflow import routing

with routing(net) as route:
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(net(x, training=True)) + 0.01 * route.loss()
grads = tape.gradient(total, net.trainable_variables)
```

`routing` clears the registry on the way in, so a block that did not run this step cannot report
the step before. Without the context manager, pair `clear_aux_loss` with `collect_aux_loss`.

## Into a model that already exists

```python
from esmoe.tensorflow import graft

graft(model, at="stem", mode="after", kernels=[3, 5, 7], top_k=1)
```

`at` takes an attribute name, several names, or a `(name, module) -> bool` test; layers inside a
`Sequential` are addressed by position. `mode` is `after`, `before` or `replace`, applied in place.

## Boundaries

Every expert is evaluated. Skipping an unrouted one depends on the batch, which a static graph
cannot express.

Batch normalisation keeps its moving statistics in untrained variables, updated when
`training=True`.
