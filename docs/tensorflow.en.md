# TensorFlow integration

```bash
uv add esmoe[tensorflow]
```

On TensorFlow the block is a Keras 3 layer. Install the extra, set `KERAS_BACKEND` to
`tensorflow`, and `esmoe.keras.ESMoE` is an ordinary TensorFlow layer.

```bash
KERAS_BACKEND=tensorflow python train.py
```

## As an ordinary layer

```python
import keras
from esmoe.keras import ESMoE

net = keras.Sequential(
    [
        keras.layers.Conv2D(16, 3, strides=2, padding="same"),
        ESMoE(kernels=[3, 5, 7], top_k=1),
    ]
)
net.compile(optimizer="adam", loss="mse")
net.fit(x, y)
```

The balancing term goes through Keras' `add_loss`, so `fit` counts it.

## In a graph, and under a tape

The layer traces into a `tf.function`, and gradients come from a `GradientTape` as usual. Read the
balancing term inside the same trace: a tensor created there cannot be used outside the function.

```python
import tensorflow as tf


@tf.function
def total(inputs):
    out = net(inputs)
    return tf.reduce_sum(out) + tf.add_n(net.losses)


with tf.GradientTape() as tape:
    value = total(tf.constant(batch))
grads = tape.gradient(value, net.trainable_variables)
```

That is what `tests/test_backends.py` runs on the TensorFlow leg of CI.

## Boundaries

Every expert is evaluated. Skipping an unrouted one depends on the batch, which a static graph
cannot express; the routing and the balancing term match the torch block bit for bit, and only the
inference-time saving is torch's alone.

Grafting takes a `keras.Sequential`; a functional model has to be rebuilt by hand. The channel axis
follows `keras.config.image_data_format()`.
