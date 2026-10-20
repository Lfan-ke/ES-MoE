# In Keras

```bash
uv add esmoe[keras]         # or esmoe[tensorflow] / esmoe[jax]
```

The block is written twice: once in native torch, once in Keras 3. Keras 3 runs on tensorflow, jax
and torch, so that second implementation covers the other three. Both take the same arguments,
because the code that reads them is the same code.

```python
import keras
from esmoe.keras import ESMoE

net = keras.Sequential(
    [
        keras.layers.Conv2D(16, 3, padding="same"),
        ESMoE(kernels=[3, 5, 7], top_k=1),
    ]
)
net.compile(optimizer="adam", loss="mse")
net.fit(x, y)
```

The auxiliary loss goes through Keras' own `add_loss`, so `fit` already counts it. A custom loop
reads it with `esmoe.keras.collect_aux_loss(model)`, which is `model.losses` summed.

## Where it differs from the torch block

**Every expert is evaluated.** Skipping an unrouted one depends on the batch, which a static jax or
tensorflow graph cannot express. Routing and the balancing term match the torch block bit for bit;
the inference-time saving is torch's alone.

**The channel axis follows Keras.** It reads `keras.config.image_data_format()`, which defaults to
`channels_last`.

**Grafting takes a `Sequential`.** A functional model holds a graph rather than a list, so changing
it means rebuilding it. `esmoe.keras.graft` hands back a new `Sequential` instead of editing one.

## Choosing a backend

Keras decides, from the environment:

```bash
KERAS_BACKEND=jax python train.py
```

`esmoe.backend("jax")`, `esmoe.backend("tensorflow")` and `esmoe.backend("keras")` all lead here.
CI runs this layer's tests once on each of the three.
