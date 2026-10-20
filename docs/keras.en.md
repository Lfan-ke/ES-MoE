# Keras 3 integration

```bash
uv add esmoe[keras]
```

`esmoe.keras.ESMoE` is a `keras.layers.Layer` written against `keras.ops`, so it runs on whichever
backend Keras is configured for: tensorflow, jax or torch.

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

## Boundaries

**Every expert is evaluated.** Skipping an unrouted one depends on the batch, which a static graph
cannot express.

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

The arguments are the ones [Torch integration](torch.md) takes: the code that reads them is shared.
