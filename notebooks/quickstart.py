import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # ES-MoE in a marimo notebook

        [esmoe](https://github.com/Lfan-ke/ES-MoE) is an expert-sparse mixture-of-experts block. It
        drops into any torch model, it adds itself to Ultralytics YOLO in one call, and a second
        implementation on Keras 3 runs it on tensorflow, jax and torch. The router's
        load-balancing term reaches the loss the optimiser sees either way.

        This notebook needs no dataset and no GPU: it builds the block three ways, reads it back,
        and shows what the router does with one batch. The training walk-through is in
        [`quickstart.ipynb`](https://colab.research.google.com/github/Lfan-ke/ES-MoE/blob/main/notebooks/quickstart.ipynb).
        """
    )
    return (mo,)


@app.cell
def _():
    import torch
    import ultralytics

    import esmoe

    versions = {
        "esmoe": esmoe.__version__,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
    }
    versions
    return esmoe, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## A plain torch model

        `kernels` gives each expert its receptive field and, with it, how many experts there are;
        `top_k` says how many run per sample. The block takes its width from the first forward
        pass, so nothing has to declare it, and `routing` hands you the balancing term for your
        own training step.
        """
    )
    return


@app.cell
def _(esmoe, torch):
    from torch import nn

    net = nn.Sequential(
        nn.Conv2d(3, 32, 3, 2, 1),
        esmoe.ESMoE(kernels=[3, 5, 7], top_k=1),
        esmoe.ESMoE(kernels=[(1, 9), (9, 1)], top_k=2),
    )

    with esmoe.routing(net) as route:
        plain = net(torch.randn(4, 3, 64, 64))
        step = plain.square().mean() + 0.01 * route.loss()
    step.backward()

    {"output": tuple(plain.shape), "router loss": round(float(route.loss()), 4)}
    return (nn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## One call puts the block in a YOLO

        `equip` registers `ESMoE`, grafts it onto the backbone, renumbers every head reference the
        insertion would break, builds the model, and attaches the auxiliary loss.
        """
    )
    return


@app.cell
def _(esmoe):
    model = esmoe.equip("yolo11n.yaml", weight=0.01)
    block = next(esmoe.blocks(model.model))
    block
    return (block,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## What the router decides

        The router scores the whole image, keeps the top `k` experts and renormalises their weights.
        Move the slider to feed the block a different random batch: the chosen experts change with
        the input, which is the whole point of the block.
        """
    )
    return


@app.cell
def _(mo):
    seed = mo.ui.slider(0, 20, value=0, label="batch seed")
    seed
    return (seed,)


@app.cell
def _(block, seed, torch):
    import torch.nn.functional as functional

    torch.manual_seed(seed.value)
    images = torch.randn(1, 64, 32, 32)
    block(images)
    probs = functional.softmax(block.router(images).float(), dim=1).flatten()
    chosen = probs.topk(block.top_k).indices.tolist()

    {
        f"k{size}": {"share": round(float(probs[index]), 4), "chosen": index in chosen}
        for index, size in enumerate(block.kernels)
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Your own expert, your own balancing term

        Both extension points are plain callables: `expert(c1, c2, k)` builds one branch, and
        `balance(probs, gate)` scores how evenly the router spreads its traffic.
        """
    )
    return


@app.cell
def _(esmoe, nn, torch):
    class ThinExpert(nn.Sequential):
        def __init__(self, c1, c2, k):
            super().__init__(nn.Conv2d(c1, c2, k, 1, k // 2, groups=c1), nn.SiLU())

    def entropy_balance(probs, gate):
        return -(probs * probs.clamp_min(1e-9).log()).sum(dim=1).mean()

    custom = esmoe.ESMoE(kernels=[3, 5, 7], top_k=2, expert=ThinExpert, balance=entropy_balance)
    custom(torch.randn(2, 32, 16, 16)).shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## The same block on Keras

        `esmoe.keras.ESMoE` is the same block written against `keras.ops`, so it runs on whichever
        backend Keras is configured for: tensorflow, jax or torch. The arguments are the torch
        block's, and the balancing term goes through Keras' own `add_loss`, so `fit` counts it.
        """
    )
    return


@app.cell
def _():
    import os

    os.environ.setdefault("KERAS_BACKEND", "torch")

    import keras
    import numpy as np

    from esmoe.keras import ESMoE as KerasESMoE
    from esmoe.keras import collect_aux_loss

    layered = keras.Sequential(
        [keras.layers.Conv2D(16, 3, strides=2, padding="same"), KerasESMoE(kernels=[3, 5, 7], top_k=1)]
    )
    layered(np.random.rand(2, 32, 32, 3).astype("float32"))

    {"backend": keras.backend.backend(), "router loss": round(float(collect_aux_loss(layered)), 4)}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Native TensorFlow and native Jax

        Keras is one of four implementations, not the way the other frameworks are reached:
        TensorFlow has a `tf.Module` of its own, Jax a pair of pure functions, and neither imports
        Keras.
        """
    )
    return


@app.cell
def _():
    try:
        import tensorflow as tf

        from esmoe.tensorflow import ESMoE as TFESMoE
        from esmoe.tensorflow import Sequential as TFSequential
        from esmoe.tensorflow import collect_aux_loss as tf_aux

        tf_net = TFSequential([TFESMoE(kernels=[3, 5, 7], top_k=1)])
        tf_out = tf_net(tf.random.normal([2, 16, 16, 8]), training=True)
        native_tf = {"output": tuple(tf_out.shape), "router loss": round(float(tf_aux(tf_net)), 4)}
    except ModuleNotFoundError:
        native_tf = "tensorflow not installed here"

    native_tf
    return


@app.cell
def _():
    try:
        import jax

        from esmoe.jax import ESMoE as JaxESMoE

        jax_block = JaxESMoE(kernels=[3, 5, 7], top_k=1)
        jax_params = jax_block.init(jax.random.PRNGKey(0), channels=8)
        jax_out, jax_aux = jax_block.apply(jax_params, jax.numpy.zeros((2, 16, 16, 8)) + 0.5)
        native_jax = {"output": tuple(jax_out.shape), "router loss": round(float(jax_aux), 4)}
    except ModuleNotFoundError:
        native_jax = "jax not installed here"

    native_jax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Where to go next

        - [Documentation](https://lfan-ke.github.io/ES-MoE/en/) — tutorial and API.
        - [Try it in the browser](https://lfan-ke.github.io/ES-MoE/demo/) — upload an image and see
          which experts each block picks.
        - [Experiments](https://lfan-ke.github.io/ES-MoE/en/experiments/) — the runs behind every
          number, three seeds or more per cell.
        """
    )
    return


if __name__ == "__main__":
    app.run()
