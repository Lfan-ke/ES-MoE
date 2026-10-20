import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # ES-MoE in a marimo notebook

        [esmoe](https://github.com/Lfan-ke/ES-MoE) adds an expert-sparse Mixture-of-Experts block to
        Ultralytics YOLO. It installs beside the official `ultralytics` package, and it wires the
        router's load-balancing term into the loss the optimiser sees.

        This notebook needs no dataset and no GPU: it builds a model, reads the block back, and
        shows what the router does with one batch. The training walk-through is in
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
        ## One call puts the block in

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
def _(esmoe, torch):
    from torch import nn

    class ThinExpert(nn.Sequential):
        def __init__(self, c1, c2, k):
            super().__init__(nn.Conv2d(c1, c2, k, 1, k // 2, groups=c1), nn.SiLU())

    def entropy_balance(probs, gate):
        return -(probs * probs.clamp_min(1e-9).log()).sum(dim=1).mean()

    custom = esmoe.ESMoE(num_experts=3, top_k=2, expert=ThinExpert, balance=entropy_balance)
    custom(torch.randn(2, 32, 16, 16)).shape
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
