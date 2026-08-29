# Tutorial

This walks through the whole path: install the block, put it into a stock YOLO, prove its auxiliary
loss actually reaches the optimiser, and run a comparison whose result you can defend.

## Why a plug-in and not a fork

The ES-MoE block comes from [YOLO-Master](https://github.com/Tencent/YOLO-Master), whose package is
itself named `ultralytics`. Installing it therefore *replaces* the official library rather than
extending it: one import path, one winner, and every downstream tool now runs on a non-official
distribution. Extracting the block by hand is no better - upstream `ES_MOE` reaches into a global
loss registry, a mixture-loss collector and trainer plumbing that a single copied file does not
bring with it.

`esmoe` takes the other route. It is a separate distribution that depends on `ultralytics`, uses
only public extension points, and can be uninstalled without a trace.

## Install

    pip install esmoe

## The block in one picture

```mermaid
flowchart LR
    X["feature map<br/>(n, c, h, w)"] --> R["router<br/>pool -> linear -> SiLU -> linear"]
    R --> P["softmax over experts"]
    P --> T["top-k, renormalised"]
    X --> E1["expert k=3"]
    X --> E2["expert k=5"]
    X --> E3["expert k=7"]
    X --> E4["expert k=9"]
    T --> S(("weighted sum"))
    E1 --> S
    E2 --> S
    E3 --> S
    E4 --> S
    S --> Y["output<br/>(n, c, h, w)"]
    P -.-> A["load-balancing loss"]
    T -.-> A
    A -.-> L["training loss"]
```

Each expert is a depthwise-separable convolution with its own kernel size (3, 5, 7, 9, ...), so the
experts differ in receptive field rather than merely in weights. A lightweight router scores the
experts per image, the top-k are kept and renormalised, and the block returns their weighted sum.
The block preserves channel count, which is what lets a stock `parse_model` size it.

The auxiliary term is the Switch-Transformer load-balancing loss,

$$
\mathcal{L}_{\text{aux}} = E \sum_{i=1}^{E} \bar{p}_i \cdot f_i ,
$$

where $E$ is the expert count, $\bar{p}_i$ the mean routing probability of expert $i$ over the batch
and $f_i$ the fraction of samples that actually activated it. It is minimised when routing mass and
realised load are spread evenly, which is what keeps the router from collapsing onto one expert.

## Five minutes

    import esmoe

    model = esmoe.equip("yolo11n.yaml", weight=0.01)
    model.train(data="coco8.yaml", epochs=3, imgsz=320)

`equip` does four things: registers `ESMoE` so a config can name it, grafts it onto the backbone,
builds the model, and wires the auxiliary loss into training. Take them apart when you need to:

    esmoe.inject_esmoe()
    esmoe.graft("yolov8n.yaml", out="v8-esmoe.yaml", at=[4, 6])
    model = YOLO("v8-esmoe.yaml")
    esmoe.attach_aux_loss(model, weight=0.01)

or from the shell:

    esmoe graft yolo11n.yaml -o yolo11n-esmoe.yaml -e 4 -k 2 --at 4,6

## What grafting has to get right

A YOLO config addresses earlier layers by absolute index:

    - [[-1, 12], 1, Concat, [1]]

Insert a layer at position 10 and every reference at or past 10 now points one layer too early. The
model still builds, still trains, and is quietly wrong. `graft` therefore renumbers every reference
that sits at or after each insertion point, and a unit test compares the rewritten head against the
original one reference by reference.

## Proving the auxiliary loss is real

A configuration key named `aux_loss` proves nothing. What proves it:

1. `results.csv` gains an `esmoe_aux` column, non-zero and moving;
2. a unit test asserts `total(with aux) == total(without) + aux * batch_size`;
3. the same test asserts the router receives gradient.

If you want the number yourself in a custom loop:

    aux = esmoe.collect_aux_loss(model)
    (task_loss + 0.01 * aux).backward()

`collect_aux_loss` only sums values published by the latest forward pass, so calling it twice cannot
double-count a stale graph.

## A comparison you can defend

    uv run python scripts/capture_env.py             # freeze versions and hardware into env/
    EPOCHS=20 FRACTION=1.0 SEEDS="0 1 2" bash scripts/sweep.sh
    uv run python scripts/report.py                  # results/summary.md

Each run writes one JSON record: model config, dataset and fraction, hardware, budget, seed,
metrics, artifact path, status, limitation. `report.py` groups arms by backbone, block config *and*
budget - never averaging two different budgets into one row - then prints per-seed paired deltas
against the baseline of the same seed.

Read the paired table, not the two means. Per-arm standard deviations overlap in this kind of
experiment; what carries the claim is that the same seed, same data and same schedule moved in the
same direction three times. On the full VisDrone training set the shipped configuration wins 3/3
seeds by +0.0021 mAP50, and one of those seeds is nearly a tie - a small, consistent effect, not a
reliable per-run improvement. `limitations.md` states the rest.

## Extending

Experts and the balancing objective are plain callables:

    class ThinExpert(nn.Sequential):
        def __init__(self, c1, c2, k):
            super().__init__(nn.Conv2d(c1, c2, k, 1, k // 2, groups=c1), nn.SiLU())

    def entropy_balance(probs, gate):
        return -(probs * probs.clamp_min(1e-9).log()).sum(dim=1).mean()

    block = esmoe.ESMoE(num_experts=3, top_k=2, expert=ThinExpert, balance=entropy_balance)

`esmoe.blocks(model)` iterates every block in a model, which is how the collector and the tests find
them.

## Pitfalls worth knowing

- **Channels are inferred on the first forward.** Ultralytics runs a forward pass while building the
  model, so this is invisible in normal use - but a model that is scripted or `state_dict`-loaded
  before any forward has no expert weights to load into yet.
- **The trainer rebuilds the model** from the config, and takes the EMA copy before any callback
  runs. The aux weight therefore lives at process scope as well as on the instance; a process trains
  one aux setting at a time.
- **Loss items changed shape across ultralytics releases** - a tensor before 8.4.13x, a named dict
  after. Both are handled; if you patch the loss yourself, handle both.
- **A checkpoint reloaded in a process that never calls `attach_aux_loss`** trains without the
  auxiliary term. The block still runs; only the extra loss is missing.
