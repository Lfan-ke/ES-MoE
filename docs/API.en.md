# API

Six entry points, all importable from the top-level `esmoe`; the package ships `py.typed`, so the signatures are visible to IDEs and mypy.

## equip

    esmoe.equip(base="yolov8n.yaml", *, weight=0.01, recipe="esmoe", out=None, **graft_kwargs) -> YOLO

Register, graft, build and wire in one call. `out` names the grafted config to keep; without it the config goes to a temporary directory (a YOLO wrapper loads models by path). `weight` and `recipe` go to `attach_aux_loss`, `graft_kwargs` pass through to `graft`.

## inject_esmoe

    esmoe.inject_esmoe() -> type[ESMoE]

Exposes `ESMoE` where `parse_model` resolves layer names, after which any model.yaml can write `[-1, 1, ESMoE, [4, 2]]`.

## graft

    esmoe.graft(base, out=None, *, at="backbone_end", num_experts=4, top_k=2,
                rewire=False, out_channels=None, **settings) -> dict

Inserts blocks after the layers named by `at` and renumbers every later reference. `at` is `"backbone_end"`, `"backbone_stages"` (one block after each backbone stage, reproducing the upstream layout), one index or several. With `rewire=True` every later consumer of an insertion layer is pointed at the block; without it, a head branch that names the old backbone end by index (YOLOv8's P5 lateral) keeps reading the pre-block feature.

`**settings` takes `balance`, `out_norm`, `dense_training`, `sparse_inference`, `dynamic_threshold` and `expert`, and **writes them into the config rather than onto the instance**: the trainer rebuilds the model from `model.yaml`, and anything set on the instance goes with the instance it discards. `balance` may be a name or a function, `expert` a class or a factory. The ones that ship are written by short name (objectives `switch`, `gshard`, `master`, `gshard_probs`; the expert `dw`); a custom one is written as `module:qualname`, and every rebuild -- the one in a DDP worker included -- imports the same object back from that name. A custom function or class therefore has to live at module level in a module the training environment can import; a lambda, a nested function or anything defined in `__main__` is refused when grafting.

`out_channels` widens the blocks to that many channels, taken literally rather than scaled by the yaml's width multiple. Stock `parse_model` takes a third-party module's output width to be its input width, so every widened block is followed by the official `Index` layer: the block hands it a one-element list, and `parse_model` reads that layer's declared width. With `rewire=True` consumers are pointed at that layer.

## attach_aux_loss

    esmoe.attach_aux_loss(model, weight=0.01, recipe="esmoe") -> model

Puts the router load-balancing loss into the optimised training loss; training logs gain an `esmoe_aux` column. Also routes `model.train()` through `esmoe.trainer`, which is how DDP workers register the block and recover the weight and recipe on their own. Inside a process group an expert no image routed to joins the graph at zero weight, so multi-GPU training works under `compile=True` too, where ultralytics turns `find_unused_parameters` off.

`recipe` decides how the block and its term train:

- `"esmoe"` (default, and what every recorded run used): the term times `weight`, counted per image the way the task loss counts.
- `"upstream"`: the three things YOLO-Master's trainer does to any model with a routed module, for runs compared against it. The term is divided by a running mean of its own magnitude (decay 0.99, starting at 1.0), multiplied by `weight`, capped at 3.0 and added once to each of box, cls and dfl; router parameters get a group of their own at half the learning rate, outside Muon; expert parameters stay frozen for the first 3 epochs. The last two happen in the trainer, so they apply to a YOLO model trained through `model.train()`. Constants and sources are in `esmoe.upstream`.

## collect_aux_loss

    esmoe.collect_aux_loss(model, device=None) -> Tensor

Sums the routing losses published by the most recent forward, for custom training loops. Calling it twice cannot count a stale value again.

## clear_aux_loss

    esmoe.clear_aux_loss() -> None

Drops every value blocks have published into the registry. The loss patch `attach_aux_loss` installs calls it before each forward; a custom training loop that does not go through that patch has to call it itself, or a block that does not run in some step will still be answering with the value from the step before.

## odd / odd_kernels

    esmoe.odd(size) -> int
    esmoe.odd_kernels(num_experts, max_kernel_size=15) -> list[int]

`odd` steps an even kernel down (4 -> 3) so the padding stays centred, which is what upstream does to both explicit kernel sizes and the cap. `odd_kernels` generates the default heterogeneous set `3, 5, 7, ...` capped at `max_kernel_size` -- the one `ESMoE` uses when `expert_kernel_sizes` is not given.

## ESMoE

    esmoe.ESMoE(num_experts=4, top_k=2, channels=None, options=None, *,
                out_channels=None, reduction=8, max_kernel_size=15,
                expert_kernel_sizes=None, expert=DWExpert, **settings)

A mixture-of-experts block, channel-preserving unless `out_channels` says otherwise. `channels` is inferred on the first forward when omitted; `top_k=None` activates every expert. `expert` is a `(c1, c2, k) -> Module` factory, or its name (a short name from `esmoe.EXPERTS`, or `module:qualname`). `options` is the mapping a config carries settings in (`[-1, 1, ESMoE, [4, 2, null, {out_norm: true}]]`); it is equivalent to `**settings` and may also carry `expert` and `out_channels`. An `out_channels` given through `options` makes the block return a one-element list for the `Index` layer after it.

The five settings and their defaults (`esmoe.SETTINGS`):

| setting | default here | upstream | what it does |
|:--:|:--:|:--:|:--|
| `balance` | `switch_balance` | `gshard_balance` | the objective, `(probs, gate) -> scalar`, or a name from `esmoe.BALANCES`, or `module:qualname` |
| `out_norm` | `False` | always on | `BatchNorm + SiLU` after the weighted sum (the paper's eq. 2 `Norm`) |
| `dense_training` | `False` | always on | run every expert while training; unrouted ones are weighted zero but their normalisation statistics keep moving |
| `sparse_inference` | `True` | same | skip unrouted experts outside training |
| `dynamic_threshold` | `0.0` | `0.4` | outside training, drop a routed expert below the threshold, keep the leader, renormalise |

The last two affect inference only, the first three affect training. The defaults for `out_norm`, `dense_training` and `dynamic_threshold` keep the runs already in `results/` reproducible; they are not a judgement against upstream. The default for `balance` is settled by data: an objective that reads the gate (upstream's `gshard`, the paper's `master`) has no gradient for an expert outside the top-k, and five of six such checkpoints lost an expert, while Switch reads the full softmax and lost none in 66 ([judgment lines](JUDGMENT.md), round six).

`block.spec()` returns the five settings a block is holding, plus `expert` when a custom one is in use, and `block.configure(**settings)` changes them on paths that never reach a trainer -- inference, export, a unit test. `esmoe.blocks(model)` walks every block in a model in module order, and `scripts/blockspec.py` reads back from any checkpoint what was actually in force.
