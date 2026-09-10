# API

Six entry points, all importable from the top-level `esmoe`; the package ships `py.typed`, so the signatures are visible to IDEs and mypy.

## equip

    esmoe.equip(base="yolov8n.yaml", *, weight=0.01, out=None, **graft_kwargs) -> YOLO

Register, graft, build and wire in one call. `out` names the grafted config to keep; without it the config goes to a temporary directory (a YOLO wrapper loads models by path). `graft_kwargs` pass through to `graft`.

## inject_esmoe

    esmoe.inject_esmoe() -> type[ESMoE]

Exposes `ESMoE` where `parse_model` resolves layer names, after which any model.yaml can write `[-1, 1, ESMoE, [4, 2]]`.

## graft

    esmoe.graft(base, out=None, *, at="backbone_end", num_experts=4, top_k=2,
                rewire=False, **settings) -> dict

Inserts blocks after the layers named by `at` and renumbers every later reference. `at` is `"backbone_end"`, `"backbone_stages"` (one block after each backbone stage, reproducing the upstream layout), one index or several. With `rewire=True` every later consumer of an insertion layer is pointed at the block; without it, a head branch that names the old backbone end by index (YOLOv8's P5 lateral) keeps reading the pre-block feature.

`**settings` takes `balance`, `out_norm`, `dense_training`, `sparse_inference` and `dynamic_threshold`, and **writes them into the config rather than onto the instance**: the trainer rebuilds the model from `model.yaml`, and anything set on the instance goes with the instance it discards. `balance` may be a name or one of the four functions that ship; a custom objective cannot go into a config and is refused with a reason.

## attach_aux_loss

    esmoe.attach_aux_loss(model, weight=0.01) -> model

Puts the router load-balancing loss into the optimised training loss; training logs gain an `esmoe_aux` column. Also routes `model.train()` through `esmoe.trainer`, which is how DDP workers register the block and recover the weight on their own.

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

A mixture-of-experts block, channel-preserving unless `out_channels` says otherwise. `channels` is inferred on the first forward when omitted; `top_k=None` activates every expert. `expert` is a replaceable `(c1, c2, k) -> Module` factory. `options` is the mapping a config carries settings in (`[-1, 1, ESMoE, [4, 2, null, {out_norm: true}]]`) and is equivalent to `**settings`.

The five settings and their defaults (`esmoe.SETTINGS`):

| setting | default here | upstream | what it does |
|:--:|:--:|:--:|:--|
| `balance` | `gshard_balance` | same | the objective, `(probs, gate) -> scalar`, or a name from `esmoe.BALANCES` |
| `out_norm` | `False` | always on | `BatchNorm + SiLU` after the weighted sum (the paper's eq. 2 `Norm`) |
| `dense_training` | `False` | always on | run every expert while training; unrouted ones are weighted zero but their normalisation statistics keep moving |
| `sparse_inference` | `True` | same | skip unrouted experts outside training |
| `dynamic_threshold` | `0.0` | `0.4` | outside training, drop a routed expert below the threshold, keep the leader, renormalise |

The last two affect inference only, the first three affect training. The defaults keep the runs already in `results/` reproducible; they are not a judgement against upstream.

`block.spec()` returns the five settings a block is holding, and `block.configure(**settings)` changes them on paths that never reach a trainer -- inference, export, a unit test. `esmoe.blocks(model)` walks every block in a model in module order, and `scripts/blockspec.py` reads back from any checkpoint what was actually in force.
