# API

Six entry points, all importable from the top-level `esmoe`; the package ships `py.typed`, so the signatures are visible to IDEs and mypy.

## equip

    esmoe.equip(base="yolov8n.yaml", *, weight=0.01, out=None, **graft_kwargs) -> YOLO

Register, graft, build and wire in one call. `out` names the grafted config to keep; without it the config goes to a temporary directory (a YOLO wrapper loads models by path). `graft_kwargs` pass through to `graft`.

## inject_esmoe

    esmoe.inject_esmoe() -> type[ESMoE]

Exposes `ESMoE` where `parse_model` resolves layer names, after which any model.yaml can write `[-1, 1, ESMoE, [4, 2]]`.

## graft

    esmoe.graft(base, out=None, *, at="backbone_end", num_experts=4, top_k=2, rewire=False) -> dict

Inserts blocks after the layers named by `at` and renumbers every later reference. `at` is `"backbone_end"`, one index or several. With `rewire=True` every later consumer of an insertion layer is pointed at the block; without it, a head branch that names the old backbone end by index (YOLOv8's P5 lateral) keeps reading the pre-block feature.

## attach_aux_loss

    esmoe.attach_aux_loss(model, weight=0.01) -> model

Puts the router load-balancing loss into the optimised training loss; training logs gain an `esmoe_aux` column. Also routes `model.train()` through `esmoe.trainer`, which is how DDP workers register the block and recover the weight on their own.

## collect_aux_loss

    esmoe.collect_aux_loss(model, device=None) -> Tensor

Sums the router losses published by the latest forward pass, for custom training loops. Repeated calls cannot double-count a stale value.

## ESMoE

    esmoe.ESMoE(num_experts=4, top_k=2, channels=None, *, reduction=8,
                max_kernel_size=15, expert_kernel_sizes=None,
                expert=DWExpert, balance=switch_balance)

The channel-preserving mixture block. `channels` is inferred on the first forward when omitted. `expert` is a `(c1, c2, k) -> Module` factory and `balance` a `(probs, gate) -> scalar` objective; both are replaceable. `esmoe.blocks(model)` iterates every block in a model in module order.
