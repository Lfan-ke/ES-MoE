"""Config-level injection: graft ESMoE blocks into a stock Ultralytics model.yaml."""

import copy
from collections.abc import Iterable

Spot = int | str | Iterable[int]


def _stage_ends(backbone: list, backbone_len: int) -> list[int]:
    """The last layer of each backbone stage: every layer a stride-2 Conv follows, plus the end.

    Upstream places one block after each stage rather than one at the end, so this reproduces
    that layout on any backbone without hard-coding indices for a particular yaml.
    """
    ends = {
        i
        for i in range(backbone_len)
        if not _downsamples(backbone[i]) and i + 1 < backbone_len and _downsamples(backbone[i + 1])
    }
    return sorted(ends | {backbone_len - 1})


# Every backbone downsamples, but not all of them do it with a stride-2 Conv: v9 uses AConv/ADown
# and v10 uses SCDown. Naming them keeps the stage split correct across generations.
DOWNSAMPLERS = {"AConv", "ADown", "SCDown"}


def _downsamples(layer: list) -> bool:
    if layer[2] in DOWNSAMPLERS:
        return True
    args = layer[3] if len(layer) > 3 else []
    return layer[2] == "Conv" and isinstance(args, list) and len(args) > 2 and args[2] == 2


def _spots(at: Spot, backbone_len: int, total: int, backbone: list | None = None) -> list[int]:
    """Normalise ``at`` to original layer indices to insert after."""
    match at:
        case "backbone_end":
            spots = [backbone_len - 1]
        case "backbone_stages":
            spots = _stage_ends(backbone or [], backbone_len)
        case int() as index:
            spots = [index if index >= 0 else total + index]
        case str() as name:
            raise ValueError(
                f"unknown insertion point {name!r}; use 'backbone_end', 'backbone_stages', an index, or indices"
            )
        case _:
            spots = [i if i >= 0 else total + i for i in at]
    if bad := [i for i in spots if not 0 <= i < total]:
        raise ValueError(f"insertion points out of range for a {total}-layer model: {bad}")
    return sorted(set(spots))


def graft(
    base: str = "yolov8n.yaml",
    out: str | None = None,
    *,
    at: Spot = "backbone_end",
    num_experts: int = 4,
    top_k: int = 2,
    rewire: bool = False,
    out_channels: int | None = None,
    **options,
) -> dict:
    """Insert ESMoE blocks after the given layers and renumber every later reference.

    Layers address earlier layers by absolute index, so an insertion invalidates every reference at
    or past it unless the references move with it.

    Args:
        base: Anything ``yaml_model_load`` accepts, e.g. ``yolo11n.yaml`` or a path.
        out: Optional path to write the grafted config to.
        at: ``"backbone_end"`` (default), ``"backbone_stages"`` for one block after every
            stage as upstream does, a layer index, or several indices.
        num_experts: Experts per block.
        top_k: Experts activated per sample.
        rewire: Also point every later consumer of an insertion layer at the block that now
            follows it. Off, a head branch that names the old backbone end by index keeps
            reading the pre-block feature (YOLOv8's P5 lateral does exactly that).
        out_channels: Widen every block to this many output channels, taken literally rather than
            scaled by the yaml's width multiple. Stock ``parse_model`` takes a third-party module's
            output width to be its input width, so each widened block is followed by the official
            ``Index`` layer, whose declared width ``parse_model`` does read; the block hands it a
            one-element list, and ``rewire`` points consumers at that layer.
        **options: Block settings to write into the config - ``balance``, ``out_norm``,
            ``dense_training``, ``sparse_inference``, ``dynamic_threshold`` - and ``expert``.
            They belong in the config because the trainer rebuilds the model from it, discarding
            anything set on the instance beforehand. A custom ``balance`` or ``expert`` is written
            as ``module:qualname`` and has to be importable wherever the model is rebuilt, DDP
            workers included.
    """
    from ultralytics.nn.tasks import yaml_model_load

    from .module import BALANCES, EXPERTS, SETTINGS, reference

    allowed = set(SETTINGS) | {"expert"}
    if unknown := set(options) - allowed:
        raise ValueError(f"unknown ESMoE options {sorted(unknown)}; expected {sorted(allowed)}")
    options = dict(options)
    if callable(balance := options.get("balance")):
        options["balance"] = reference(balance, BALANCES)
    if callable(expert := options.get("expert")):
        options["expert"] = reference(expert, EXPERTS)
    layers = [[-1, 1, "ESMoE", [num_experts, top_k]]]
    if out_channels is not None:
        import ultralytics.nn.tasks as tasks

        if not hasattr(tasks, "Index"):
            raise RuntimeError("widening a grafted block needs an ultralytics release that has the Index layer")
        options["out_channels"] = int(out_channels)
        layers.append([-1, 1, "Index", [int(out_channels), 0]])
    if options:
        layers[0][3] += [None, options]

    d = dict(yaml_model_load(base))
    d.pop("yaml_file", None)
    backbone, head = list(d["backbone"]), list(d["head"])
    spots = _spots(at, len(backbone), len(backbone) + len(head), backbone)

    grafted: list[list] = []
    moved: dict[int, int] = {}
    backbone_len = len(backbone)
    for index, layer in enumerate([*backbone, *head]):
        moved[index] = len(grafted)
        grafted.append(list(layer))
        if index in spots:
            # A copy per insertion: rows that share one options mapping come out of the yaml dump as
            # anchors and aliases, which reload but no longer read as the layer they describe.
            grafted.extend(copy.deepcopy(layers))
            backbone_len += len(layers) * (index < len(backbone))

    targets = moved | {s: moved[s] + len(layers) for s in spots} if rewire else moved
    renumbered = [[_shift(layer[0], targets), *layer[1:]] for layer in grafted]
    d["backbone"], d["head"] = renumbered[:backbone_len], renumbered[backbone_len:]
    if out:
        from ultralytics.utils import YAML

        YAML.save(out, d)
    return d


def _shift(source, moved: dict[int, int]):
    if isinstance(source, list):
        return [_shift(s, moved) for s in source]
    return moved.get(source, source) if isinstance(source, int) and source >= 0 else source
