"""Config-level injection: graft ESMoE blocks into a stock Ultralytics model.yaml."""

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
        **options: Block settings to write into the config - ``balance``, ``out_norm``,
            ``dense_training``. They belong in the config because the trainer rebuilds the
            model from it, discarding anything set on the instance beforehand.
    """
    from ultralytics.nn.tasks import yaml_model_load

    from .module import BALANCES, SETTINGS

    if unknown := set(options) - set(SETTINGS):
        raise ValueError(f"unknown ESMoE options {sorted(unknown)}; expected {SETTINGS}")
    options = dict(options)
    if callable(balance := options.get("balance")):
        # A config holds names, not functions. The shipped objectives have names; a custom one has
        # to be set on the blocks, and cannot survive the trainer rebuilding the model from here.
        named = {fn: name for name, fn in BALANCES.items()}
        if balance not in named:
            raise ValueError(f"{balance.__name__} is not a named objective; pass one of {sorted(BALANCES)}")
        options["balance"] = named[balance]
    args = [num_experts, top_k] + ([None, options] if options else [])
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
            grafted.append([-1, 1, "ESMoE", list(args)])
            backbone_len += index < len(backbone)

    targets = moved | {s: moved[s] + 1 for s in spots} if rewire else moved
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
