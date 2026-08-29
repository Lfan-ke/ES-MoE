"""Config-level injection: graft ESMoE blocks into a stock Ultralytics model.yaml."""

from collections.abc import Iterable

Spot = int | str | Iterable[int]


def _spots(at: Spot, backbone_len: int, total: int) -> list[int]:
    """Normalise ``at`` to original layer indices to insert after."""
    match at:
        case "backbone_end":
            spots = [backbone_len - 1]
        case int() as index:
            spots = [index if index >= 0 else total + index]
        case str() as name:
            raise ValueError(f"unknown insertion point {name!r}; use 'backbone_end', an index, or indices")
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
) -> dict:
    """Insert ESMoE blocks after the given layers and renumber every later reference.

    Layers address earlier layers by absolute index, so an insertion invalidates every reference at
    or past it unless the references move with it.

    Args:
        base: Anything ``yaml_model_load`` accepts, e.g. ``yolo11n.yaml`` or a path.
        out: Optional path to write the grafted config to.
        at: ``"backbone_end"`` (default), a layer index, or several indices.
        num_experts: Experts per block.
        top_k: Experts activated per sample.
    """
    from ultralytics.nn.tasks import yaml_model_load

    d = dict(yaml_model_load(base))
    d.pop("yaml_file", None)
    backbone, head = list(d["backbone"]), list(d["head"])
    spots = _spots(at, len(backbone), len(backbone) + len(head))

    grafted: list[list] = []
    moved: dict[int, int] = {}
    backbone_len = len(backbone)
    for index, layer in enumerate([*backbone, *head]):
        moved[index] = len(grafted)
        grafted.append(list(layer))
        if index in spots:
            grafted.append([-1, 1, "ESMoE", [num_experts, top_k]])
            backbone_len += index < len(backbone)

    renumbered = [[_shift(layer[0], moved), *layer[1:]] for layer in grafted]
    d["backbone"], d["head"] = renumbered[:backbone_len], renumbered[backbone_len:]
    if out:
        from ultralytics.utils import YAML

        YAML.save(out, d)
    return d


def _shift(source, moved: dict[int, int]):
    if isinstance(source, list):
        return [_shift(s, moved) for s in source]
    return moved.get(source, source) if isinstance(source, int) and source >= 0 else source
