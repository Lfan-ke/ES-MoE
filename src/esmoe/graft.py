"""Config-level injection: graft an ESMoE block onto a stock Ultralytics model.yaml."""


def _shift(source, at):
    if isinstance(source, list):
        return [_shift(s, at) for s in source]
    return source + 1 if isinstance(source, int) and source >= at else source


def graft(base="yolov8n.yaml", out=None, num_experts=4, top_k=2):
    """Append an ESMoE block to the end of the backbone and renumber the head.

    Head layers address earlier layers by absolute index, so inserting a layer invalidates every
    reference at or past the insertion point unless they are shifted with it.
    """
    from ultralytics.nn.tasks import yaml_model_load

    d = yaml_model_load(base)
    d.pop("yaml_file", None)
    at = len(d["backbone"])
    d["backbone"] = [*d["backbone"], [-1, 1, "ESMoE", [num_experts, top_k]]]
    d["head"] = [[_shift(layer[0], at), *layer[1:]] for layer in d["head"]]
    if out:
        from ultralytics.utils import YAML

        YAML.save(out, d)
    return d
