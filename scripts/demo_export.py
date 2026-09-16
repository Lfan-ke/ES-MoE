"""Export the models the docs site runs in the browser: a detector plus, for a routed model, each
block's router probabilities.

A checkpoint trained on YOLO-Master's fork needs the fork's classes to unpickle, so it comes over in
two steps:

    PYTHONPATH=<fork> uv run python scripts/demo_export.py transfer --weights <fork.pt> --out <fork.state.pt>
    uv run --with onnx --with onnxruntime python scripts/demo_export.py export --out <pages>/models --weights <dir>

Every model comes from a checkpoint that already exists; nothing here trains.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO

import esmoe

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "coco-yolo11n": {
        "group": "coco",
        "label": {"zh": "YOLO11n，无块", "en": "YOLO11n, no blocks"},
        "imgsz": 640,
    },
    "coco-esmoe-n": {
        "group": "coco",
        "label": {"zh": "YOLO-Master-EsMoE-N", "en": "YOLO-Master-EsMoE-N"},
        "imgsz": 640,
        "config": "configs/yolo-master-n-e3k3-esmoe.yaml",
    },
    "drone-baseline": {
        "group": "drone",
        "label": {"zh": "无块基线", "en": "Baseline, no blocks"},
        "imgsz": 800,
    },
    "drone-esmoe": {
        "group": "drone",
        "label": {"zh": "四块，本包", "en": "Four blocks, this package"},
        "imgsz": 800,
    },
    "drone-upstream": {
        "group": "drone",
        "label": {"zh": "四块，上游配方", "en": "Four blocks, upstream recipe"},
        "imgsz": 800,
        "config": "configs/yolo-master-n-esmoe.yaml",
    },
}

SOURCES = {
    "coco-yolo11n": "yolo11n.pt",
    "coco-esmoe-n": "YOLO-Master-EsMoE-N.state.pt",
    "drone-baseline": "yolo-master-n-baseline-e120-s0-p800f-fp32-last.pt",
    "drone-esmoe": "yolo-master-n-esmoe-upstream-w1-e120-s0-p800f-fp32-last.pt",
    "drone-upstream": "yolo-master-n-upstream-e120-s0-p800f-fp32-fork-last.state.pt",
}


def kernels(block) -> list[int]:
    return [int(expert.dw.kernel_size[0]) for expert in block.experts]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def entry(model_id: str, spec: dict, blocks: list, names: dict, source: Path, origin: str | None = None) -> dict:
    return {
        "id": model_id,
        "group": spec["group"],
        "label": spec["label"],
        "imgsz": spec["imgsz"],
        "classes": [names[index] for index in sorted(names)],
        "source": origin or source.name,
        "sha256": digest(source),
        "blocks": [{"experts": block.num_experts, "top_k": block.top_k, "kernels": kernels(block)} for block in blocks],
    }


class Gated(torch.nn.Module):
    """The detector plus one output per block: that block's router probabilities."""

    routed: list

    def __init__(self, net: torch.nn.Module, blocks: list) -> None:
        super().__init__()
        self.net = net
        # A plain attribute: the blocks already belong to `net`, and registering them again would
        # export their weights twice.
        object.__setattr__(self, "routed", blocks)
        self.seen: list[torch.Tensor] = []
        for block in blocks:
            block.register_forward_pre_hook(lambda _module, inputs: self.seen.append(inputs[0]))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self.seen.clear()
        prediction = self.net(images)
        prediction = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        # The router is a pooling and two small layers; running it again beside the block keeps the
        # block's own forward untouched, so the exported detections stay the checkpoint's.
        probs = [
            F.softmax(block.router(seen).float().clamp(-30.0, 30.0), dim=1).flatten(1)
            for block, seen in zip(self.routed, self.seen, strict=True)
        ]
        return (prediction, *probs)


def slim(target: Path) -> int:
    """A block accumulates into a zero tensor, and the tracer writes that tensor out in full: four of
    them at 800 pixels are 17 MB of zeros in a file the page has to download. Keep the shape instead."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(target))
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.attribute or node.attribute[0].type != onnx.AttributeProto.TENSOR:
            continue
        tensor = node.attribute[0].t
        if tensor.ByteSize() < 100_000 or not (numpy_helper.to_array(tensor) == 0).all():
            continue
        shape = numpy_helper.from_array(np.array(tensor.dims, dtype=np.int64), f"{node.name}/shape")
        model.graph.initializer.append(shape)
        zero = numpy_helper.from_array(np.zeros(1, dtype=numpy_helper.to_array(tensor).dtype))
        node.CopyFrom(helper.make_node("ConstantOfShape", [shape.name], list(node.output), name=node.name, value=zero))
        replaced += 1
    if replaced:
        onnx.save(model, str(target))
    return replaced


def check(target: Path, images: torch.Tensor, reference: tuple, names: list[str]) -> None:
    """A demo whose routing differs from the checkpoint would be worse than no demo."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"images": images.numpy()})
    for name, got, want in zip(names, outputs, reference, strict=True):
        want = want.numpy()
        gap = float(np.abs(got - want).max())
        # Probabilities live in [0, 1] and are compared as they are; decoded boxes carry the image's
        # scale, so a pixel-level difference there is relative to it.
        scale = 1.0 if name.startswith("probs") else max(1.0, float(np.abs(want).max()))
        if gap / scale > 1e-4:
            raise SystemExit(f"{target.name}: {name} differs from the checkpoint by {gap:.2e}")


def load(model_id: str, weights: Path) -> tuple[torch.nn.Module, str]:
    esmoe.inject_esmoe()
    if weights.name.endswith(".state.pt"):
        from recipe_parity import renamed
        from ultralytics.nn.tasks import DetectionModel

        saved = torch.load(weights, map_location="cpu", weights_only=False)
        net = DetectionModel(str(ROOT / MODELS[model_id]["config"]), ch=3, nc=len(saved["names"]), verbose=False)
        # The fork keeps the auxiliary term's running magnitude in the checkpoint; it belongs to its
        # trainer, not to the blocks, and inference never reads it.
        state = {key: value for key, value in renamed(saved["state"]).items() if "_mixture_loss_ema" not in key}
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(f"{model_id}: missing {missing[:3]}, unexpected {unexpected[:3]}")
        net.names = saved["names"]
        return net.float().eval(), saved["source"]
    return YOLO(str(weights)).model.float().eval(), weights.name


def export(model_id: str, weights: Path, out: Path) -> dict:
    spec = MODELS[model_id]
    net, origin = load(model_id, weights)
    blocks = list(esmoe.blocks(net))
    wrapper = Gated(net, blocks).eval()
    images = torch.rand(1, 3, spec["imgsz"], spec["imgsz"])
    # Not inference_mode: the head caches anchors, and a tensor made there cannot be traced later.
    with torch.no_grad():
        reference = wrapper(images)
    names = ["pred", *[f"probs{index}" for index in range(len(blocks))]]
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{model_id}.onnx"
    torch.onnx.export(
        wrapper,
        (images,),
        str(target),
        input_names=["images"],
        output_names=names,
        opset_version=17,
        dynamo=False,
    )
    slim(target)
    check(target, images, reference, names)
    names: dict = net.names
    record = entry(model_id, spec, blocks, names, weights, origin)
    parameters = sum(p.numel() for p in net.parameters())
    return record | {"file": target.name, "bytes": target.stat().st_size, "parameters": parameters}


def transfer(args) -> None:
    """Carry a fork-trained checkpoint over as plain tensors, which official ultralytics can open."""
    net = YOLO(str(args.weights)).model.float().eval()
    torch.save({"state": net.state_dict(), "names": net.names, "source": args.weights.name}, args.out)
    print(f"{args.out.name}: {sum(p.numel() for p in net.parameters())} parameters, {len(net.names)} classes")


def publish(args) -> None:
    listing = args.out / "index.json"
    index = json.loads(listing.read_text(encoding="utf-8")) if listing.exists() else []
    index = [record for record in index if record["id"] not in args.only]
    for model_id in args.only:
        record = export(model_id, args.weights / SOURCES[model_id], args.out)
        index.append(record)
        size = record["bytes"] / 1e6
        print(f"{model_id}: {size:.1f} MB, {len(record['blocks'])} blocks, {len(record['classes'])} classes")
        # Written per model: a later export failing should not take the finished ones with it.
        index.sort(key=lambda record: list(MODELS).index(record["id"]))
        listing.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("transfer", help="save a fork checkpoint's tensors")
    one.add_argument("--weights", type=Path, required=True)
    one.add_argument("--out", type=Path, required=True)
    one.set_defaults(run=transfer)
    every = commands.add_parser("export", help="export the models the site serves")
    every.add_argument("--out", type=Path, required=True, help="directory the site serves as /models")
    every.add_argument("--weights", type=Path, required=True, help="directory holding the checkpoints")
    every.add_argument("--only", nargs="*", default=list(MODELS), help="model ids to export")
    every.set_defaults(run=publish)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
