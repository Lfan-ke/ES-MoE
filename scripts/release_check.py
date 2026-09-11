"""The model YOLO-Master released for its paper, evaluated as released and as rebuilt with this package.

YOLO-Master-EsMoE-N (release YOLO-Master-v26.02) records mAP50-95 0.4269 on COCO val2017; the paper
reports 42.4. It is yolo-master-n with three experts per block, all three active. This evaluates that
checkpoint on YOLO-Master's own fork, then carries the same weights onto `ESMoE` blocks in official
ultralytics and evaluates again, on the same images with the same settings. Both run on CPU in FP32.

    PYTHONPATH=<fork> python scripts/release_check.py fork --weights YOLO-Master-EsMoE-N.pt --coco /data/coco/coco
    python scripts/release_check.py package --out release                          # official + esmoe
    python scripts/release_check.py compare --out release
"""

import argparse
import json
from pathlib import Path

import yaml
from recipe_parity import renamed, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")


def data_yaml(coco: Path, names: dict, out: Path) -> Path:
    path = out / "coco-val.yaml"
    spec = {"path": str(coco), "train": "val2017.txt", "val": "val2017.txt", "names": dict(names)}
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return path


def evaluate(model, data: Path, args) -> dict:
    metrics = model.val(data=str(data), imgsz=640, batch=args.batch, device="cpu", workers=args.workers, plots=False)
    return {key: float(metrics.results_dict[key]) for key in KEYS}


def fork(args) -> None:
    """On YOLO-Master's fork: the checkpoint as released."""
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    core = model.model
    torch.save(core.float().state_dict(), out / "released.state.pt")
    data = data_yaml(Path(args.coco), core.names, out)
    record = {
        "ultralytics": ultralytics.__file__,
        "parameters": sum(p.numel() for p in core.parameters()),
        "blocks": [(type(m).__name__, getattr(m, "num_experts", None), getattr(m, "top_k", None)) for m in core.model],
        "metrics": evaluate(model, data, args),
    }
    (out / "fork.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps(record["metrics"], indent=1))


def package(args) -> None:
    """On official ultralytics: the same weights on ESMoE blocks."""
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    import esmoe

    torch.set_num_threads(args.threads)
    esmoe.inject_esmoe()
    out = Path(args.out)
    model = DetectionModel(str(ROOT / "configs" / "yolo-master-n-e3k3-esmoe.yaml"), ch=3, nc=80, verbose=False)
    state = renamed(torch.load(out / "released.state.pt", map_location="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"missing {missing[:5]}, unexpected {unexpected[:5]}")
    model.names = yaml.safe_load((out / "coco-val.yaml").read_text(encoding="utf-8"))["names"]
    save_checkpoint(model, out / "package.pt")
    wrapped = YOLO(str(out / "package.pt"))
    record = {
        "ultralytics": ultralytics.__file__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "blocks": [b.spec() | {"num_experts": b.num_experts, "top_k": b.top_k} for b in esmoe.blocks(model)][:1],
        "metrics": evaluate(wrapped, out / "coco-val.yaml", args),
    }
    (out / "package.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps(record["metrics"], indent=1))


def compare(args) -> None:
    out = Path(args.out)
    fork_record = json.loads((out / "fork.json").read_text(encoding="utf-8"))
    package_record = json.loads((out / "package.json").read_text(encoding="utf-8"))
    lines = [
        "# YOLO-Master-EsMoE-N on COCO val2017: as released and rebuilt with esmoe",
        "",
        f"Parameters: {fork_record['parameters']:,} and {package_record['parameters']:,}.",
        "",
        "| metric | YOLO-Master fork | esmoe on official ultralytics | gap | recorded in the checkpoint |",
        "|:--:|:--:|:--:|:--:|:--:|",
    ]
    recorded = {"metrics/mAP50(B)": 0.58844, "metrics/mAP50-95(B)": 0.4269}
    for key in KEYS:
        a, b = fork_record["metrics"][key], package_record["metrics"][key]
        lines.append(f"| {key} | {a:.5f} | {b:.5f} | {b - a:+.5f} | {recorded.get(key, '-')} |")
    (out / "release.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("fork", "package", "compare"))
    p.add_argument("--weights", default="YOLO-Master-EsMoE-N.pt")
    p.add_argument("--coco", default="/data/coco/coco")
    p.add_argument("--out", default="release")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=48)
    args = p.parse_args()
    {"fork": fork, "package": package, "compare": compare}[args.step](args)


if __name__ == "__main__":
    main()
