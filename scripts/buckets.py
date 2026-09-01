"""COCO-style area buckets for VisDrone val, which the official protocol does not report.

VisDrone's own evaluation gives AP, AP50, AP75 and AR@1/10/100/500 with no size split, so the
buckets here follow the COCO definition - small < 32^2, medium 32^2..96^2, large >= 96^2 - measured
on ground-truth boxes in the original image. maxDets is 500, the VisDrone figure, not COCO's 100.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "buckets"
OUT = ROOT / "results" / "buckets"
STATS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR", "ARs", "ARm", "ARl")
SMALL, MEDIUM = 32**2, 96**2


def ground_truth(data: Path) -> tuple[dict, dict[str, int]]:
    """A COCO annotation dict built from the YOLO labels, plus the stem -> image id map."""
    spec = yaml.safe_load(data.read_text(encoding="utf-8"))
    base = Path(spec["path"])
    images_dir = base / spec["val"]
    labels_dir = base / str(spec["val"]).replace("images", "labels", 1)
    images, annotations, ids = [], [], {}
    for index, image_path in enumerate(sorted(images_dir.glob("*.jpg")), start=1):
        width, height = Image.open(image_path).size
        images.append({"id": index, "file_name": image_path.name, "width": width, "height": height})
        ids[image_path.stem] = index
        label = labels_dir / f"{image_path.stem}.txt"
        if not label.exists():
            continue
        for line in label.read_text(encoding="utf-8").splitlines():
            cls, cx, cy, w, h = (float(v) for v in line.split()[:5])
            w, h = w * width, h * height
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": index,
                    # ultralytics numbers classes from 1 in predictions.json for non-COCO data
                    "category_id": int(cls) + 1,
                    "bbox": [cx * width - w / 2, cy * height - h / 2, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
    categories = [{"id": int(k) + 1, "name": v} for k, v in spec["names"].items()]
    return {"images": images, "annotations": annotations, "categories": categories}, ids


def bucket_counts(gt: dict) -> dict[str, int]:
    counts = {"small": 0, "medium": 0, "large": 0}
    for ann in gt["annotations"]:
        counts["small" if ann["area"] < SMALL else "medium" if ann["area"] < MEDIUM else "large"] += 1
    return counts


def evaluate(weights: Path, data: Path, gt_file: Path, ids: dict[str, int], args) -> dict:
    from faster_coco_eval import COCO, COCOeval_faster
    from ultralytics import YOLO

    import esmoe

    esmoe.inject_esmoe()
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        max_det=args.max_det,
        conf=0.001,
        save_json=True,
        plots=False,
        verbose=False,
        workers=0,
        project=str(RUNS),
        name=weights.stem,
        exist_ok=True,
    )
    predictions = json.loads((Path(metrics.save_dir) / "predictions.json").read_text(encoding="utf-8"))
    for p in predictions:
        p["image_id"] = ids[str(p["image_id"])]
    coco = COCO(str(gt_file))
    ev = COCOeval_faster(coco, coco.loadRes(predictions), "bbox")
    ev.params.maxDets = [1, 10, args.max_det]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {
        "weights": weights.name,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "ultralytics": {"mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map)},
        "coco": dict(zip(STATS, (float(v) for v in ev.stats), strict=True)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "configs" / "visdrone.yaml")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    gt, ids = ground_truth(args.data)
    gt_file = OUT / "val_gt.json"
    gt_file.write_text(json.dumps(gt), encoding="utf-8")
    counts = bucket_counts(gt)
    print(f"val: {len(gt['images'])} images, {len(gt['annotations'])} boxes, {counts}", flush=True)

    for weights in args.weights:
        record = evaluate(weights, args.data, gt_file, ids, args) | {"gt_buckets": counts}
        (OUT / f"{weights.stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        c = record["coco"]
        print(
            f"{weights.stem}: AP {c['AP']:.4f} AP50 {c['AP50']:.4f} APs {c['APs']:.4f} APm {c['APm']:.4f} "
            f"APl {c['APl']:.4f} AR@{args.max_det} {c['AR']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
