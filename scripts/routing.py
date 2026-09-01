"""What the router actually does on real images: expert usage, dead experts, and whether the
choice tracks object scale.

A load-balancing loss keeps the usage histogram flat on average; it says nothing about whether
the router has learnt anything image-specific. This script answers that on the validation set.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "routing"


def per_image_scale(data: Path) -> dict[str, dict[str, float]]:
    """Mean sqrt(box area) in original pixels and the box count, per validation image."""
    spec = yaml.safe_load(data.read_text(encoding="utf-8"))
    base = Path(spec["path"])
    labels = base / str(spec["val"]).replace("images", "labels", 1)
    stats = {}
    for image in sorted((base / spec["val"]).glob("*.jpg")):
        width, height = Image.open(image).size
        sizes = []
        label = labels / f"{image.stem}.txt"
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                _, _, _, w, h = (float(v) for v in line.split()[:5])
                sizes.append(((w * width) * (h * height)) ** 0.5)
        stats[image.stem] = {"boxes": len(sizes), "scale": sum(sizes) / len(sizes) if sizes else 0.0}
    return stats


def route(weights: Path, data: Path, args) -> dict:
    from ultralytics import YOLO

    import esmoe

    esmoe.inject_esmoe()
    model = YOLO(str(weights))
    block = next(esmoe.blocks(model.model))
    captured: list[torch.Tensor] = []
    handle = block.router.register_forward_hook(lambda _m, _i, out: captured.append(out.detach().cpu()))

    spec = yaml.safe_load(data.read_text(encoding="utf-8"))
    images = sorted((Path(spec["path"]) / spec["val"]).glob("*.jpg"))
    stems = []
    for start in range(0, len(images), args.batch):
        chunk = images[start : start + args.batch]
        model.predict([str(p) for p in chunk], imgsz=args.imgsz, device=args.device, verbose=False, conf=0.25)
        stems += [p.stem for p in chunk]
    handle.remove()

    logits = torch.cat(captured)
    probs = logits.softmax(dim=1)
    chosen = probs.topk(block.top_k, dim=1).indices
    n, e = probs.shape
    usage = torch.zeros(e)
    for k in range(block.top_k):
        usage += torch.bincount(chosen[:, k], minlength=e).float()
    usage /= n
    top1 = torch.bincount(chosen[:, 0], minlength=e).float() / n

    scale = per_image_scale(data)
    scales = torch.tensor([scale[s]["scale"] for s in stems])
    counts = torch.tensor([float(scale[s]["boxes"]) for s in stems])
    by_expert = defaultdict(list)
    for i in range(n):
        by_expert[int(chosen[i, 0])].append(float(scales[i]))
    corr = [float(torch.corrcoef(torch.stack([probs[:, j], scales]))[0, 1]) for j in range(e)]

    return {
        "weights": weights.name,
        "images": n,
        "kernels": block.expert_kernel_sizes,
        "top_k": block.top_k,
        "usage": [round(float(u), 4) for u in usage],
        "top1_share": [round(float(u), 4) for u in top1],
        "dead_experts": [j for j in range(e) if usage[j] < 0.01],
        "mean_prob": [round(float(p), 4) for p in probs.mean(dim=0)],
        "prob_entropy_mean": round(float(-(probs * probs.clamp_min(1e-9).log()).sum(dim=1).mean()), 4),
        "prob_entropy_max": round(float(torch.log(torch.tensor(float(e)))), 4),
        "unique_top2_sets": len({tuple(sorted(row.tolist())) for row in chosen}),
        "scale_of_top1_choice": {str(j): round(sum(v) / len(v), 1) for j, v in sorted(by_expert.items()) if v},
        "prob_vs_scale_corr": [round(c, 3) for c in corr],
        "prob_vs_count_corr": [
            round(float(torch.corrcoef(torch.stack([probs[:, j], counts]))[0, 1]), 3) for j in range(e)
        ],
    }


def summarise(records: list[dict]) -> str:
    """One table per checkpoint plus the reading that survives all of them."""
    lines = [
        "# Router behaviour on VisDrone val",
        "",
        "Per checkpoint: the share of images on which each expert is the top-1 choice, the share on which it "
        "is in the top-2, the mean routing probability, and the correlation of that probability with the "
        "mean object size and the object count of the image.",
        "",
    ]
    for r in records:
        e = len(r["kernels"])
        lines += [
            f"## {r['weights']}",
            "",
            f"{r['images']} images, kernels {r['kernels']}, top-{r['top_k']}, "
            f"dead experts: {r['dead_experts'] or 'none'}, "
            f"mean entropy {r['prob_entropy_mean']} of {r['prob_entropy_max']}, "
            f"distinct top-2 pairs seen: {r['unique_top2_sets']} of {e * (e - 1) // 2}.",
            "",
            "| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |",
            "|:--:|:--:|:--:|:--:|:--:|:--:|:--:|",
        ]
        for j in range(e):
            cells = (
                f"{r['kernels'][j]} | {r['top1_share'][j]:.3f} | {r['usage'][j]:.3f} | {r['mean_prob'][j]:.3f} | "
                f"{r['prob_vs_scale_corr'][j]:+.2f} | {r['prob_vs_count_corr'][j]:+.2f}"
            )
            lines.append(f"| {j} | {cells} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", nargs="*", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "configs" / "visdrone.yaml")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summarise", action="store_true", help="only rebuild results/routing.md from saved records")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.summarise:
        records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*.json"))]
        target = ROOT / "results" / "routing.md"
        target.write_text(summarise(records), encoding="utf-8")
        print(f"{len(records)} records -> {target}")
        return 0
    for weights in args.weights:
        record = route(weights, args.data, args)
        (OUT / f"{weights.stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(
            f"{weights.stem}: usage {record['usage']} dead {record['dead_experts']} "
            f"entropy {record['prob_entropy_mean']}/{record['prob_entropy_max']} "
            f"top-2 sets {record['unique_top2_sets']} scale-corr {record['prob_vs_scale_corr']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
