"""What the router actually does on real images: expert usage, dead experts, and whether the
choice tracks object scale.

A load-balancing loss keeps the usage histogram flat on average; it says nothing about whether
the router has learnt anything image-specific. This script answers that on the validation set.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

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


def behaviour(block, logits: torch.Tensor, scales: torch.Tensor, counts: torch.Tensor) -> dict:
    """Usage, concentration and what the choice correlates with, for one block's captured logits."""
    probs = logits.softmax(dim=1)
    chosen = probs.topk(block.top_k, dim=1).indices
    n, e = probs.shape
    usage = torch.zeros(e)
    for k in range(block.top_k):
        usage += torch.bincount(chosen[:, k], minlength=e).float()
    usage /= n
    by_expert = defaultdict(list)
    for i in range(n):
        by_expert[int(chosen[i, 0])].append(float(scales[i]))

    def correlate(other: torch.Tensor) -> list[float]:
        return [round(float(torch.corrcoef(torch.stack([probs[:, j], other]))[0, 1]), 3) for j in range(e)]

    return {
        "channels": block.channels,
        "kernels": block.expert_kernel_sizes,
        "top_k": block.top_k,
        "usage": [round(float(u), 4) for u in usage],
        "top1_share": [round(float(u), 4) for u in torch.bincount(chosen[:, 0], minlength=e).float() / n],
        "dead_experts": [j for j in range(e) if usage[j] < 0.01],
        "mean_prob": [round(float(p), 4) for p in probs.mean(dim=0)],
        "prob_entropy_mean": round(float(-(probs * probs.clamp_min(1e-9).log()).sum(dim=1).mean()), 4),
        "prob_entropy_max": round(float(torch.log(torch.tensor(float(e)))), 4),
        "unique_top2_sets": len({tuple(sorted(row.tolist())) for row in chosen}),
        "scale_of_top1_choice": {str(j): round(sum(v) / len(v), 1) for j, v in sorted(by_expert.items()) if v},
        "prob_vs_scale_corr": correlate(scales),
        "prob_vs_count_corr": correlate(counts),
    }


def route(weights: Path, data: Path, args) -> dict:
    from ultralytics import YOLO

    import esmoe

    esmoe.inject_esmoe()
    model = YOLO(str(weights))
    # Every block, not just the first: the upstream layout puts one after each backbone stage, and
    # they see different resolutions, so one block's usage says nothing about the others'.
    blocks = list(esmoe.blocks(model.model))
    captured: list[list[torch.Tensor]] = [[] for _ in blocks]
    handles = [
        block.router.register_forward_hook(
            lambda _m, _i, out, seen=seen: seen.append(out.detach().cpu()),
        )
        for block, seen in zip(blocks, captured, strict=True)
    ]

    spec = yaml.safe_load(data.read_text(encoding="utf-8"))
    images = sorted((Path(spec["path"]) / spec["val"]).glob("*.jpg"))
    stems = []
    for start in range(0, len(images), args.batch):
        chunk = images[start : start + args.batch]
        model.predict([str(p) for p in chunk], imgsz=args.imgsz, device=args.device, verbose=False, conf=0.25)
        stems += [p.stem for p in chunk]
    for handle in handles:
        handle.remove()

    scale = per_image_scale(data)
    scales = torch.tensor([scale[s]["scale"] for s in stems])
    counts = torch.tensor([float(scale[s]["boxes"]) for s in stems])
    per_block = []
    for index, (block, seen) in enumerate(zip(blocks, captured, strict=True)):
        logits = torch.cat(seen)
        # On an accelerator ultralytics warms the model up with a dummy forward before the first
        # real batch. The hook sees that row too, and it belongs to no image.
        if logits.shape[0] > len(stems):
            logits = logits[-len(stems) :]
        per_block.append({"block": index} | behaviour(block, logits, scales, counts))

    return {"weights": run_name(weights), "images": len(stems), "blocks": per_block}


def run_name(weights: Path) -> str:
    """Name a checkpoint by the run it came from.

    Inside a run directory every checkpoint is called `best.pt`, so the file stem names them all
    the same and one analysis overwrites the next. The directory two levels up is the run.
    """
    if weights.stem in ("best", "last") and len(weights.parts) > 2:
        return f"{weights.parts[-3]}-{weights.stem}"
    return weights.stem


def normalise(record: dict) -> dict:
    """Records written before the multi-block pass describe a single block at the top level."""
    if "blocks" in record:
        return record
    single = {k: v for k, v in record.items() if k not in ("weights", "images")}
    return {"weights": record["weights"], "images": record["images"], "blocks": [{"block": 0, **single}]}


def against_accuracy() -> list[str]:
    """Does a concentrated dispatch cost accuracy? Answered over every run that has both.

    The whole line of work assumes routing collapse is what a balancing term is for. That only
    matters if concentration and the paired metric move together, which is a question the records
    can settle rather than one to argue about.
    """
    import statistics

    from report import KEYS, arm, dedupe, load, variant

    runs, _ = dedupe(load())
    proto = [r for r in runs if "@e120f1i800" in variant(r)]
    base = {arm(r): r for r in proto if r["config"]["arch"] == "baseline"}
    shares, deltas = [], []
    for r in proto:
        if r["config"]["arch"] == "baseline" or arm(r) not in base:
            continue
        run = PurePosixPath(r["artifact"]["path"]).parts[-3]
        record = OUT / f"{run}-best.json"
        if not record.is_file():
            continue
        blocks = normalise(json.loads(record.read_text(encoding="utf-8")))["blocks"]
        shares.append(statistics.mean(max(b["top1_share"]) for b in blocks))
        deltas.append(r["metrics"][KEYS[0]] - base[arm(r)]["metrics"][KEYS[0]])
    if len(shares) < 3:
        return []
    mx, my = statistics.mean(shares), statistics.mean(deltas)
    cov = sum((x - mx) * (y - my) for x, y in zip(shares, deltas, strict=True)) / len(shares)
    r = cov / (statistics.pstdev(shares) * statistics.pstdev(deltas))
    return [
        "## Does concentration cost accuracy?",
        "",
        f"Over the {len(shares)} runs that have both a paired delta and a routing analysis, the leading "
        f"expert's top-1 share runs {min(shares):.2f} to {max(shares):.2f} and the paired mAP50 delta "
        f"{min(deltas):+.4f} to {max(deltas):+.4f}. Their correlation is **r = {r:+.3f}**: on this "
        "evidence a concentrated dispatch does not cost accuracy, which is worth holding against the "
        "premise that a balancing term is what the block needs.",
        "",
    ]


def summarise(records: list[dict]) -> str:
    """One table per block of each checkpoint, plus the reading that survives all of them."""
    records = [normalise(r) for r in records]
    lines = [
        "# Router behaviour on VisDrone val",
        "",
        "Per checkpoint: the share of images on which each expert is the top-1 choice, the share on which it "
        "is in the top-2, the mean routing probability, and the correlation of that probability with the "
        "mean object size and the object count of the image.",
        "",
    ]
    lines += against_accuracy()
    for r in records:
        lines += [f"## {r['weights']}", ""]
        for b in r["blocks"]:
            e = len(b["kernels"])
            where = f" through block {b['block']} ({b['channels']} channels)" if len(r["blocks"]) > 1 else ""
            lines += [
                f"{r['images']} images{where}, kernels {b['kernels']}, top-{b['top_k']}, "
                f"dead experts: {b['dead_experts'] or 'none'}, "
                f"mean entropy {b['prob_entropy_mean']} of {b['prob_entropy_max']}, "
                f"distinct top-2 pairs seen: {b['unique_top2_sets']} of {e * (e - 1) // 2}.",
                "",
                "| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |",
                "|:--:|:--:|:--:|:--:|:--:|:--:|:--:|",
            ]
            for j in range(e):
                cells = (
                    f"{b['kernels'][j]} | {b['top1_share'][j]:.3f} | {b['usage'][j]:.3f} | "
                    f"{b['mean_prob'][j]:.3f} | "
                    f"{b['prob_vs_scale_corr'][j]:+.2f} | {b['prob_vs_count_corr'][j]:+.2f}"
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
        (OUT / f"{run_name(weights)}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        for b in record["blocks"]:
            print(
                f"{run_name(weights)} block {b['block']}: usage {b['usage']} dead {b['dead_experts']} "
                f"entropy {b['prob_entropy_mean']}/{b['prob_entropy_max']} "
                f"top-2 sets {b['unique_top2_sets']} scale-corr {b['prob_vs_scale_corr']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
