"""Describe the dataset every protocol run trains on: images, boxes, resolutions and box sizes per split.

    uv run python scripts/dataset.py /data/ds/VisDrone_dataset   # an unpacked copy
    uv run python scripts/dataset.py VisDrone_dataset.zip          # or the archive as downloaded

Writes results/dataset.json (or `--out`), which scripts/charts.py draws on the experiments page. Box
sizes are read against each image's own resolution, so the COCO buckets here are the ones
results/buckets.md evaluates on, and the side at 800 px is what the protocol's letterbox actually
hands the model.
"""

import argparse
import bisect
import json
import math
import statistics
import zipfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dataset.json"
SPLITS = ("train", "val", "test")
IMGSZ = 800
EDGES = (0, 8, 16, 24, 32, 48, 64, 96, 128, 256, 801)


def entries(source: Path, stack: ExitStack):
    """(path inside the dataset, bytes, opener) for every file, from a directory or a zip alike."""
    if source.suffix.lower() == ".zip":
        archive = stack.enter_context(zipfile.ZipFile(source))
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            # Archives packed on macOS carry resource forks that look like images and are not.
            if info.is_dir() or "__MACOSX" in path.parts or path.name.startswith("._"):
                continue
            yield path, info.file_size, lambda info=info: archive.open(info)
    else:
        for path in source.rglob("*"):
            if path.is_file():
                inside = PurePosixPath(path.relative_to(source).as_posix())
                yield inside, path.stat().st_size, lambda path=path: path.open("rb")


def bin_of(side: float) -> int:
    """Index into the side histogram. A box a label rounds past the image edge still lands in the top bin."""
    return min(max(bisect.bisect_right(EDGES, side), 1), len(EDGES) - 1)


def describe(source: Path) -> dict:
    images: dict[str, dict] = {split: {} for split in SPLITS}
    labels: dict[str, dict] = {split: {} for split in SPLITS}
    with ExitStack() as stack:
        for path, size, opener in entries(source, stack):
            if len(path.parts) < 3 or path.parts[-2] not in SPLITS:
                continue
            split, kind, suffix = path.parts[-2], path.parts[-3], path.suffix.lower()
            if kind == "images" and suffix in {".jpg", ".jpeg", ".png"}:
                with opener() as handle, Image.open(handle) as image:
                    images[split][path.stem] = (image.size, size)
            elif kind == "labels" and suffix == ".txt":
                with opener() as handle:
                    labels[split][path.stem] = handle.read().decode()

    splits = {}
    for split in SPLITS:
        classes, buckets, bins, sides, fitted = Counter(), Counter(), Counter(), [], []
        for stem, ((width, height), _) in images[split].items():
            fit = IMGSZ / max(width, height)
            for line in labels[split].get(stem, "").splitlines():
                if not line.strip():
                    continue
                cls, _, _, w, h = line.split()[:5]
                classes[int(cls)] += 1
                side = math.sqrt(float(w) * width * float(h) * height)
                buckets["small" if side < 32 else "medium" if side < 96 else "large"] += 1
                sides.append(side)
                fitted.append(side * fit)
                bins[bin_of(side * fit)] += 1
        splits[split] = {
            "images": len(images[split]),
            "boxes": len(sides),
            "bytes": sum(size for _, size in images[split].values()),
            "resolutions": [[w, h, n] for (w, h), n in Counter(s for s, _ in images[split].values()).most_common()],
            "classes": [classes[i] for i in range(max(classes, default=-1) + 1)],
            "area": {name: buckets[name] for name in ("small", "medium", "large")},
            "side_at_800": {"edges": list(EDGES), "counts": [bins[i] for i in range(1, len(EDGES))]},
            "median_side": {
                "original": round(statistics.median(sides), 1) if sides else None,
                "at_800": round(statistics.median(fitted), 1) if fitted else None,
            },
        }
    return splits


def class_names(data: Path) -> list[str]:
    """Ultralytics accepts `names` as a list or as an index-to-name mapping."""
    names = yaml.safe_load(data.read_text(encoding="utf-8"))["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="the unpacked dataset root or its zip")
    parser.add_argument("--data", type=Path, default=ROOT / "configs" / "visdrone.yaml")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    record = {
        "name": "VisDrone2019-DET",
        "imgsz": IMGSZ,
        "names": class_names(args.data),
        "splits": describe(args.source),
    }
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for split, facts in record["splits"].items():
        counts = f"{facts['images']:>5} images {facts['boxes']:>7} boxes"
        print(f"{split:<5} {counts}  {facts['area']}  median side {facts['median_side']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
