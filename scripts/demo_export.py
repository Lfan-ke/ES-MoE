"""Export the models the docs site runs in the browser: a detector plus, for a routed model, each
block's router probabilities.

    uv run --with onnx --with onnxruntime python scripts/demo_export.py --out <pages>/models --weights <dir>

Every model comes from a checkpoint that already exists; nothing here trains.
"""

import hashlib
from pathlib import Path

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
    },
}

SOURCES = {
    "coco-yolo11n": "yolo11n.pt",
    "coco-esmoe-n": "YOLO-Master-EsMoE-N.pt",
    "drone-baseline": "yolo-master-n-baseline-e120-s0-p800f-fp32-last.pt",
    "drone-esmoe": "yolo-master-n-esmoe-upstream-w1-e120-s0-p800f-fp32-last.pt",
    "drone-upstream": "yolo-master-n-upstream-e120-s0-p800f-fp32-fork-last.pt",
}


def kernels(block) -> list[int]:
    return [int(expert.dw.kernel_size[0]) for expert in block.experts]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def entry(model_id: str, spec: dict, blocks: list, names: dict, source: Path) -> dict:
    return {
        "id": model_id,
        "group": spec["group"],
        "label": spec["label"],
        "imgsz": spec["imgsz"],
        "classes": [names[index] for index in sorted(names)],
        "source": source.name,
        "sha256": digest(source),
        "blocks": [{"experts": block.num_experts, "top_k": block.top_k, "kernels": kernels(block)} for block in blocks],
    }
