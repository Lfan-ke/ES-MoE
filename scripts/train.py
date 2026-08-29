"""Budget-fair ESMoE on/off runner.

One process = one experiment record. Baseline and ESMoE runs differ only in the grafted block,
so any metric gap is attributable; every field the task book's experiment record asks for is
written to results/<experiment_id>.json.
"""

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params

import esmoe

ROOT = Path(__file__).resolve().parents[1]

# Locked by the E1 task book: acceptance runs against the release tag, research against HEAD.
YOLO_MASTER_RELEASE = "YOLO-Master-v26.08 @ 43d4011"
YOLO_MASTER_HEAD = "57b9ea3"


def git_ref():
    """Identify the exact toolkit revision a record came from.

    A run launched from a copied tree has no git metadata, so the launcher can pass the revision
    in ESMOE_GIT_REF rather than let the record claim a version it cannot prove.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get("ESMOE_GIT_REF", "unversioned-copy")


def build(args):
    if not args.esmoe:
        return YOLO(args.base), args.base
    esmoe.inject_esmoe()
    cfg = ROOT / "configs" / f"{Path(args.base).stem}-esmoe-e{args.num_experts}k{args.top_k}.yaml"
    esmoe.graft(args.base, out=str(cfg), num_experts=args.num_experts, top_k=args.top_k)
    model = YOLO(str(cfg))
    esmoe.attach_aux_loss(model, weight=args.aux_weight)
    return model, str(cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="yolov8n.yaml")
    p.add_argument("--data", default=str(ROOT / "configs" / "visdrone.yaml"))
    p.add_argument("--esmoe", action="store_true")
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--aux-weight", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    model, cfg = build(args)
    arch = "esmoe" if args.esmoe else "baseline"
    name = f"{Path(args.base).stem}-{arch}-e{args.epochs}-s{args.seed}{args.tag}"
    experiment_id = f"{name}-{time.strftime('%Y%m%d%H%M%S')}"

    started = time.time()
    status, error = "success", None
    try:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            pretrained=False,
            amp=bool(args.amp),
            fraction=args.fraction,
            plots=False,
            val=True,
            project=str(ROOT / "runs"),
            name=name,
            exist_ok=True,
        )
    except Exception as exc:  # a failed run is still a record, not a silent gap
        status, error = "failed", repr(exc)
    elapsed = time.time() - started

    metrics = {}
    trainer = getattr(model, "trainer", None)
    if trainer is not None and getattr(trainer, "metrics", None):
        metrics = {k: float(v) for k, v in trainer.metrics.items() if isinstance(v, (int, float))}

    record = {
        "experiment_id": experiment_id,
        "git_ref": {
            "toolkit": git_ref(),
            "toolkit_version": esmoe.__version__,
            "ultralytics": ultralytics.__version__,
            "yolo_master_release": YOLO_MASTER_RELEASE,
            "yolo_master_head": YOLO_MASTER_HEAD,
        },
        "config": {
            "model_yaml": cfg,
            "arch": arch,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "aux_weight": args.aux_weight if args.esmoe else 0.0,
        },
        "dataset": {"yaml": args.data, "fraction": args.fraction},
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "budget": {
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "amp": bool(args.amp),
            "wall_seconds": round(elapsed, 1),
        },
        "seed": args.seed,
        "metrics": metrics,
        "params": get_num_params(model.model),
        "artifact": str(ROOT / "runs" / name / "weights" / "best.pt"),
        "status": status,
        "limitation": error or "single machine, single GPU; see limitations.md",
    }
    out = ROOT / "results" / f"{experiment_id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
