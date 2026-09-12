"""Budget-fair ESMoE on/off runner.

One process = one experiment record. Baseline and ESMoE runs differ only in the grafted block,
so any metric gap is attributable; version, environment, budget, seed, metrics and artifact path
all land in results/<experiment_id>.json.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params

import esmoe

ROOT = Path(__file__).resolve().parents[1]

# Public baseline locked by the 2026-08-23 increment-acceptance rules: main at 2026-08-21
# 23:59:59 (UTC+8). The v26.08 tag only documents the release the library version comes from.
YOLO_MASTER_BASE_REF = "acce839c7e895d6b179de7f7093fa879e237cc7b"
YOLO_MASTER_RELEASE = "YOLO-Master-v26.08 @ 43d40117c30811204fb9347efeabddce15f11a62"


def revision(path: Path, env: str) -> str:
    """The short git revision of a checkout, or what ``env`` says when the tree carries no git metadata."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get(env, "unversioned-copy")


def git_ref():
    """Identify the exact toolkit revision a record came from.

    A run launched from a copied tree has no git metadata, so the launcher can pass the revision
    in ESMOE_GIT_REF rather than let the record claim a version it cannot prove.
    """
    return revision(ROOT, "ESMOE_GIT_REF")


def framework() -> str:
    """Which ultralytics trains the run: the official package, or YOLO-Master's fork at a revision.

    Both call themselves 8.4.101, so the version string cannot tell them apart; the fork's mixture
    registry can.
    """
    if importlib.util.find_spec("ultralytics.nn.mixture_registry") is None:
        return "ultralytics"
    return f"yolo-master@{revision(Path(ultralytics.__file__).resolve().parents[1], 'YOLO_MASTER_REF')}"


def digest(path: Path) -> str | None:
    """sha256 of a file, or None when it is not there to hash."""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_facts(data: str) -> dict:
    """Name, split sizes and label counts, so a record says what it was measured on."""
    spec = yaml.safe_load(Path(data).read_text(encoding="utf-8"))
    base = Path(spec.get("path", "."))
    splits = {}
    for split in ("train", "val", "test"):
        folder = base / spec[split] if split in spec else None
        if folder and folder.is_dir():
            splits[split] = sum(1 for _ in folder.glob("*.jpg"))
    return {"name": base.name or str(base), "classes": len(spec.get("names", {})), "splits": splits}


def requested(args) -> dict:
    """The block settings this run asks for, in the names `esmoe.SETTINGS` uses."""
    return {
        "balance": args.balance,
        "out_norm": args.out_norm,
        "dense_training": args.dense_training,
        "sparse_inference": not args.dense_inference,
        "dynamic_threshold": args.dynamic_threshold,
    }


def block_facts(model) -> dict:
    """The block settings the trained model actually has, read off it rather than off the flags.

    A record that repeats the request cannot catch a setting that failed to reach the model, which
    is how three arms trained the default objective while their records named another.
    """
    found = list(esmoe.blocks(model.model))
    if found:
        return found[0].spec() | {"blocks": len(found), "num_experts": found[0].num_experts, "top_k": found[0].top_k}
    theirs = [m for m in model.model.modules() if type(m).__name__ == "ES_MOE"]
    if theirs:
        # YOLO-Master's block fixes in code what this package leaves to settings: the balance term
        # on the gate at coefficient 1.0, the output norm and every expert run in training.
        first = theirs[0]
        return {
            "balance": "gshard",
            "out_norm": True,
            "dense_training": True,
            "sparse_inference": bool(first.use_sparse_inference),
            "dynamic_threshold": float(first.dynamic_threshold),
            "blocks": len(theirs),
            "num_experts": first.num_experts,
            "top_k": first.top_k,
            "aux_weight": float(first.balance_loss_coeff),
        }
    return {"balance": "none", "blocks": 0}


def watch(model) -> dict:
    """What the trainer actually did, where its arguments do not say.

    YOLO-Master's trainer answers the first non-finite gradient by replaying that epoch and training
    the rest of the run without mixed precision, while the run's arguments still read amp=True. Both
    trainers also halve the batch of a run that runs out of memory in its first epoch.
    """
    seen = {"epochs_started": 0, "amp": None, "batch": None}

    def started(trainer):
        seen["epochs_started"] += 1

    def ended(trainer):
        seen["amp"], seen["batch"] = bool(trainer.amp), int(trainer.batch_size)

    model.add_callback("on_train_epoch_start", started)
    model.add_callback("on_train_end", ended)
    return seen


def build(args):
    if args.upstream or not args.esmoe:
        return YOLO(args.base), args.base
    esmoe.inject_esmoe()
    if args.grafted:
        # Settings come from the config itself, as upstream's own configs carry theirs.
        return esmoe.attach_aux_loss(YOLO(args.base), weight=args.aux_weight, recipe=args.recipe), args.base
    wire = "-rewire" if args.rewire else ""
    # The seed is in the filename because two runs of the same arm train side by side: a shared
    # path lets one truncate the config the other is still reading.
    cfg = (
        ROOT / "configs" / f"{Path(args.base).stem}-esmoe-e{args.num_experts}k{args.top_k}{wire}"
        f"-{args.balance}-{args.at}-s{args.seed}.yaml"
    )
    esmoe.graft(
        args.base,
        out=str(cfg),
        at=args.at,
        num_experts=args.num_experts,
        top_k=args.top_k,
        rewire=args.rewire,
        # In the config, not on the built blocks: the trainer rebuilds the model from this file and
        # drops whatever was set on the instance. Runs that set them afterwards trained the
        # defaults while their records claimed otherwise.
        **requested(args),
    )
    model = YOLO(str(cfg))
    esmoe.attach_aux_loss(model, weight=args.aux_weight, recipe=args.recipe)
    return model, str(cfg)


def architecture(args) -> str:
    """The arm's name, spelling out every switch that makes it a different model.

    `scripts/queue.sh` rebuilds this string to find the directory a resumed run left behind, so
    the two have to agree; `tests/test_queue.py` holds them to it. The auxiliary weight trails
    the structural suffixes because the queue appends it last.
    """
    if args.upstream:
        return "upstream"
    if not args.esmoe:
        return "baseline"
    arch = "esmoe-rewire" if args.rewire else "esmoe"
    if args.balance != "switch":
        arch = f"{arch}-{args.balance}"
    if args.at != "backbone_end":
        arch = f"{arch}-{args.at.replace('backbone_', '')}"
    if args.out_norm:
        arch = f"{arch}-norm"
    if args.dense_training:
        arch = f"{arch}-dense"
    if args.dense_inference:
        arch = f"{arch}-denseval"
    if args.dynamic_threshold:
        arch = f"{arch}-t{args.dynamic_threshold:g}"
    if args.recipe != "esmoe":
        arch = f"{arch}-{args.recipe}"
    if args.aux_weight != 0.01:
        arch = f"{arch}-w{args.aux_weight:g}"
    return arch


def run_name(args, fork: bool) -> str:
    """The run directory, which `scripts/queue.sh` rebuilds as well.

    A config that already holds the blocks drops its ``-esmoe`` so the arm is not named twice. A
    run on YOLO-Master's fork ends in ``-fork``: its baseline and the official one share every other
    part of the name, and two runs under one name is how a checkpoint once took another's hash.
    """
    stem = Path(args.base).stem
    stem = stem.removesuffix("-esmoe") if args.grafted else stem
    # A run that asked for FP32 carries it in the name: two runs of one configuration at different
    # precision are different experiments, and a shared directory is how one overwrites the other.
    precision = "" if args.amp else "-fp32"
    return f"{stem}-{architecture(args)}-e{args.epochs}-s{args.seed}{args.tag}{precision}" + ("-fork" if fork else "")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="yolov8n.yaml")
    p.add_argument("--data", default=str(ROOT / "configs" / "visdrone.yaml"))
    p.add_argument("--esmoe", action="store_true")
    p.add_argument("--grafted", action="store_true", help="--base already holds ESMoE blocks and their settings")
    p.add_argument("--upstream", action="store_true", help="train --base with YOLO-Master's own blocks, on its fork")
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--rewire", action="store_true")
    p.add_argument("--balance", choices=("switch", "gshard", "master", "gshard_probs"), default="switch")
    p.add_argument("--at", default="backbone_end", help="graft point: backbone_end or backbone_stages")
    p.add_argument("--out-norm", action="store_true", help="normalise the mixed output, as upstream does")
    p.add_argument("--dense-training", action="store_true", help="run every expert while training")
    p.add_argument("--dense-inference", action="store_true", help="run every expert outside training too")
    p.add_argument("--dynamic-threshold", type=float, default=0.0, help="upstream's inference pruning; 0.4 there")
    p.add_argument("--recipe", choices=esmoe.inject.RECIPES, default="esmoe", help="how the aux term and block train")
    p.add_argument("--aux-weight", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--amp", type=int, default=1)
    # ultralytics 把 0 读作 "no patience" 并禁用早停，这正是复现协议要的固定周期。
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--tag", default="")
    return p


def mismatch(args, stack: str) -> str | None:
    """Why these flags cannot run on this ultralytics, or None when they can.

    A plain or ESMoE run trains on either framework, which is how the fork gets its own baseline;
    only YOLO-Master's own blocks need the fork.
    """
    if args.upstream and stack == "ultralytics":
        return f"--upstream trains YOLO-Master's own blocks, but {ultralytics.__file__} is the official package"
    if args.grafted and not args.esmoe:
        return "--grafted names a config that holds ESMoE blocks, so it needs --esmoe"
    return None


def main():
    args = build_parser().parse_args()
    stack = framework()
    if problem := mismatch(args, stack):
        raise SystemExit(problem)

    model, cfg = build(args)
    facts = block_facts(model)
    # Fail before burning a card rather than record an arm the model does not have. A grafted
    # config states its own settings, so there is no request to hold it to.
    if args.esmoe and not args.grafted and any(facts[key] != value for key, value in requested(args).items()):
        raise SystemExit(f"block settings did not reach the model: asked {requested(args)}, got {facts}")
    arch = architecture(args)
    name = run_name(args, fork=stack != "ultralytics")
    experiment_id = f"{name}-{time.strftime('%Y%m%d%H%M%S')}"
    seen = watch(model)

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
            patience=args.patience,
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

    weights = ROOT / "runs" / name / "weights" / "best.pt"
    # Re-read after training: `model.model` is now the checkpoint that was actually saved.
    facts = block_facts(model)
    record = {
        "experiment_id": experiment_id,
        "git_ref": {
            "toolkit": git_ref(),
            "toolkit_version": esmoe.__version__,
            "ultralytics": ultralytics.__version__,
            "framework": stack,
            "yolo_master_base_ref": YOLO_MASTER_BASE_REF,
            "yolo_master_release": YOLO_MASTER_RELEASE,
        },
        "config": {
            "model_yaml": cfg,
            "sha256": digest(Path(cfg)),
            "arch": arch,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "rewire": bool(args.rewire and args.esmoe),
            "aux_weight": args.aux_weight if args.esmoe else 0.0,
            **({"recipe": "upstream" if args.upstream else args.recipe} if facts["blocks"] else {}),
            **facts,
        },
        "dataset": {"yaml": args.data, "fraction": args.fraction, **dataset_facts(args.data)},
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "budget": {
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "patience": args.patience,
            "amp": bool(args.amp),
            "amp_at_end": seen["amp"],
            "batch_at_end": seen["batch"],
            "epochs_replayed": max(seen["epochs_started"] - args.epochs, 0),
            "wall_seconds": round(elapsed, 1),
            "gpu_hours": round(elapsed / 3600, 3),
        },
        "seed": args.seed,
        "metrics": metrics,
        "params": get_num_params(model.model),
        "artifact": {
            "path": str(weights),
            "sha256": digest(weights),
        },
        "status": status,
        "limitation": error or "single machine, single GPU; see limitations.md",
    }
    out = ROOT / "results" / f"{experiment_id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
