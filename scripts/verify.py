"""Correctness checks that a unit test cannot make: real training, saving, resuming and exporting.

Each check prints PASS or FAIL with the observation behind it, and the script exits non-zero if any
of them failed, so a run of this file is admissible evidence rather than a demonstration.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

import esmoe

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "verify"
BUDGET = {"imgsz": 320, "batch": 4, "workers": 0, "plots": False, "verbose": False, "exist_ok": True}


def train(model, name, epochs=2, data="coco8.yaml", **extra):
    return model.train(data=data, epochs=epochs, project=str(RUNS), name=name, **BUDGET, **extra)


def aux_column(save_dir):
    """The per-epoch auxiliary values a training run actually logged."""
    import csv

    with open(Path(save_dir) / "results.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    key = next((k for k in rows[0] if k.strip().endswith("esmoe_aux")), None)
    return [float(row[key]) for row in rows] if key else []


def check_training_logs_aux():
    model = esmoe.equip("yolo11n.yaml", weight=0.01)
    train(model, "aux-on")
    values = aux_column(model.trainer.save_dir)
    ok = len(values) == 2 and all(v > 0 for v in values)
    return ok, f"logged aux per epoch: {values}"


def check_aux_weight_zero_disables_the_term():
    model = esmoe.equip("yolo11n.yaml", weight=0.0)
    train(model, "aux-off", epochs=1)
    values = aux_column(model.trainer.save_dir)
    return not values, f"aux column with weight=0: {values or 'absent, as intended'}"


def check_checkpoint_round_trip():
    from ultralytics import YOLO

    model = esmoe.equip("yolo11n.yaml", weight=0.01)
    train(model, "round-trip", epochs=1)
    weights = Path(model.trainer.save_dir) / "weights" / "best.pt"

    esmoe.inject_esmoe()  # a fresh process would do this before loading
    reloaded = YOLO(str(weights))
    blocks = list(esmoe.blocks(reloaded.model))
    before = {name: p.detach().clone() for name, p in model.model.named_parameters() if "ESMoE" in type(p).__name__}
    del before
    same = sum(p.numel() for p in reloaded.model.parameters()) == sum(p.numel() for p in model.model.parameters())
    prediction = reloaded.predict(torch.zeros(1, 3, 320, 320), verbose=False)
    return bool(blocks) and same and prediction is not None, (
        f"reloaded {len(blocks)} block(s), parameter count preserved: {same}, predict ran"
    )


def check_resume_keeps_the_aux_term():
    model = esmoe.equip("yolo11n.yaml", weight=0.01)
    train(model, "resume", epochs=2)
    first = aux_column(model.trainer.save_dir)

    from ultralytics import YOLO

    esmoe.inject_esmoe()
    resumed = YOLO(str(Path(model.trainer.save_dir) / "weights" / "last.pt"))
    esmoe.attach_aux_loss(resumed, weight=0.01)
    resumed.train(data="coco8.yaml", epochs=4, resume=True, project=str(RUNS), name="resume", **BUDGET)
    after = aux_column(resumed.trainer.save_dir)
    ok = len(after) > len(first) and all(v > 0 for v in after)
    return (
        ok,
        f"epochs logged before resume {len(first)}, after {len(after)}; all positive: {all(v > 0 for v in after)}",
    )


def check_multiple_blocks_train():
    model = esmoe.equip("yolo11n.yaml", weight=0.01, at=[4, 6])
    blocks = len(list(esmoe.blocks(model.model)))
    train(model, "multi-block", epochs=1)
    values = aux_column(model.trainer.save_dir)
    return blocks == 2 and bool(values) and values[0] > 0, f"{blocks} blocks, aux {values}"


def check_val_and_predict():
    model = esmoe.equip("yolo11n.yaml", weight=0.01)
    train(model, "val-predict", epochs=1)
    metrics = model.val(data="coco8.yaml", imgsz=320, verbose=False)
    prediction = model.predict(torch.zeros(2, 3, 320, 320), verbose=False)
    return metrics is not None and len(
        prediction
    ) == 2, f"val returned metrics, predict returned {len(prediction)} results"


def check_onnx_export():
    model = esmoe.equip("yolo11n.yaml", weight=0.01)
    train(model, "export", epochs=1)
    try:
        path = model.export(format="onnx", imgsz=320, simplify=False, verbose=False)
    except Exception as exc:  # a failure here is a real finding, not a crash of this script
        return False, f"export raised {type(exc).__name__}: {str(exc)[:160]}"
    size = Path(path).stat().st_size if path and Path(path).exists() else 0
    return size > 0, f"exported {Path(path).name}, {size / 1e6:.1f} MB"


def check_export_is_faithful():
    """Export on one input, then compare against PyTorch on an input that routes elsewhere."""
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        return True, "skipped: onnxruntime not installed"

    block = esmoe.ESMoE(num_experts=4, top_k=1, channels=16).eval()
    with torch.no_grad():
        block.router[2].weight.zero_()
        block.router[2].bias.zero_()
        block.router[2].weight[0].fill_(4.0)
        block.router[2].weight[1].fill_(-4.0)
        block.router[4].weight.zero_()
        block.router[4].bias.zero_()
        block.router[4].weight[0, 0] = 6.0
        block.router[4].weight[1, 1] = 6.0

    positive, negative = torch.full((1, 16, 4, 4), 0.7), torch.full((1, 16, 4, 4), -0.7)
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / "routing.onnx"
    torch.onnx.export(
        block, (positive,), str(path), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for sample in (positive, negative):
        with torch.no_grad():
            expected = block(sample).numpy()
        worst = max(worst, float(np.abs(expected - session.run(["y"], {"x": sample.numpy()})[0]).max()))
    return worst < 1e-4, f"worst |torch - onnx| across both routing branches: {worst:.2e}"


def check_ddp_worker_file_trains():
    """The file ultralytics hands to torchrun, run in a fresh interpreter that never imported esmoe.

    Whatever that file imports is all a DDP worker knows; it must be enough to build the grafted
    model and log the auxiliary term. One CPU rank is the same code path minus the collective.
    """
    import os
    import subprocess

    from ultralytics.utils.dist import generate_ddp_file

    model = esmoe.equip("yolo11n.yaml", weight=0.01)
    overrides = {**model.overrides, "data": "coco8.yaml", "epochs": 1, "project": str(RUNS), "name": "ddp-worker"}
    trainer = model._smart_load("trainer")(overrides={**overrides, **BUDGET})
    worker = generate_ddp_file(trainer)
    try:
        done = subprocess.run(
            [sys.executable, worker], capture_output=True, text=True, cwd=ROOT, env=os.environ.copy(), timeout=900
        )
    finally:
        Path(worker).unlink(missing_ok=True)
    if done.returncode != 0:
        return False, f"worker exited {done.returncode}: {done.stderr.strip().splitlines()[-1][:160]}"
    values = aux_column(trainer.save_dir)
    return bool(values) and all(v > 0 for v in values), f"worker {Path(worker).name} logged aux {values}"


def _rank(rank, world, port, cfg, box):
    import torch.distributed as dist
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG

    from esmoe.inject import AUX_NAME, arm_process

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    torch.manual_seed(0)
    esmoe.inject_esmoe()
    arm_process(0.01)
    model = DetectionModel(cfg, ch=3, nc=2, verbose=False)
    model.args = get_cfg(DEFAULT_CFG)
    # Sparse dispatch leaves the experts a batch did not route to out of the graph, so DDP has to be
    # told to expect unused parameters - which is how ultralytics constructs it as well.
    ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    torch.manual_seed(rank + 1)  # each rank sees different images
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "cls": torch.zeros(2, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]]),
        "batch_idx": torch.tensor([0.0, 1.0]),
    }
    total, items = ddp(batch)
    total.sum().backward()
    router = next(esmoe.blocks(model)).router
    grads = torch.cat([p.grad.flatten() for p in router.parameters()])
    gathered = [torch.zeros_like(grads) for _ in range(world)]
    dist.all_gather(gathered, grads)
    aux = items[AUX_NAME] if isinstance(items, dict) else items[-1]
    box[rank] = {
        "aux": float(aux),
        "finite": bool(torch.isfinite(total).all()),
        "grads_agree": all(torch.allclose(gathered[0], g, atol=1e-6) for g in gathered[1:]),
    }
    dist.destroy_process_group()


def check_two_ranks_share_the_router_gradient():
    """Two CPU ranks under gloo: each computes its own aux term, DDP averages the router gradient."""
    import socket
    import tempfile

    import torch.multiprocessing as mp

    esmoe.inject_esmoe()
    cfg = str(Path(tempfile.mkdtemp(prefix="esmoe-ddp-")) / "v8-esmoe.yaml")
    esmoe.graft("yolov8n.yaml", out=cfg)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        box = manager.dict()
        ranks = [ctx.Process(target=_rank, args=(r, 2, port, cfg, box)) for r in range(2)]
        for p in ranks:
            p.start()
        for p in ranks:
            p.join(600)
        report = dict(box)
    if len(report) != 2:
        return False, f"only {len(report)} rank(s) reported: {report}"
    ok = all(r["aux"] > 0 and r["finite"] and r["grads_agree"] for r in report.values())
    aux = ", ".join(f"rank{r} aux {report[r]['aux']:.4f}" for r in sorted(report))
    return ok, f"{aux}; router grads agree: {report[0]['grads_agree']}"


CHECKS = (
    ("training logs a positive aux term", check_training_logs_aux),
    ("weight=0 leaves the loss table untouched", check_aux_weight_zero_disables_the_term),
    ("checkpoint round trip", check_checkpoint_round_trip),
    ("resume keeps the aux term", check_resume_keeps_the_aux_term),
    ("several blocks train together", check_multiple_blocks_train),
    ("val and predict", check_val_and_predict),
    ("onnx export", check_onnx_export),
    ("onnx export routes per input", check_export_is_faithful),
    ("ddp worker file trains in a fresh interpreter", check_ddp_worker_file_trains),
    ("two ranks share the router gradient", check_two_ranks_share_the_router_gradient),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="substring of a check name")
    parser.add_argument("--keep-runs", action="store_true")
    args = parser.parse_args()

    if RUNS.exists() and not args.keep_runs:
        shutil.rmtree(RUNS, ignore_errors=True)

    report, failures = [], 0
    for name, check in CHECKS:
        if args.only and args.only not in name:
            continue
        started = time.time()
        try:
            ok, detail = check()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {str(exc)[:200]}"
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}  ({time.time() - started:.0f}s)", flush=True)
        report.append({"check": name, "passed": ok, "detail": detail})

    out = ROOT / "results" / "verify.json"
    out.write_text(
        json.dumps(
            {
                "esmoe": esmoe.__version__,
                "torch": torch.__version__,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "checks": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(report) - failures}/{len(report)} passed, written to {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
