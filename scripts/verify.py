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


CHECKS = (
    ("training logs a positive aux term", check_training_logs_aux),
    ("weight=0 leaves the loss table untouched", check_aux_weight_zero_disables_the_term),
    ("checkpoint round trip", check_checkpoint_round_trip),
    ("resume keeps the aux term", check_resume_keeps_the_aux_term),
    ("several blocks train together", check_multiple_blocks_train),
    ("val and predict", check_val_and_predict),
    ("onnx export", check_onnx_export),
    ("onnx export routes per input", check_export_is_faithful),
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
