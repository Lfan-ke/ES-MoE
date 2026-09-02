"""esmoe on YOLO-Master's vendored ultralytics: graft their config, train, read the aux column."""

import csv
import tempfile
from pathlib import Path

import ultralytics
from ultralytics import YOLO

import esmoe

print("fork ultralytics", ultralytics.__version__, "| esmoe", esmoe.__version__, flush=True)
esmoe.inject_esmoe()

target = Path(tempfile.mkdtemp(prefix="esmoe-fork-")) / "yolo-master-n-esmoe.yaml"
esmoe.graft("yolo-master-n.yaml", out=str(target))
model = YOLO(str(target))
print("graft+build ok |", sum(p.numel() for p in model.model.parameters()), "params", flush=True)

esmoe.attach_aux_loss(model, weight=0.01)
model.train(
    data="coco8.yaml",
    epochs=1,
    imgsz=64,
    batch=4,
    workers=0,
    plots=False,
    verbose=False,
    project=str(Path(tempfile.mkdtemp(prefix="esmoe-fork-runs-"))),
    name="smoke",
    device="cpu",
)
with open(Path(model.trainer.save_dir) / "results.csv", newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
key = next(k for k in rows[0] if k.strip().endswith("esmoe_aux"))
values = [float(r[key]) for r in rows]
assert all(v > 0 for v in values), values
print("train ok | esmoe_aux per epoch:", values, flush=True)

their_cfg = "yolo-master-esmoe-n-visdrone.yaml"
their = YOLO(their_cfg)
print("their ES_MOE config builds alongside ours:", type(their.model).__name__, flush=True)
