# Checkpoints

The 36 `best.pt` checkpoints behind the protocol matrix on `main` — four backbones (YOLOv8n / 11n / 12n / 26n) × three arms (baseline / esmoe / esmoe-rewire) × three seeds, VisDrone, imgsz 800, 120 epochs, `patience=0`, batch 32.

This branch is an orphan and every `*.pt` is Git LFS, so `main` stays small; nothing here is needed to use the package.

- `weights/<run>-best.pt` — the checkpoint; `<run>` matches the run records in `main:results/`.
- `args/<run>.yaml` — the full ultralytics argument dump of that run.

To recompute any number on the docs site from a checkpoint:

    git lfs pull --include "weights/yolo12n-*"
    python scripts/buckets.py weights/yolo12n-*-best.pt      # area buckets
    python scripts/routing.py weights/yolo12n-esmoe-*-best.pt  # router statistics

Verdicts and judgment lines: `main:docs/JUDGMENT.md`.
