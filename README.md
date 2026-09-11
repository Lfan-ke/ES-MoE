# Checkpoints

The 113 `best.pt` checkpoints behind the protocol matrix on `main`: seven backbones (YOLOv5n / v8n / v9t / v10n / 11n / 12n / 26n) with the baseline, esmoe and esmoe-rewire arms, and the alignment arms that followed (output norm, dense training, four blocks, the gate-reading objectives, no balancing term). Three seeds or more each; VisDrone, imgsz 800, 120 epochs, `patience=0`, batch 32.

YOLOv5n appears twice: the `-p800` runs were measured on one MetaX C500 host and the `-p800h2` runs on another, which is the cross-host replication reported on the judgment-lines page. One YOLOv9t baseline (seed 0) was trained on an RTX 4090 under the same directory name as its later MetaX run; the MetaX run keeps the name, and the 4090 run is published with a `-4090` suffix, which its record gives as `artifact.published_as`.

This branch is an orphan and every checkpoint and curve is Git LFS, so `main` stays small; nothing here is needed to use the package.

- `weights/<run>-best.pt` — the checkpoint; `<run>` matches the run records in `main:results/`.
- `args/<run>.yaml` — the full ultralytics argument dump of that run.
- `curves/<run>.csv` — the per-epoch `results.csv` the run wrote, so a rerun can be compared with it epoch by epoch.

To recompute any number on the docs site from a checkpoint:

    git lfs pull --include "weights/yolo12n-*"
    python scripts/buckets.py weights/yolo12n-*-best.pt      # area buckets
    python scripts/routing.py weights/yolo12n-esmoe-*-best.pt  # router statistics

Verdicts and judgment lines: `main:docs/JUDGMENT.md`.
