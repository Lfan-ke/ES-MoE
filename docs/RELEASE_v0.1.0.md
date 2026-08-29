# esmoe 0.1.0

First release. An ES-MoE block that installs next to stock Ultralytics instead of living inside a
fork, with the evidence needed to judge whether it is worth using.

## What it does

- `inject_esmoe()` registers `ESMoE` where `parse_model` resolves layer names, so a model.yaml can
  reference it directly.
- `graft(base, out=...)` appends the block to a stock backbone and renumbers the head, which is
  required because head layers address earlier layers by absolute index.
- `attach_aux_loss(model, weight)` puts the router load-balancing loss into the optimised training
  loss and logs it as its own `esmoe_aux` column.
- `collect_aux_loss(model)` returns this step's router loss for custom training loops.

## Verified

- YOLOv8n, YOLO11n and YOLO12n build, forward and train with the grafted block on
  `ultralytics==8.4.101`.
- The aux term changes the total loss and produces router gradients (unit tested, not asserted by
  configuration alone).
- Default config `num_experts=4, top_k=2, weight=0.01` selected under one budget and confirmed on
  three seeds: paired win 3/3, +0.0021 mAP50 against the same-budget baseline, +10.4% parameters.
  Evidence in `docs/SELECTION.md` and `results/`.

## Known limits

Numbers come from a unified small benchmark (VisDrone, 25% subset, 20 epochs from scratch), not
from COCO or a converged schedule. The block is channel preserving. Channels are inferred on first
forward. One aux-loss setting per process. Full list in `limitations.md`.

## Install

    pip install esmoe
