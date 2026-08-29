# esmoe

Drop-in ES-MoE (expert-sparse Mixture-of-Experts) block for Ultralytics YOLO. It installs beside the
official `ultralytics` package instead of replacing it with a fork, and it ships the evidence needed
to judge whether the block is worth using.

## Install

    pip install esmoe

## Use

    from ultralytics import YOLO
    import esmoe

    esmoe.inject_esmoe()                                   # make `ESMoE` resolvable in model.yaml
    esmoe.graft("yolov8n.yaml", out="yolov8n-esmoe.yaml")  # append the block, renumber the head
    model = YOLO("yolov8n-esmoe.yaml")
    esmoe.attach_aux_loss(model, weight=0.01)              # router loss joins the training loss
    model.train(data="coco8.yaml", epochs=10)

`attach_aux_loss` adds an `esmoe_aux` column to the trainer's loss table, so the auxiliary term is
visible in `results.csv` as a back-propagated number rather than a configuration key.

## Compatibility

| backbone | build + forward | grafted config | aux loss in training |
|:--:|:--:|:--:|:--:|
| YOLOv8 | yes | yes | yes |
| YOLO11 | yes | yes | yes |
| YOLO12 | yes | yes | yes |

Verified by `tests/test_ultralytics.py` against `ultralytics==8.4.101`.

## Evidence

- [Selection](SELECTION.md) - which configuration ships as the default and why.
- [Limitations](limitations.md) - what the numbers do not say. Read this before quoting any of them.
- [Baseline and increment](BASELINE.md) - locked references and what counts as new work.
- [Mid-term report](MIDTERM.md) - question, budget, evidence, open points.
