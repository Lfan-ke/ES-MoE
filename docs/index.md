# esmoe

Drop-in ES-MoE (expert-sparse Mixture-of-Experts) block for Ultralytics YOLO. It installs beside the
official `ultralytics` package instead of replacing it with a fork, and it ships the evidence needed
to judge whether the block is worth using.

## Install

    pip install esmoe

## Use

    import esmoe

    model = esmoe.equip("yolo11n.yaml", weight=0.01)   # register + graft + build + wire
    model.train(data="coco8.yaml", epochs=10)

The steps are also available on their own - `inject_esmoe()`, `graft(base, out=..., at=...)`,
`attach_aux_loss(model, weight=...)` - and from the shell:

    esmoe graft yolo11n.yaml -o yolo11n-esmoe.yaml -e 4 -k 2 --at backbone_end

`attach_aux_loss` adds an `esmoe_aux` column to the trainer's loss table, so the auxiliary term is
visible in `results.csv` as a back-propagated number rather than a configuration key.

## Compatibility

| backbone | build + forward | grafted config | aux loss in training |
|:--:|:--:|:--:|:--:|
| YOLOv8 | yes | yes | yes |
| YOLO11 | yes | yes | yes |
| YOLO12 | yes | yes | yes |

Verified by `tests/test_ultralytics.py` on ultralytics 8.4.101 and 8.4.132, plus a real 1-epoch
training run per generation logging a non-zero `train/esmoe_aux`.

## Evidence

- [Selection](SELECTION.md) - which configuration ships as the default and why.
- [Limitations](limitations.md) - what the numbers do not say. Read this before quoting any of them.
- [Baseline and increment](BASELINE.md) - locked references and what counts as new work.
- [Mid-term report](MIDTERM.md) - question, budget, evidence, open points.
