# esmoe-toolkit

Drop-in ES-MoE (expert-sparse Mixture-of-Experts) block for Ultralytics YOLO. It turns the
community "single-seed accuracy bump" into an installable module with budget-fair evidence and an
auxiliary loss that provably reaches the optimiser.

## Install

    pip install -e .

Requires a stock `ultralytics` install; nothing in the YOLO-Master fork is needed at runtime.

## Use

    from ultralytics import YOLO
    import esmoe

    esmoe.inject_esmoe()                                  # make `ESMoE` resolvable in model.yaml
    esmoe.graft("yolov8n.yaml", out="yolov8n-esmoe.yaml")  # append the block, renumber the head
    model = YOLO("yolov8n-esmoe.yaml")
    esmoe.attach_aux_loss(model, weight=0.01)             # router loss joins the training loss
    model.train(data="coco8.yaml", epochs=10)

`attach_aux_loss` adds an `esmoe_aux` column to the trainer's loss table, so a non-zero,
back-propagated auxiliary term is visible in `results.csv` rather than merely configured.

Written by hand, a grafted config layer looks like:

    [-1, 1, ESMoE, [4, 2]]   # num_experts, top_k

The block is channel preserving, which is what lets stock `parse_model` size it without a patch.

## Compatibility

| backbone | build + forward | grafted config | aux loss in training |
|:--:|:--:|:--:|:--:|
| YOLOv8 | yes | yes | yes |
| YOLO11 | yes | yes | yes |
| YOLO12 | yes | yes | yes |

Verified by `tests/test_ultralytics.py` against ultralytics 8.4.101.

## Selected default

`ESMoE(num_experts=4, top_k=2)` with `attach_aux_loss(weight=0.01)`, chosen under one budget over
2/4/8-expert and top-1 variants, then confirmed on three seeds (paired win 3/3, +0.0021 mAP50 over
the same-budget baseline, +10.4% parameters). Reasoning and the full table: `docs/SELECTION.md`.

## Reproduce the numbers

    python scripts/capture_env.py                       # freeze environment into env/
    EPOCHS=20 FRACTION=0.25 SEEDS="0 1 2" bash scripts/sweep.sh
    python scripts/report.py                            # results/summary.md

Every run writes one machine-readable record to `results/` (config, dataset, hardware, budget,
seed, metrics, artifact, status, limitation). Read `limitations.md` before quoting any number.

## License

AGPL-3.0-only, matching the Ultralytics ecosystem it builds on.
