# esmoe

Drop-in ES-MoE (expert-sparse Mixture-of-Experts) block for Ultralytics YOLO. It installs beside the
official `ultralytics` package instead of replacing it with a fork, and it ships budget-fair evidence
plus an auxiliary loss that provably reaches the optimiser.

Docs: https://lfan-ke.github.io/esmoe-toolkit/

## Install

    pip install esmoe

Requires a stock `ultralytics`; nothing from the YOLO-Master fork is needed at runtime.

## Use

One call covers register, graft, build and wire:

    import esmoe

    model = esmoe.equip("yolo11n.yaml", weight=0.01)
    model.train(data="coco8.yaml", epochs=10)

Or take the steps apart when you need control over each one:

    from ultralytics import YOLO
    import esmoe

    esmoe.inject_esmoe()                                    # make `ESMoE` resolvable in model.yaml
    esmoe.graft("yolov8n.yaml", out="v8-esmoe.yaml", at=[4, 6])  # insert blocks, renumber the head
    model = YOLO("v8-esmoe.yaml")
    esmoe.attach_aux_loss(model, weight=0.01)               # router loss joins the training loss

From the shell:

    esmoe graft yolo11n.yaml -o yolo11n-esmoe.yaml -e 4 -k 2 --at backbone_end

`attach_aux_loss` adds an `esmoe_aux` entry to the trainer's loss table, so a non-zero,
back-propagated auxiliary term shows up in `results.csv` rather than merely in a config.

Written by hand, a grafted config layer is just:

    [-1, 1, ESMoE, [4, 2]]   # num_experts, top_k

The block is channel preserving and infers its width on the first forward, which is what lets stock
`parse_model` size it without a patch.

## Extend

Experts and the balancing objective are plain callables, so a variant is a few lines:

    esmoe.ESMoE(num_experts=4, top_k=2, expert=MyExpert, balance=my_balance_fn)

`MyExpert(c1, c2, k) -> Module`, `my_balance_fn(probs, gate) -> scalar`. `esmoe.blocks(model)` walks
every block in a model, and `esmoe.collect_aux_loss(model)` returns the current step's router loss
for custom training loops.

## Compatibility

| backbone | build + forward | grafted config | aux loss in training |
|:--:|:--:|:--:|:--:|
| YOLOv8 | yes | yes | yes |
| YOLO11 | yes | yes | yes |
| YOLO12 | yes | yes | yes |

Verified by `tests/test_ultralytics.py` on ultralytics 8.4.101 and 8.4.132, which report loss items
in two different shapes; both are handled.

## Selected default

`ESMoE(num_experts=4, top_k=2)` with `attach_aux_loss(weight=0.01)`, chosen under one budget over
2/4/8-expert and top-1 variants, then confirmed on three seeds (paired win 3/3, +0.0021 mAP50 over
the same-budget baseline, +10.4% parameters). Reasoning and full table: `docs/SELECTION.md`.

## Develop and reproduce

    uv sync --group dev
    uv run pytest -q
    uv run python scripts/capture_env.py                  # freeze environment into env/
    EPOCHS=20 FRACTION=0.25 SEEDS="0 1 2" bash scripts/sweep.sh
    uv run python scripts/report.py                       # results/summary.md

Every run writes one machine-readable record to `results/` (config, dataset, hardware, budget, seed,
metrics, artifact, status, limitation). Read `limitations.md` before quoting any number.

## License

AGPL-3.0-only, matching the Ultralytics ecosystem it builds on.
