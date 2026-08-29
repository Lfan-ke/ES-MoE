# P0 selection: which ES-MoE configuration ships as the default

Budget: VisDrone2019-DET `fraction=0.25`, imgsz 640, 20 epochs from scratch, batch 32, YOLOv8n,
one RTX 4090 D. Every arm sees the same data, schedule and augmentation; only the block config
moves. Raw records are in `results/`, rendered by `scripts/report.py` into `results/summary.md`.

## Provenance

Public baseline `acce839c7e895d6b179de7f7093fa879e237cc7b` (YOLO-Master main at 2026-08-21
23:59:59 +0800), release reference `YOLO-Master-v26.08` -> `43d40117c...`; both carry
`ultralytics 8.4.101`, which is exactly the stock version this toolkit runs against, so the plug-in
path is measured on the same library version as the baseline. Full statement in `docs/BASELINE.md`. Records produced before the toolkit started stamping its own revision
carry `git_ref = "0.1.0"`; every later record stamps the commit, ultralytics version and both
locked baselines.

## Candidates at seed 0

| variant | experts | top_k | aux weight | params | mAP50 | delta vs baseline |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| baseline | - | - | - | 3.01M | 0.0933 | - |
| esmoe-e2k1 | 2 | 1 | 0.01 | 3.16M | 0.0903 | -0.0031 |
| esmoe-e4k1 | 4 | 1 | 0.01 | 3.33M | 0.0928 | -0.0006 |
| esmoe-e4k2 | 4 | 2 | 0.01 | 3.33M | 0.0950 | +0.0017 |
| esmoe-e4k2, aux off | 4 | 2 | 0.00 | 3.33M | 0.0938 | +0.0005 |
| esmoe-e8k2 | 8 | 2 | 0.01 | 3.78M | 0.0934 | +0.0000 |

## Confirmation across seeds

| arch | seeds | mAP50 | mAP50-95 | paired mean delta | wins |
|:--:|:--:|:--:|:--:|:--:|:--:|
| baseline | 0,1,2 | 0.0909 ± 0.0022 | 0.0405 ± 0.0013 | - | - |
| esmoe-e4k2 | 0,1,2 | 0.0930 ± 0.0022 | 0.0410 ± 0.0013 | +0.0021 mAP50 | 3/3 |

## Decision

`ESMoE(num_experts=4, top_k=2)` with `attach_aux_loss(weight=0.01)` is the shipped default.

- Routing to a single expert loses to the baseline (e2k1, e4k1). The gain needs a mixture, not a
  switch, at this budget.
- Doubling the expert count to 8 buys nothing measurable and costs 13% more parameters than e4k2.
- Turning the auxiliary loss off costs 0.0012 mAP50 at the same parameter count, which is the
  concrete argument for wiring it into the optimised loss rather than merely computing it.

## How much this claim is worth

The per-arm standard deviations overlap; only the paired per-seed comparison supports the
ranking, and it does so on three seeds at one short budget. The absolute gain (+2.3% relative
mAP50) comes with +10.4% parameters. Rerunning an identical config at the same seed reproduced
the metric exactly, so the deltas are not run-to-run jitter. See `limitations.md` for what these
numbers do not say.
