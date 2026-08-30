# PR draft: reference esmoe from YOLO-Master

Prepared, not submitted. Submitting anything upstream needs the maintainer conversation to be worth
their time, and that call is not mine to make alone. The four sections below follow the task book's
required PR structure.

## Change summary

Add one paragraph and one link to the YOLO-Master documentation, pointing readers who want ES-MoE on
a stock `ultralytics` install to `esmoe` (PyPI: `esmoe`, source: `Lfan-ke/ES-MoE`, AGPL-3.0-only).

Why: the block currently ships only inside this repository, whose distribution is itself named
`ultralytics`. Anyone who wants to try ES-MoE without moving their whole toolchain onto the fork has
no supported path today. The plug-in reimplements the published design against public extension
points - module registration, config grafting and a loss wrapper - so it neither vendors code from
this repository nor requires patching `ultralytics`.

No code in YOLO-Master changes.

## Test evidence

For the plug-in itself (`Lfan-ke/ES-MoE`, CI green on push):

- 21 tests on Python 3.10 and 3.12, covering block forward and gradient flow, config grafting with
  head renumbering, three-generation model construction (YOLOv8n / YOLO11n / YOLO12n), and the
  identity `total(with aux) == total(without) + aux * batch_size`.
- The same suite passes against `ultralytics` 8.4.101 and 8.4.132, which report loss items in two
  different shapes.
- A quick-start notebook is executed on CPU in CI so the documented path cannot rot.
- One real 1-epoch training run per backbone generation, each logging a non-zero `train/esmoe_aux`
  column in `results.csv`.

## Ablation data

VisDrone2019-DET, imgsz 640, from scratch, YOLOv8n, one RTX 4090 D. Candidate selection at 20 epochs
on a 25% subset, seed 0:

| variant | params | mAP50 | delta |
|:--:|:--:|:--:|:--:|
| baseline | 3.01M | 0.0933 | - |
| 2 experts, top-1 | 3.16M | 0.0903 | -0.0031 |
| 4 experts, top-1 | 3.33M | 0.0928 | -0.0006 |
| 4 experts, top-2 | 3.33M | 0.0950 | +0.0017 |
| 4 experts, top-2, aux off | 3.33M | 0.0938 | +0.0005 |
| 8 experts, top-2 | 3.78M | 0.0934 | +0.0000 |

Confirmation of the selected arm on the full training set, 20 epochs, three seeds: baseline
0.1496 ± 0.0022 mAP50 against 0.1517 ± 0.0011, paired deltas +0.0001 / +0.0032 / +0.0029, 3/3 seeds,
mean +0.0021 mAP50 and +0.0019 mAP50-95. Repeating an identical configuration at the same seed
reproduced the metric exactly.

## Known limitations

- Numbers are VisDrone, not COCO, and 20 epochs from scratch is far from convergence. They do not
  reproduce or contest the ES-MoE-N anchor.
- One of the three confirmation seeds is effectively a tie (+0.0001 mAP50): the effect is consistent
  in sign, small, and not reliable within a single run.
- Cost: +10.4% parameters and about +5% wall-clock per epoch.
- The plug-in block is channel preserving; upstream's `c1 -> c2` path is deliberately not
  reproduced, because stock `parse_model` assumes `c2 == ch[f]` for third-party modules.
- No multi-GPU run: DDP behaviour of the auxiliary term is untested.
