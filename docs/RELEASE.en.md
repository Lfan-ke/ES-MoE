# Release notes

Current version **0.1.5**.

## Added

- **`gshard_balance`, the term upstream's `ES_MOE` actually optimises.** YOLO-Master's `ES_MOE` uses the GShard form `N * sum(usage^2)` at `balance_loss_coeff = 1.0`; this package defaults to the Switch form `E * sum(p_i f_i)` at `weight = 0.01`. Both are now selectable: `ESMoE(..., balance=esmoe.gshard_balance)`, or `scripts/train.py --balance gshard`. On a real routing record the upstream pairing puts 31x more balancing gradient on the mean probabilities than this package's default - a larger difference than the choice of formula, and the comparison's criteria are registered on the judgment-lines page ahead of the results.

## Fixed

- **The routing statistics counted a warmup forward.** On an accelerator ultralytics runs one dummy forward before the first real batch, and the router hook captured that row too: 549 rows against 548 images. It never fired on CPU, which is why it went unseen.
- **Two runs of one arm truncated each other's config.** The grafted `configs/*.yaml` carried no seed, so two lanes training the same arm wrote one file; one truncated it while the other was reading, and the reader died on `KeyError: 'backbone'`. Earlier batches escaped it only because concurrent lanes always happened to run different arms.
- **`report.py` folded hardware together.** The grouping key ignored hardware, so the same configuration measured on two machines counted as repeats of one cell. The stack is now part of the key and a rerun on another host forms its own group.
- **`buckets.py` could not evaluate on some accelerators.** `val()` fuses conv+bn inside an inference-mode context and some builds refuse to view an inference tensor. Fusing beforehand leaves that path nothing to do; the arithmetic is unchanged.

## Feedback and iteration

Most of 0.1.4 answers the first round of user feedback: the evaluation caliber moves to COCO-style 32²/96² buckets at maxDets 500 (`scripts/buckets.py`, source credited in the docs); `--patience` and `IMGSZ` exist to match the repository reproduction protocol (imgsz 800, 120 epochs, patience 0); the half-precision tests answer the ask to check that losses and gradients stay finite and consistent under FP32/AMP. The upstream loop closed as well: the `OptimizedMOE` tracing guard fix was merged into YOLO-Master (#241).

## Fixed

- `scripts/report.py` now keys groups by image size as well. Records of the same schedule at different resolutions used to be averaged into one row, which is exactly what the documentation promised would not happen.

## Earlier: 0.1.3

- **Exported models no longer ignore routing.** The block skips experts whose gate is zero, which is a data-dependent decision: a tracer records the routing of the example input, and the exported graph then uses those same experts for every later input. On a block whose routing follows its input, an ONNX export taken on one input differed from PyTorch by 0.2 on an input that routes elsewhere; it now differs by 1e-7. The block runs all experts while tracing and keeps the shortcut at run time, so nothing outside export gets slower.

    Anyone who exported a model with 0.1.0 through 0.1.2 should re-export.

## Added

- `scripts/verify.py`: correctness checks that unit tests cannot make — a real training run logging a positive auxiliary term, `weight=0` leaving the loss table untouched, checkpoint round trip, resume, several blocks training together, `val` and `predict`, and ONNX export.
- A regression test that exports a block whose routing follows the sign of its input and compares both branches against PyTorch.

## Earlier point releases

0.1.0 was the first release, with the four entry points `inject_esmoe`, `graft`, `attach_aux_loss` and `collect_aux_loss`, plus the selection argument and three-seed evidence. 0.1.1 fixed `equip()` handing the grafted config to `YOLO()` as a dict when no `out` path was given, and added the Colab quick start. 0.1.2 fixed the training log header on releases from 8.4.13x, where `loss_names` is empty at `on_train_start`, and moved the package from `src/` to the repository root.

## Install

    pip install esmoe
