# Release notes

Current version **0.1.3**.

## Fixed

- **Exported models no longer ignore routing.** The block skips experts whose gate is zero, which is a data-dependent decision: a tracer records the routing of the example input, and the exported graph then uses those same experts for every later input. On a block whose routing follows its input, an ONNX export taken on one input differed from PyTorch by 0.2 on an input that routes elsewhere; it now differs by 1e-7. The block runs all experts while tracing and keeps the shortcut at run time, so nothing outside export gets slower.

    Anyone who exported a model with 0.1.0 through 0.1.2 should re-export.

## Added

- `scripts/verify.py`: correctness checks that unit tests cannot make — a real training run logging a positive auxiliary term, `weight=0` leaving the loss table untouched, checkpoint round trip, resume, several blocks training together, `val` and `predict`, and ONNX export.
- A regression test that exports a block whose routing follows the sign of its input and compares both branches against PyTorch.

## Earlier point releases

0.1.0 was the first release, with the four entry points `inject_esmoe`, `graft`, `attach_aux_loss` and `collect_aux_loss`, plus the selection argument and three-seed evidence. 0.1.1 fixed `equip()` handing the grafted config to `YOLO()` as a dict when no `out` path was given, and added the Colab quick start. 0.1.2 fixed the training log header on releases from 8.4.13x, where `loss_names` is empty at `on_train_start`, and moved the package from `src/` to the repository root.

## Install

    pip install esmoe
