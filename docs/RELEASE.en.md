# Release notes

Current version **0.1.4**.

## Added

- **DDP support.** ultralytics launches multi-GPU workers from a generated file that imports only the trainer's module, so a worker never imported esmoe and failed to build the model with `KeyError: 'ESMoE'`. `attach_aux_loss` now makes `model.train()` pick a trainer subclass that lives in `esmoe.trainer`: importing it registers the block, restores the auxiliary-loss weight from the environment and patches `BaseModel.loss`. `scripts/verify.py` gains two checks: the real worker file trains one epoch in a fresh interpreter, and two gloo ranks each compute their own auxiliary term with router gradients agreeing after all-reduce. Limit: sparse dispatch needs `find_unused_parameters=True`, which `compile=True` switches off; that combination is unsupported.
- **`graft(..., rewire=True)`.** Renumbering moves references without retargeting them, so a head branch that names the old backbone end by index (YOLOv8's P5 lateral `[-1, 9] Concat`) keeps reading SPPF after the insertion and the block reaches P5 only through the top-down path. `rewire` points every such consumer at the block. Off by default to keep existing records comparable; the two wirings have not yet been compared under one budget.
- `scripts/train.py --patience` (default 0, early stopping off) and `IMGSZ` / `PATIENCE` / `ARMS` in `sweep.sh`, for runs under the repository protocol (imgsz 800, 120 epochs).
- `scripts/buckets.py`, COCO-style area buckets at maxDets 500; `scripts/routing.py`, expert usage on the validation set and whether routing follows object scale.
- Two half-precision tests: the auxiliary term stays finite and non-zero under bf16 autocast with parameters kept in fp32; the renormalised gate still sums to one in fp16.

## Feedback and iteration

Most of 0.1.4 answers the course community's first round of feedback: the evaluation caliber adopts the mentor's COCO-style 32²/96² buckets at maxDets 500 (`scripts/buckets.py`, source credited in the docs); `--patience` and `IMGSZ` exist to match the repository reproduction protocol (imgsz 800, 120 epochs, patience 0); the half-precision tests answer the ask to check that losses and gradients stay finite and consistent under FP32/AMP. The upstream loop closed as well: the `OptimizedMOE` tracing guard fix was merged into YOLO-Master (#241).

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
