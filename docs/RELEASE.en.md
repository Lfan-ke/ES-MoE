# Release notes

Current version **0.1.5**.

## Added

- **Parameter parity with upstream's `ES_MOE`.** The block now takes everything upstream's constructor does: `out_channels`, `top_k=None` for every expert, `sparse_inference` (upstream's `use_sparse_inference`), and `dynamic_threshold` (0.4 upstream, 0.0 here so nothing is pruned -- every record in `results/` was measured that way). Even kernel sizes step down to odd and cap at `max_kernel_size`, so a pruned checkpoint's kernels reload, and `num_experts`, `reduction`, `dynamic_threshold` and `max_kernel_size` are validated at construction against the same bounds.
- **Four balancing objectives.** `gshard_balance` (the default, matching upstream: it reads the gate, after the top-k mask and renormalisation), `switch_balance`, `master_balance` (the paper's eq. 13, an affine map `(L-1)/E^2` of the first), and `gshard_probs_balance`, which reads the raw probabilities and exists to isolate that one variable. On the command line: `--balance {switch,gshard,master,gshard_probs}`.
- **Upstream's block layout and block internals.** `graft(at="backbone_stages")` places one block after each backbone stage, four on all seven generations, with the stage boundaries derived from the downsampling layers. `out_norm` adds the `BatchNorm + SiLU` after the weighted sum (the paper's eq. 2 `Norm`). `dense_training` runs every expert while training, weighting unrouted ones by zero so their normalisation statistics keep moving. All three are off by default so the runs already in `results/` still reproduce.
- **`scripts/blockspec.py`** reads back the block settings in force from any checkpoint. **`scripts/backfill.py`** fills and audits a record's config hash, dataset split sizes, GPU-hours and artifact checksum; `--settings` rewrites a record to match its checkpoint and names the change inside the record. **`scripts/queue.sh`** is in the repository, with a test holding it and `scripts/train.py` to the same run names.
- `scripts/report.py` puts a 95% confidence interval on each paired delta; `scripts/routing.py` analyses every block rather than only the first.

## Fixed

- **Block settings never reached the model that trained.** The trainer rebuilds the model from `model.yaml`, so an objective or a switch set on the blocks after `YOLO(cfg)` returns disappears with the discarded instance -- no error, no trace. `--balance`, `--out-norm` and `--dense-training` were all inert while the record repeated the command line. `graft()` and `equip()` now write the settings into the config, `ESMoE` takes them as an options mapping and resolves an objective by name, and `scripts/train.py` records the block settings read off the trained model, refusing to start when they disagree with the request. **If you set these through `equip()` + `configure()` on 0.1.4, what trained was the constructor default; check with `scripts/blockspec.py`.**
- **`dynamic_threshold` would not trace.** The mask used `scatter_` with a Python bool, for which a tracer has no op, breaking `torch.jit.trace` and every export built on it. It is tensor arithmetic now, recomputed per input.
- **The routing statistics counted a warmup forward.** On an accelerator ultralytics runs one dummy forward before the first real batch, and the router hook captured that row too: 549 rows against 548 images. It never fired on CPU, which is why it went unseen.
- **Two runs of one arm truncated each other's config.** The grafted `configs/*.yaml` carried no seed, so two lanes training the same arm wrote one file; one truncated it while the other was reading, and the reader died on `KeyError: 'backbone'`.
- **`report.py` folded hardware together.** The grouping key ignored hardware, so the same configuration measured on two machines counted as repeats of one cell. Hardware stack and block configuration are both part of the key now.
- **`buckets.py` could not evaluate on some accelerators.** `val()` fuses conv+bn inside an inference-mode context and some builds refuse to view an inference tensor. Fusing beforehand leaves that path nothing to do; the arithmetic is unchanged.
- **Router logits were not clamped.** Under mixed precision a runaway logit reaches the softmax as inf and takes the whole gate to NaN. They are clamped to `[-30, 30]` before an fp32 softmax, as upstream does.

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
