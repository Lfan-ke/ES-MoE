# ES-MoE 0.1.3

## Fixed

- **Exported models silently ignored routing.** The block skipped experts whose gate was zero, which
  is a data-dependent decision: a tracer recorded the routing of the example input and the exported
  graph then used those experts for every future input. On a block whose routing follows the input,
  an ONNX export taken on one input differed from PyTorch by 0.2 on an input that routes elsewhere;
  it now differs by 1e-7. The block runs all experts while tracing and keeps the shortcut at run
  time, so nothing gets slower outside export.

  Anyone who exported a model with 0.1.0 - 0.1.2 should re-export.

## Added

- `scripts/verify.py`: correctness checks that unit tests cannot make - a real training run logging a
  positive auxiliary term, `weight=0` leaving the loss table untouched, checkpoint round trip,
  resume, several blocks training together, `val`/`predict`, and ONNX export.
- A regression test that exports a block whose routing follows the sign of its input and compares
  both branches against PyTorch.
