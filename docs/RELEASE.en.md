# Release notes

Current version **2.0.0**: the block is no longer tied to Ultralytics. `uv add esmoe` installs torch
and nothing else, and the block drops into any torch model; the Ultralytics integration is
`uv add esmoe[yolo]` and works as it did in 1.x. A second implementation on Keras 3 runs the same
block on tensorflow, jax and torch.

## Changes

- **Torch and YOLO as equals.** The core needs torch alone and an `nn.Sequential` takes the block
  as written; it sizes itself on the first forward pass, so no width has to be declared. The
  auxiliary loss is read with `with esmoe.routing(net) as route:`, which clears the registry on the
  way in, so a block that did not run this step cannot report the step before.
- **Two knobs decide the experts.** `kernels` gives each expert a receptive field, an `int` or an
  `(h, w)` pair, and with it how many experts there are; `expert` gives a branch its structure, one
  factory for all or one each. One of three, five experts, strip convolutions, repeated kernels --
  each is one line.
- **A Keras 3 backend.** `esmoe.keras` holds the same block as a Keras layer, taking the same
  arguments, with the auxiliary loss going through Keras' own `add_loss`. `esmoe[tensorflow]` and
  `esmoe[jax]` bring the backend with them. It evaluates every expert: skipping an unrouted one
  depends on the data, which a static graph cannot express.
- **Grafting into a model that exists.** `esmoe.graft(model, at="layer3", mode="after", ...)` finds
  submodules by name, by several names or by a test, and puts the block `after`, `before` or in
  place of each. The same name still rewrites a config, `esmoe.graft("yolo11n.yaml", ...)`,
  dispatching on the first argument.
- **Three spellings in a config.** `ESMoE`, `ES_MoE` and `ES_MOE` all resolve to the block, and a
  config that already names one is taken as written: `equip` grafts nothing into it.
- **However the arguments are spelled.** A mapping among the positional arguments is read as
  `options`; `c`/`c1`/`ch`/`c2`/`e`/`k` are understood, and `experts` lands where its value fits.
  An unrecognised key warns once and is ignored, and a `top_k` above the expert count falls back to
  all of them -- neither costs a run at construction any more. A bad value still raises.
- **Python 3.12 and up**, which buys spellings like `type` aliases today.

## Moving from 1.x

| 1.x | 2.0 |
|:--:|:--:|
| `pip install esmoe` | `uv add esmoe` for the block, `uv add esmoe[yolo]` with the integration |
| `esmoe.inject_esmoe()` | `esmoe.register()`, the old name still works |
| `expert_kernel_sizes=` | `kernels=`, the old name is still accepted |
| `max_kernel_size=` | `max_kernel=`, the old name is still accepted |
| `from esmoe.inject import X` | `from esmoe.yolo.inject import X` |
| an unrecognised argument raised | it warns and is ignored |

Checkpoints written by 1.x still load.

## Before that, 1.x

1.0.0 was the first stable release: the compatibility matrix across seven backbone generations, the
full-protocol experiments and judgment lines, and a bilingual documentation site. 1.0.1 fixed the
closure audit's substring matching, the dataset statistics on out-of-bounds boxes, charts reading
runs that were never registered, and several display issues on the site. 1.1.0 added the
[try it](demo.md) page -- up to three models on one image side by side, per-block expert scores for
the ones that carry a block, inference in the reader's own browser -- and the export script
`scripts/demo_export.py`.

## Earlier point releases

The 0.1 series introduced the four entry points `inject_esmoe`, `graft`, `attach_aux_loss` and
`collect_aux_loss` with three-seed evidence, then fixed the config handed to `YOLO()` without an
`out` path, the training-log header on 8.4.13x, and moved the package from `src/` to the repository
root.

## Install

```bash
uv add esmoe              # the block, for any torch model
uv add esmoe[yolo]        # ... and the Ultralytics integration
pip install esmoe         # if you insist
```
