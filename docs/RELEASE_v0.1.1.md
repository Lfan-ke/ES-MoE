# esmoe 0.1.1

Bug fix and packaging follow-up to the first release.

## Fixed

- `equip()` without an `out` path handed the grafted config to `YOLO()` as a dict, which only loads
  models by path. The config now lands in a temporary file, so the one-line entry point works with no
  arguments beyond the base config.

## Changed

- Project URLs follow the repository rename to `Lfan-ke/ES-MoE`; the old paths still redirect.
- `torchvision` is pinned to the same CPU index as `torch` for development, because a PyPI
  torchvision against a CPU torch loses `torchvision::nms` on Linux. Published metadata is unaffected.

## Added

- A Colab quick start (`notebooks/quickstart.ipynb`) that installs, equips, trains on `coco8` and
  shows the `esmoe_aux` column, then swaps in a custom expert and balancing objective. CI executes it
  on CPU so it cannot rot as ultralytics moves.
