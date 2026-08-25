# esmoe-toolkit

Drop-in ES-MoE (expert-sparse Mixture-of-Experts) block for Ultralytics YOLO. Turns the community "single-seed accuracy bump" into an installable, budget-fair, aux-loss-aware module.

Status: early scaffold, interfaces only. See docs/ROADMAP.md.

## Install

    pip install -e .

## Target API

    from esmoe import inject_esmoe
    inject_esmoe()          # register ESMoE so ultralytics YAML can reference it
    # then use `ESMoE` as a module in your model.yaml, across YOLOv8 / YOLO11 / YOLOv12

## License

AGPL-3.0-only, matching the Ultralytics ecosystem it builds on.
