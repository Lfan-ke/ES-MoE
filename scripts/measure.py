"""Every arm's final weights measured again, by one piece of code, under one protocol.

A training curve is measured by the trainer that produced it, so arms trained on different
frameworks are compared through whatever each of them validated with. This loads each run's last
checkpoint -- the EMA weights, as the trainer validated them -- rebuilds the model from the config
that run trained, and evaluates every arm the same way.

The fork's arms need the fork on the path, this package's arms need the official ultralytics, so a
card is measured in two passes into the same directory:

    PYTHONPATH=/data/yolo-master python3 scripts/measure.py runs/*-fork --out results/measured
    python3 scripts/measure.py runs/yolo-master-n-esmoe-* runs/yolo-master-n-baseline-*p800c --out results/measured
"""

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")


def rebuild(run: Path, cfg: Path, nc: int, out: Path):
    """The run's EMA weights on a model built from its config.

    A checkpoint written inside `torch.inference_mode` carries tensors that cannot be fused, which
    is where a plain `YOLO(last.pt).val()` stops; rebuilding from the config sidesteps that and
    also takes the block settings from the config rather than from the pickled module.
    """
    import torch
    from ultralytics.nn.tasks import DetectionModel

    saved = torch.load(run / "weights" / "last.pt", map_location="cpu", weights_only=False)
    core = saved.get("ema") or saved["model"]
    state = {k: torch.from_numpy(v.detach().cpu().float().numpy().copy()) for k, v in core.state_dict().items()}
    model = DetectionModel(str(cfg), ch=3, nc=nc, verbose=False)
    # The fork's trainer registers `_mixture_loss_ema_buf` on the model it trains, a running scale
    # for the auxiliary term that a freshly built model has no slot for. Anything the config does
    # build has to arrive, so a weight going missing still stops here.
    fresh = model.state_dict()
    if missing := [key for key in fresh if key not in state]:
        raise SystemExit(f"{run.name}: checkpoint is missing {missing[:5]}")
    dropped = [key for key in state if key not in fresh]
    model.load_state_dict({key: value for key, value in state.items() if key in fresh})
    model.names = core.names
    path = out / "rebuilt" / f"{run.name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model, "train_args": {}}, path)
    return path, sum(p.numel() for p in model.parameters()), dropped


def measure(run: Path, args, out: Path) -> dict:
    import ultralytics
    from ultralytics import YOLO

    trained = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    cfg = Path(trained["model"])
    data = Path(args.data or trained["data"])
    nc = len(yaml.safe_load(data.read_text(encoding="utf-8"))["names"])
    path, parameters, dropped = rebuild(run, cfg, nc, out)
    metrics = YOLO(str(path)).val(
        data=str(data),
        imgsz=args.imgsz or trained["imgsz"],
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
        project=str(out / "val"),
        name=run.name,
        exist_ok=True,
    )
    return {
        "run": run.name,
        "config": str(cfg),
        "framework": ultralytics.__file__,
        "parameters": parameters,
        "dropped": dropped,
        "imgsz": args.imgsz or trained["imgsz"],
        "metrics": {key: round(float(metrics.results_dict[key]), 5) for key in KEYS},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="results/measured")
    p.add_argument("--data", default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    import esmoe

    esmoe.inject_esmoe()  # only registers the block name; a run without one is unaffected
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.runs:
        run = Path(name)
        record = measure(run, args, out)
        (out / f"{run.name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(record["run"], json.dumps(record["metrics"]), flush=True)


if __name__ == "__main__":
    main()
