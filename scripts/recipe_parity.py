"""Train YOLO-Master's fork and this package from one set of weights, step for step, and compare.

The same-configuration comparison claims that arm B trains the way arm A does. Unit tests hold the
recipe's pieces to upstream's source; this holds the whole training loop to upstream's trainer. Both
start from the same weights, read the same batches in the same order and run on CPU in FP32, where the
fork's mixed-precision fallback cannot fire; with a batch count the accumulation divides, the fork's
extra end-of-epoch step does not fire either. Whatever still differs is the implementation.

A routed model magnifies rounding: near-tied router scores flip which experts a sample takes. So a gap
between A and B is read against the gap between a run and itself with one router layer nudged by a
float-sized amount (`--perturb`).

    PYTHONPATH=<fork> python scripts/recipe_parity.py init --out parity          # fork: A and A0 weights
    python scripts/recipe_parity.py map --out parity                             # official: B and C from them
    PYTHONPATH=<fork> python scripts/recipe_parity.py trace --arm upstream --out parity
    PYTHONPATH=<fork> python scripts/recipe_parity.py trace --arm forkbase --out parity
    python scripts/recipe_parity.py trace --arm recipe --out parity
    python scripts/recipe_parity.py trace --arm baseline --out parity
    PYTHONPATH=<fork> python scripts/recipe_parity.py trace --arm upstream --perturb 1e-6 --out parity
    python scripts/recipe_parity.py trace --arm recipe --perturb 1e-6 --out parity
    python scripts/recipe_parity.py compare --out parity                         # parity/parity.md
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORK_YAML = "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml"
ARMS = {
    "upstream": ("fork", "A", "init-upstream.pt"),
    "forkbase": ("fork", "A0", "init-forkbase.pt"),
    "recipe": ("official", "B", "init-recipe.pt"),
    "baseline": ("official", "C", "init-baseline.pt"),
}
PAIRS = (
    ("upstream", "recipe", "A against B"),
    ("forkbase", "baseline", "A0 against C"),
    ("upstream", "upstream-perturbed", "A against A with one router layer nudged"),
    ("recipe", "recipe-perturbed", "B against B with one router layer nudged"),
)
BLOCKS = (3, 6, 9, 12)
# ES_MOE's router is two 1x1 convolutions over the pooled map; ESMoE's is two Linear layers, the same
# operation with the two trailing unit dimensions dropped.
RENAMES = (
    (r"\.routing\.routing_network\.0\.", ".router.2."),
    (r"\.routing\.routing_network\.2\.", ".router.4."),
    (r"\.experts\.(\d+)\.conv\.depthwise\.", r".experts.\1.dw."),
    (r"\.experts\.(\d+)\.conv\.pointwise\.", r".experts.\1.pw."),
    (r"\.experts\.(\d+)\.conv\.bn\.", r".experts.\1.bn."),
)


def settings(args) -> dict:
    return {
        "data": str(ROOT / "configs" / "visdrone.yaml"),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        # 192 of 6471 training images: 12 batches of 16, which accumulation over 4 divides exactly.
        "fraction": args.fraction,
        "workers": args.workers,
        "device": "cpu",
        "seed": 0,
        "deterministic": True,
        # What optimizer="auto" resolves to on the protocol's iteration count, spelled out because a
        # short run would otherwise resolve to AdamW and never exercise Muon's router exclusion.
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "momentum": 0.9,
        "warmup_bias_lr": 0.0,
        "amp": False,
        "val": False,
        "plots": False,
        "patience": 0,
        "exist_ok": True,
        "verbose": False,
    }


def save_checkpoint(model, path: Path) -> None:
    import torch

    torch.save({"model": model, "train_args": {}}, path)


def init(args) -> None:
    """On the fork: build A and A0 as its parser does and keep their weights."""
    import torch
    import ultralytics
    from ultralytics.nn.tasks import DetectionModel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fork = Path(ultralytics.__file__).resolve().parents[1]
    for arm, yaml in (("upstream", fork / FORK_YAML), ("forkbase", ROOT / "configs" / "yolo-master-n.yaml")):
        torch.manual_seed(0)
        model = DetectionModel(str(yaml), ch=3, nc=10, verbose=False)
        save_checkpoint(model, out / ARMS[arm][2])
        torch.save(model.state_dict(), out / f"{arm}.state.pt")
        print(arm, sum(p.numel() for p in model.parameters()), "parameters")


def renamed(state: dict) -> dict:
    moved = {}
    for key, value in state.items():
        new = key
        if any(key.startswith(f"model.{b}.") for b in BLOCKS):
            for pattern, repl in RENAMES:
                new = re.sub(pattern, repl, new)
            if ".router." in new and new.endswith(".weight"):
                value = value.flatten(1)
        moved[new] = value
    return moved


def map_weights(args) -> None:
    """On official ultralytics: B and C, holding exactly A's and A0's weights."""
    import torch
    from ultralytics.nn.tasks import DetectionModel

    import esmoe

    esmoe.inject_esmoe()
    out = Path(args.out)
    for arm, source, yaml in (
        ("recipe", "upstream", "yolo-master-n-esmoe.yaml"),
        ("baseline", "forkbase", "yolo-master-n.yaml"),
    ):
        model = DetectionModel(str(ROOT / "configs" / yaml), ch=3, nc=10, verbose=False)
        state = torch.load(out / f"{source}.state.pt", map_location="cpu")
        state = renamed(state) if arm == "recipe" else state
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(f"{arm}: missing {missing[:5]}, unexpected {unexpected[:5]}")
        save_checkpoint(model, out / ARMS[arm][2])
        print(arm, "holds", source, "weights:", len(state), "tensors")


def router_layer(core, fork: bool):
    block = core.model[BLOCKS[0]]
    return block.routing.routing_network[0] if fork else block.router[2]


def snapshot(trainer, label: str, fork: bool) -> dict:
    """What one iteration left behind, in terms both trainers can be read in."""
    from ultralytics.utils.torch_utils import unwrap_model

    core = unwrap_model(trainer.model)
    items = trainer.loss_items
    groups = [g for g in trainer.optimizer.param_groups if g["params"]]
    row = {
        "epoch": trainer.epoch,
        "items": [float(v) for v in (items.values() if isinstance(items, dict) else items)],
        "lr": sorted(round(float(g["lr"]), 12) for g in groups),
        "first_conv": float(core.model[0].conv.weight.detach().double().sum()),
    }
    if label in ("A", "B"):
        routers = [core.model[b].routing if fork else core.model[b].router for b in BLOCKS]
        owned = {id(p) for router in routers for p in router.parameters()}
        experts = [p for b in BLOCKS for p in core.model[b].experts.parameters()]
        buffer = getattr(core, "_mixture_loss_ema_buf", None)
        row["router"] = float(router_layer(core, fork).weight.detach().double().sum())
        row["router_lr"] = sorted(
            {round(float(g["lr"]), 12) for g in groups if any(id(p) in owned for p in g["params"])}
        )
        row["experts_training"] = sum(1 for p in experts if p.requires_grad)
        row["scale"] = float(buffer[0]) if fork and buffer is not None else getattr(core, "_esmoe_aux_scale", None)
    return row


def trace(args) -> None:
    import torch
    import ultralytics
    from ultralytics import YOLO

    stack, label, init_file = ARMS[args.arm]
    fork = importlib.util.find_spec("ultralytics.nn.mixture_registry") is not None
    if fork != (stack == "fork"):
        raise SystemExit(f"--arm {args.arm} needs the {stack} ultralytics, found {ultralytics.__file__}")
    if args.perturb and label not in ("A", "B"):
        raise SystemExit("--perturb nudges a router, which only the arms with blocks have")
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    name = args.arm + ("-perturbed" if args.perturb else "")
    if stack == "official":
        import esmoe

        esmoe.inject_esmoe()
    model = YOLO(str(out / init_file))
    if args.perturb:
        with torch.no_grad():
            router_layer(model.model, fork).weight.mul_(1.0 + args.perturb)
    if args.arm == "recipe":
        esmoe.attach_aux_loss(model, weight=1.0, recipe="upstream")
    rows, batches, start = [], [], {}

    def started(trainer):
        from ultralytics.utils.torch_utils import unwrap_model

        start["first_conv"] = float(unwrap_model(trainer.model).model[0].conv.weight.detach().double().sum())
        preprocess = trainer.preprocess_batch

        # Whether both trainers were fed the same images is the first thing a gap has to rule out.
        def fingerprinted(batch):
            batches.append(
                {
                    "images": float(batch["img"].double().sum()),
                    "instances": int(batch["cls"].shape[0]),
                    "first": Path(batch["im_file"][0]).name,
                }
            )
            return preprocess(batch)

        trainer.preprocess_batch = fingerprinted

    def ended(trainer):
        rows.append(snapshot(trainer, label, fork) | {"batch": batches[-1] if batches else None})

    model.add_callback("on_train_start", started)
    model.add_callback("on_train_batch_end", ended)
    model.train(project=str(out / "runs"), name=name, **settings(args))
    record = {
        "arm": args.arm,
        "label": label,
        "perturb": args.perturb,
        "ultralytics": ultralytics.__file__,
        "torch": torch.__version__,
        "settings": settings(args),
        "start": start,
        "rows": rows,
    }
    (out / f"trace-{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(name, len(rows), "iterations traced")


def epoch_line(epoch: int, a: list[dict], b: list[dict]) -> str:
    n = min(len(a), len(b))
    pairs = list(zip(a[:n], b[:n], strict=True))
    gaps = [(abs(x - y), abs(x)) for ra, rb in pairs for x, y in zip(ra["items"], rb["items"], strict=True)]
    loss_gap = max(gap for gap, _ in gaps)
    relative = max(gap / max(size, 1e-12) for gap, size in gaps)
    conv = max(abs(ra["first_conv"] - rb["first_conv"]) for ra, rb in pairs)
    lr_equal = all(ra["lr"] == rb["lr"] for ra, rb in pairs)
    experts = f"{a[n - 1].get('experts_training', '-')} / {b[n - 1].get('experts_training', '-')}"
    scales = [abs(ra["scale"] - rb["scale"]) for ra, rb in pairs if None not in (ra.get("scale"), rb.get("scale"))]
    scale = f"{max(scales):.2e}" if scales else "-"
    return f"| {epoch} | {n} | {loss_gap:.2e} | {relative:.2e} | {conv:.2e} | {lr_equal} | {experts} | {scale} |"


def compare(args) -> None:
    out = Path(args.out)
    traces = {}
    for name in {name for pair in PAIRS for name in pair[:2]}:
        path = out / f"trace-{name}.json"
        if path.exists():
            traces[name] = json.loads(path.read_text(encoding="utf-8"))
    lines = ["# Step for step: YOLO-Master's trainer against this package", ""]
    for left, right, what in PAIRS:
        if left not in traces or right not in traces:
            continue
        a, b = traces[left]["rows"], traces[right]["rows"]
        nudge = traces[right].get("perturb") or traces[left].get("perturb")
        lines += [
            f"## {what}" + (f" (relative {nudge:g})" if nudge else ""),
            "",
            f"{len(a)} and {len(b)} iterations traced.",
            "",
            "| epoch | iterations | largest loss gap | largest relative gap | first-conv gap | lr groups equal "
            f"| experts training ({left} / {right}) | aux scale gap |",
            "|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|",
        ]
        for epoch in sorted({r["epoch"] for r in a}):
            lines.append(
                epoch_line(epoch, [r for r in a if r["epoch"] == epoch], [r for r in b if r["epoch"] == epoch])
            )
        starts = traces[left].get("start", {}), traces[right].get("start", {})
        lines += [
            "",
            f"Starting first-conv checksum: {starts[0].get('first_conv')} and {starts[1].get('first_conv')}.",
            "",
            f"| iteration | same images | loss items ({left}) | loss items ({right}) | first conv |",
            "|:--:|:--:|:--:|:--:|:--:|",
        ]
        for i, (ra, rb) in enumerate(zip(a[:8], b[:8], strict=True)):
            same = ra.get("batch") == rb.get("batch") if ra.get("batch") else "-"
            items = [", ".join(f"{x:.5f}" for x in r["items"]) for r in (ra, rb)]
            lines.append(
                f"| {i} | {same} | {items[0]} | {items[1]} | {ra['first_conv']:.6f} / {rb['first_conv']:.6f} |"
            )
        lines.append("")
    (out / "parity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("init", "map", "trace", "compare"))
    p.add_argument("--arm", choices=tuple(ARMS))
    p.add_argument("--out", default="parity")
    p.add_argument("--perturb", type=float, default=0.0, help="scale the first router layer by 1 + this")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--fraction", type=float, default=0.0297)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=32)
    args = p.parse_args()
    {"init": init, "map": map_weights, "trace": trace, "compare": compare}[args.step](args)


if __name__ == "__main__":
    main()
