"""Command line: graft a config without writing a script, or report what is installed."""

import argparse
from importlib.util import find_spec


def __at(value: str) -> int | str | list[int]:
    if value in ("backbone_end", "backbone_stages"):
        return value
    indices = [int(part) for part in value.split(",") if part]
    return indices[0] if len(indices) == 1 else indices


def __kernels(value: str) -> list[int | tuple[int, int]]:
    """``3,5,7`` or ``3x3,5x5,1x9``."""
    found = []
    for part in value.split(","):
        height, cross, width = part.strip().partition("x")
        found.append((int(height), int(width)) if cross else int(height))
    return found


def __graft(args: argparse.Namespace) -> None:
    from .yolo.graft import graft

    options = {"num_experts": args.num_experts, "top_k": args.top_k, "rewire": args.rewire, "kernels": args.kernels}
    # Only what was asked for: a default written out would still be a key the config carries.
    options |= {"balance": args.balance, "expert": args.expert, "out_channels": args.out_channels}
    options |= {"out_norm": args.out_norm or None, "dense_training": args.dense_training or None}
    graft(args.base, out=args.out, at=args.at, **{key: value for key, value in options.items() if value is not None})
    print(args.out)


def __info() -> None:
    import torch

    from . import (
        BACKENDS,
        INTEGRATIONS,
        __version__,
    )

    print(f"esmoe {__version__}\ntorch {torch.__version__}")
    for name in (*BACKENDS, *INTEGRATIONS):
        found = find_spec("ultralytics" if name in INTEGRATIONS else name)
        print(f"{name} {'installed' if found else 'absent'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="esmoe", description="ES-MoE blocks for torch models and Ultralytics YOLO")
    commands = parser.add_subparsers(dest="command", required=True)

    graft = commands.add_parser("graft", help="write a model.yaml with ESMoE blocks grafted in")
    graft.add_argument("base", help="stock config, e.g. yolo11n.yaml")
    graft.add_argument("-o", "--out", required=True, help="path to write")
    graft.add_argument("-e", "--num-experts", type=int, default=4)
    graft.add_argument("-k", "--top-k", type=int, default=2)
    graft.add_argument("--kernels", type=__kernels, help="per-expert receptive fields, 3,5,7 or 3x3,1x9")
    graft.add_argument(
        "--at", type=__at, default="backbone_end", help="'backbone_end', 'backbone_stages', an index, or i,j,k"
    )
    graft.add_argument("--rewire", action="store_true", help="point later consumers of the insertion layer at it")
    graft.add_argument("--out-channels", type=int, help="widen the blocks to this literal width")
    graft.add_argument("--balance", help="switch, gshard, master, gshard_probs, or module:qualname")
    graft.add_argument("--expert", help="module:qualname of a custom expert factory")
    graft.add_argument("--out-norm", action="store_true", help="normalise the mixed output, as upstream does")
    graft.add_argument("--dense-training", action="store_true", help="run every expert while training")
    commands.add_parser("info", help="print the installed version, backends and integrations")

    args = parser.parse_args(argv)
    match args.command:
        case "graft":
            __graft(args)
        case "info":
            __info()
    return 0


__all__ = ["main"]
