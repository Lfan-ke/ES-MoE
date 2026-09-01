"""Command line: graft a config without writing a script, or report the installed versions."""

import argparse


def _at(value: str) -> int | str | list[int]:
    if value == "backbone_end":
        return value
    indices = [int(part) for part in value.split(",") if part]
    return indices[0] if len(indices) == 1 else indices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="esmoe", description="ES-MoE blocks for Ultralytics YOLO")
    commands = parser.add_subparsers(dest="command", required=True)

    graft_cmd = commands.add_parser("graft", help="write a model.yaml with ESMoE blocks grafted in")
    graft_cmd.add_argument("base", help="stock config, e.g. yolo11n.yaml")
    graft_cmd.add_argument("-o", "--out", required=True, help="path to write")
    graft_cmd.add_argument("-e", "--num-experts", type=int, default=4)
    graft_cmd.add_argument("-k", "--top-k", type=int, default=2)
    graft_cmd.add_argument("--at", type=_at, default="backbone_end", help="'backbone_end', an index, or i,j,k")
    graft_cmd.add_argument("--rewire", action="store_true", help="point later consumers of the insertion layer at it")
    commands.add_parser("info", help="print the installed esmoe, ultralytics and torch versions")

    args = parser.parse_args(argv)
    match args.command:
        case "graft":
            from .graft import graft

            options = {"num_experts": args.num_experts, "top_k": args.top_k, "rewire": args.rewire}
            graft(args.base, out=args.out, at=args.at, **options)
            print(args.out)
        case "info":
            import torch
            import ultralytics

            from . import __version__

            print(f"esmoe {__version__}\nultralytics {ultralytics.__version__}\ntorch {torch.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
