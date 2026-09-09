"""Read the block settings out of saved checkpoints, on the host that holds them.

A record states what a run was asked for; the checkpoint holds what it got. Runs that predate a
setting have no attribute for it, which is itself the answer. Pipe the output to a file and feed it
to `scripts/backfill.py --settings`.

    uv run python scripts/blockspec.py > settings-a.txt
"""

import json
import sys
from pathlib import Path

import torch

import esmoe

ROOT = Path(__file__).resolve().parents[1]


def spec(path: Path) -> dict:
    model = torch.load(path, map_location="cpu", weights_only=False)["model"]
    found = list(esmoe.blocks(model))
    if not found:
        return {"balance": "none", "blocks": 0}
    first = found[0]
    # A block pickled before a setting existed carries no attribute for it, and that absence is
    # the answer: it ran with whatever the default was then, which is what `SETTINGS` still holds.
    for key, default in esmoe.SETTINGS.items():
        if not hasattr(first, key):
            setattr(first, key, default)
    return first.spec() | {"blocks": len(found)}


def main() -> int:
    for path in sorted(ROOT.glob("runs/*/weights/best.pt")):
        try:
            print(f"{path.parts[-3]}\t{json.dumps(spec(path))}", flush=True)
        except Exception as exc:  # a checkpoint that will not load is a fact worth printing
            print(f"{path.parts[-3]}\tERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
