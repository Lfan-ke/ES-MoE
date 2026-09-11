"""Every protocol run is published under a name that belongs to it alone.

The checkpoint, the curve and the bucket and routing analyses are all found by that name. Two hosts
once trained under one directory name; one record then carried the other run's checkpoint hash,
filled in by name, and nothing failed.
"""

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import backfill  # noqa: E402
from report import load, published, variant  # noqa: E402


def _protocol():
    return [r for r in load() if "@e120f1i800" in variant(r)]


def test_no_two_protocol_runs_share_a_published_name():
    shared = {name: n for name, n in Counter(published(r) for r in _protocol()).items() if n > 1}
    assert not shared, f"published by more than one run: {shared}"


def test_no_two_runs_claim_one_checkpoint_hash():
    hashes = Counter(r["artifact"]["sha256"] for r in _protocol() if r["artifact"].get("sha256"))
    assert not [h for h, n in hashes.items() if n > 1]


def test_backfill_does_not_fill_a_name_that_two_records_claim():
    records = [
        {"artifact": {"path": "/data/runs/shared/weights/best.pt"}},
        {"artifact": {"path": "/elsewhere/runs/shared/weights/best.pt"}},
        {"artifact": {"path": "/data/runs/alone/weights/best.pt"}},
    ]
    hashes = {"shared": ("a" * 64, "host.txt"), "alone": ("b" * 64, "host.txt")}
    assert backfill.unambiguous(hashes, records) == {"alone": ("b" * 64, "host.txt")}
