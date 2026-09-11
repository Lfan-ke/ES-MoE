"""The run record has to say what the trainer did, not only what it was asked to do."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train  # noqa: E402


def test_the_record_learns_what_the_trainer_did():
    callbacks: dict = {}
    model = SimpleNamespace(add_callback=lambda event, fn: callbacks.setdefault(event, []).append(fn))
    seen = train.watch(model)
    # A fork run that replayed its first epoch in FP32, after a first-epoch out-of-memory halved its batch.
    trainer = SimpleNamespace(amp=False, batch_size=16)
    for _ in range(121):
        for fn in callbacks["on_train_epoch_start"]:
            fn(trainer)
    for fn in callbacks["on_train_end"]:
        fn(trainer)
    assert seen == {"epochs_started": 121, "amp": False, "batch": 16}


def test_the_record_carries_every_fact_watch_collects():
    source = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    for key in ('"amp_at_end": seen["amp"]', '"batch_at_end": seen["batch"]', 'seen["epochs_started"]'):
        assert key in source
