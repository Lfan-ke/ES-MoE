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


def test_a_failed_run_exits_non_zero_after_writing_its_record():
    """`queue.sh` logs FAILED only from the exit code; a record alone reads as done in its log."""
    source = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    body = source[source.index("def main():") :]
    record_written = body.index("out.write_text(json.dumps(record")
    exit_on_failure = body.index('if status != "success":')
    assert record_written < exit_on_failure, "the record must exist before the process says it failed"
    assert "raise SystemExit(1)" in body[exit_on_failure : exit_on_failure + 200]
    queue = (ROOT / "scripts" / "queue.sh").read_text(encoding="utf-8")
    assert 'wait "$pid" || echo' in queue and "FAILED" in queue
