"""The run record has to say what the trainer did, not only what it was asked to do."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


def crashed(run: Path, epochs_done: int, saved_by: Path) -> Path:
    (run / "weights").mkdir(parents=True)
    rows = "".join(f"{epoch},{epoch * 165.5}\n" for epoch in range(1, epochs_done + 1))
    (run / "results.csv").write_text("epoch,time\n" + rows, encoding="utf-8")
    last = run / "weights" / "last.pt"
    torch.save({"epoch": epochs_done - 1, "train_args": {"save_dir": str(saved_by)}}, last)
    return last


def test_a_resumed_run_counts_the_epochs_and_seconds_its_checkpoint_holds(tmp_path):
    run = tmp_path / "runs" / "m-baseline-e120-s2-p800f-fp32-fork"
    last = crashed(run, 114, run)
    found = train.resumed(str(last), run)
    assert found["epochs_done"] == 114
    assert found["seconds_before"] == 114 * 165.5
    assert found["sha256"] == train.digest(last)


def test_a_checkpoint_from_another_run_is_refused(tmp_path):
    """The trainer would finish it in the directory it came from, under this run's record."""
    run = tmp_path / "runs" / "m-baseline-e120-s2-p800f-fp32-fork"
    other = tmp_path / "runs" / "_failed" / "m-baseline-e120-s2-p800f-fp32-fork-crash"
    with pytest.raises(SystemExit):
        train.resumed(str(crashed(other, 114, other)), run)
    with pytest.raises(SystemExit):
        train.resumed(str(crashed(run, 114, other)), run)


def test_a_resumed_run_counts_replays_against_the_epochs_left():
    source = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert 'max(seen["epochs_started"] - (args.epochs - done), 0)' in source
    assert "resume=resume" in source
