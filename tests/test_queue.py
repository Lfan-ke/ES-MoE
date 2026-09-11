"""The queue script and the trainer have to name a run the same way.

`scripts/queue.sh` rebuilds the run directory name to decide whether a job is already finished.
If it and `scripts/train.py` disagree, a resumed run starts a second directory and two lanes end
up writing one experiment -- which is how a shared grafted config got truncated once already.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train  # noqa: E402

CASE = re.compile(r'^\s*(\w+)\)\s*flag="([^"]*)";\s*arch="([^"]*)"(;\s*fork=1)?', re.MULTILINE)
ARMS = CASE.findall((ROOT / "scripts" / "queue.sh").read_text(encoding="utf-8"))


def test_queue_offers_every_arm_the_protocol_uses():
    assert {name for name, *_ in ARMS} == {
        "baseline",
        "esmoe",
        "rewire",
        "gshard",
        "master",
        "stages",
        "norm",
        "dense",
        "recipe",
        "upstream",
        "forkbase",
    }


def test_only_the_arms_on_yolo_masters_fork_train_there():
    assert {name for name, _, _, fork in ARMS if fork} == {"upstream", "forkbase"}


# "0.0" and "1.50" are the shapes a hand-written job line takes; the runner normalises both.
@pytest.mark.parametrize("weight", ["0.01", "1.5", "0.0", "1.50", "0.0025"])
@pytest.mark.parametrize("arm", ARMS, ids=lambda arm: arm[0])
def test_queue_and_trainer_agree_on_the_run_name(arm, weight):
    name, flags, arch, _ = arm
    argv = [*flags.split(), "--base", "yolov8n.yaml"]
    weighted = "--esmoe" in flags
    if weighted:
        argv += ["--aux-weight", weight]
    # The runner pipes the weight through `printf %g`, which is what python's `:g` produces.
    normalised = f"{float(weight):g}"
    expected = arch + ("" if not weighted or normalised == "0.01" else f"-w{normalised}")
    assert train.architecture(train.build_parser().parse_args(argv)) == expected


def test_only_yolo_masters_own_blocks_need_its_fork():
    parse = train.build_parser().parse_args
    fork = "yolo-master@acce839c"
    assert train.mismatch(parse(["--upstream"]), "ultralytics")
    assert train.mismatch(parse(["--upstream"]), fork) is None
    assert train.mismatch(parse([]), fork) is None, "the fork trains its own baseline"
    assert train.mismatch(parse(["--esmoe", "--grafted"]), "ultralytics") is None
    assert train.mismatch(parse(["--grafted"]), "ultralytics")


def test_a_config_holding_its_blocks_is_not_named_twice_and_a_fork_run_says_so():
    parse = train.build_parser().parse_args
    grafted = parse(["--esmoe", "--grafted", "--recipe", "upstream", "--aux-weight", "1", "--base", "c/m-esmoe.yaml"])
    assert train.run_name(grafted, fork=False) == "m-esmoe-upstream-w1-e10-s0"
    plain = parse(["--base", "c/m.yaml", "--epochs", "120", "--seed", "2", "--tag=-p800h3"])
    assert train.run_name(plain, fork=True) == "m-baseline-e120-s2-p800h3-fork"
