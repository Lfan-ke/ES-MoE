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

CASE = re.compile(r'^\s*(\w+)\)\s*flag="([^"]*)";\s*arch="([^"]*)"', re.MULTILINE)
ARMS = CASE.findall((ROOT / "scripts" / "queue.sh").read_text(encoding="utf-8"))


def test_queue_offers_every_arm_the_protocol_uses():
    assert {name for name, _, _ in ARMS} == {
        "baseline",
        "esmoe",
        "rewire",
        "gshard",
        "master",
        "stages",
        "norm",
        "dense",
    }


@pytest.mark.parametrize("weight", ["0.01", "1.5"])
@pytest.mark.parametrize("arm", ARMS, ids=lambda arm: arm[0])
def test_queue_and_trainer_agree_on_the_run_name(arm, weight):
    name, flags, arch = arm
    argv = [*flags.split(), "--base", "yolov8n.yaml"]
    if name != "baseline":
        argv += ["--aux-weight", weight]
    expected = arch + ("" if name == "baseline" or weight == "0.01" else f"-w{float(weight):g}")
    assert train.architecture(train.build_parser().parse_args(argv)) == expected
