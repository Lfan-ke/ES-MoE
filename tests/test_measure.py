"""Re-measurement has to rebuild the model a run trained, whatever its arguments were rewritten to."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import measure  # noqa: E402


def test_a_run_is_rebuilt_from_the_config_its_arguments_name():
    core = SimpleNamespace(yaml={"yaml_file": "configs/other.yaml"})
    assert measure.trained_config("configs/yolo-master-n.yaml", core) == Path("configs/yolo-master-n.yaml")


def test_a_resumed_run_is_rebuilt_from_the_config_its_checkpoint_was_built_from():
    """Resuming rewrites `model` in args.yaml to the checkpoint, which no model can be built from."""
    core = SimpleNamespace(yaml={"yaml_file": "configs/yolo-master-n.yaml"})
    named = "/data/esmoe-toolkit/runs/m-baseline-e120-s2-p800f-fp32-fork/weights/last.pt"
    assert measure.trained_config(named, core) == Path("configs/yolo-master-n.yaml")
