"""The same-configuration table has to sort runs into the right arm and pair them inside a card."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import report  # noqa: E402
import same_config  # noqa: E402


def record(arm, seed, value, framework="ultralytics", torch="2.8.0+metax3.3.0.2", **config):
    yaml = {"A": "/data/yolo-master/ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml"}.get(
        arm, "configs/yolo-master-n-esmoe.yaml" if arm == "B" else "configs/yolo-master-n.yaml"
    )
    arch = {"A": "upstream", "A0": "baseline", "B": "esmoe-upstream-w1", "C": "baseline"}[arm]
    return {
        "experiment_id": f"{arm}-{seed}",
        "seed": seed,
        "git_ref": {"framework": framework},
        "hardware": {"gpu": "MetaX C500", "torch": torch},
        "config": {"model_yaml": yaml, "arch": arch, **config},
        "dataset": {"fraction": 1.0},
        "budget": {"epochs": 120, "imgsz": 800, "gpu_hours": 5.0},
        "metrics": {key: value for key in same_config.KEYS},
        "artifact": {"path": f"/runs/{arm}-{seed}/weights/best.pt"},
    }


def four(seed, a, a0, b, c, torch="2.8.0+metax3.3.0.2"):
    fork = "yolo-master@acce839c"
    return [
        record("A", seed, a, fork, torch, recipe="upstream"),
        record("A0", seed, a0, fork, torch),
        record("B", seed, b, torch=torch, recipe="upstream"),
        record("C", seed, c, torch=torch),
    ]


def test_every_run_lands_in_its_arm():
    assert [same_config.arm_of(r) for r in four(0, 0.4, 0.38, 0.41, 0.39)] == ["A", "A0", "B", "C"]


def test_a_seed_missing_an_arm_is_left_out():
    runs = four(0, 0.4, 0.38, 0.41, 0.39) + four(1, 0.4, 0.38, 0.41, 0.39)[:3]
    assert list(same_config.collect(runs)) == [0]


def test_the_differences_are_taken_inside_a_seed():
    value = {"A": 0.40, "A0": 0.38, "B": 0.41, "C": 0.395}
    found = {name: fn(value) for name, fn in same_config.DELTAS.items()}
    assert found["A - B"] == pytest.approx(-0.01)
    assert found["(A - A0) - (B - C)"] == pytest.approx(0.02 - 0.015)


def test_a_seed_split_across_stacks_is_refused():
    runs = four(0, 0.4, 0.38, 0.41, 0.39)
    runs[2]["hardware"]["torch"] = "2.8.0+metax3.7.1.3"
    with pytest.raises(ValueError, match="different stacks"):
        same_config.card(same_config.collect(runs)[0])


def test_a_seed_is_reported_as_measured_only_when_every_arm_was(tmp_path, monkeypatch):
    """`report.load` swaps in `scripts/measure.py`'s metrics and leaves the trainer's beside them."""
    results = tmp_path / "results"
    (results / "measured").mkdir(parents=True)
    metrics = {key: 0.5 for key in same_config.KEYS}
    runs = four(0, 0.4, 0.38, 0.41, 0.39)
    for run in runs:
        run["status"] = "success"
        (results / f"{run['experiment_id']}.json").write_text(json.dumps(run), encoding="utf-8")
    (results / "measured" / "A-0.json").write_text(json.dumps({"run": "A-0", "metrics": metrics}), encoding="utf-8")
    monkeypatch.setattr(report, "ROOT", tmp_path)
    loaded = {r["experiment_id"]: r for r in report.load()}
    assert loaded["A-0"]["metrics"] == metrics
    assert loaded["A-0"]["metrics_by_trainer"][same_config.KEYS[0]] == 0.4
    assert "metrics_by_trainer" not in loaded["B-0"]


def test_precision_separates_both_the_variant_and_the_pairing_key():
    """An FP32 arm pairs with an FP32 baseline; it is not a repeat of the mixed-precision run."""
    block = {"num_experts": 4, "top_k": 2, "aux_weight": 1.0, "balance": "gshard", "blocks": 4}
    mixed, fp32 = record("B", 0, 0.41, **block), record("B", 0, 0.41, **block)
    fp32["budget"] = {**fp32["budget"], "amp": False}
    assert report.variant(mixed) != report.variant(fp32)
    assert report.variant(fp32).endswith("fp32[metaxc500/metax3.3]") or "fp32" in report.variant(fp32)
    assert report.arm(mixed) != report.arm(fp32)
    base_mixed, base_fp32 = record("C", 0, 0.39), record("C", 0, 0.39)
    base_fp32["budget"] = {**base_fp32["budget"], "amp": False}
    assert report.arm(fp32) == report.arm(base_fp32), "an FP32 arm pairs with its own baseline"
    assert report.arm(mixed) == report.arm(base_mixed)
