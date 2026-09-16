"""The demo page reads models/index.json; a wrong entry there ships a page that mislabels the routing."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import demo_export  # noqa: E402

import esmoe  # noqa: E402


def built_block() -> esmoe.ESMoE:
    block = esmoe.ESMoE(num_experts=4, top_k=2)
    block(torch.zeros(1, 16, 8, 8))
    return block


def test_kernels_read_back_from_a_built_block():
    assert demo_export.kernels(built_block()) == [3, 5, 7, 9]


def test_entry_describes_the_model_for_the_page():
    record = demo_export.entry(
        "drone-esmoe", demo_export.MODELS["drone-esmoe"], [built_block()], {0: "pedestrian"}, ROOT / "README.md"
    )
    assert record["id"] == "drone-esmoe"
    assert record["imgsz"] == 800
    assert record["classes"] == ["pedestrian"]
    assert record["blocks"] == [{"experts": 4, "top_k": 2, "kernels": [3, 5, 7, 9], "threshold": 0.0}]
    assert len(record["checkpoint_sha256"]) == 16


def test_the_entry_carries_the_pruning_threshold():
    """The page marks an expert as chosen only when the threshold leaves it running."""
    pruning = built_block()
    pruning.dynamic_threshold = 0.4
    assert (
        demo_export.entry("drone-esmoe", demo_export.MODELS["drone-esmoe"], [pruning], {0: "car"}, ROOT / "README.md")[
            "blocks"
        ][0]["threshold"]
        == 0.4
    )

    dense = esmoe.ESMoE(num_experts=3, top_k=3)
    dense(torch.zeros(1, 16, 8, 8))
    dense.dynamic_threshold = 0.4
    record = demo_export.entry(
        "coco-esmoe-n", demo_export.MODELS["coco-esmoe-n"], [dense], {0: "car"}, ROOT / "README.md"
    )
    assert record["blocks"][0]["threshold"] == 0.0, "top-k over every expert leaves nothing to prune"


def test_every_listed_model_has_a_source_a_group_and_a_size():
    assert demo_export.MODELS.keys() == demo_export.SOURCES.keys()
    for model_id, spec in demo_export.MODELS.items():
        assert spec["group"] in {"coco", "drone"}, model_id
        assert spec["imgsz"] in {640, 800}, model_id
        assert spec["label"].keys() == {"zh", "en"}, model_id
