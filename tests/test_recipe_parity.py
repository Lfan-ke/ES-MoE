"""The step-for-step check carries upstream's block weights onto this package's block by name."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import recipe_parity  # noqa: E402


def test_upstream_block_weights_land_on_the_matching_tensors_here():
    state = {
        "model.3.routing.routing_network.0.weight": torch.ones(8, 64, 1, 1),
        "model.3.routing.routing_network.2.bias": torch.ones(4),
        "model.6.experts.1.conv.depthwise.weight": torch.ones(128, 1, 5, 5),
        "model.9.experts.3.conv.pointwise.weight": torch.ones(128, 128, 1, 1),
        "model.9.experts.3.conv.bn.running_var": torch.ones(128),
        "model.12.norm.0.weight": torch.ones(256),
        "model.0.conv.weight": torch.ones(16, 3, 3, 3),
    }
    moved = recipe_parity.renamed(state)
    assert set(moved) == {
        "model.3.router.2.weight",
        "model.3.router.4.bias",
        "model.6.experts.1.dw.weight",
        "model.9.experts.3.pw.weight",
        "model.9.experts.3.bn.running_var",
        "model.12.norm.0.weight",
        "model.0.conv.weight",
    }
    # A 1x1 convolution's weight drops its two unit dimensions to become the Linear router's.
    assert moved["model.3.router.2.weight"].shape == (8, 64)
    assert moved["model.9.experts.3.pw.weight"].shape == (128, 128, 1, 1)
    assert moved["model.0.conv.weight"].shape == (16, 3, 3, 3)
