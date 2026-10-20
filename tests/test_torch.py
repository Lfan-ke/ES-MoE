"""The torch backend on its own: no ultralytics, no integration, nothing but torch."""

import pytest
import torch
from torch import nn

import esmoe
from esmoe import ESMoE


def test_package_needs_no_integration():
    assert "ultralytics" not in str(esmoe.ESMoE.__module__)
    assert esmoe.backend("torch").ESMoE is ESMoE


def test_kernels_decide_the_experts():
    block = ESMoE(kernels=[3, 5, 7], top_k=1)
    assert block.num_experts == 3
    assert block.kernels == [3, 5, 7]
    assert block.top_k == 1


def test_kernels_may_be_rectangular():
    block = ESMoE(kernels=[(3, 3), (1, 9), (9, 1)], channels=8)
    assert block.kernels == [(3, 3), (1, 9), (9, 1)]
    assert block(torch.randn(2, 8, 16, 16)).shape == (2, 8, 16, 16)


def test_even_kernels_step_down_and_cap():
    assert ESMoE(kernels=[4, 6, 20], max_kernel=9).kernels == [3, 5, 9]
    assert ESMoE(kernels=[(4, 20)], max_kernel=9).kernels == [(3, 9)]


def test_kernels_repeat_freely():
    assert ESMoE(kernels=[3, 3, 5, 5, 1]).kernels == [3, 3, 5, 5, 1]


def test_experts_may_differ_per_branch():
    class Thin(nn.Module):
        def __init__(self, c1, c2, k):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, 1)

        def forward(self, x):
            return self.conv(x)

    block = ESMoE(kernels=[3, 5], expert=[esmoe.DWExpert, Thin], channels=8)
    assert isinstance(block.experts[1], Thin)
    assert block(torch.randn(1, 8, 8, 8)).shape == (1, 8, 8, 8)


def test_channels_are_inferred_on_the_first_pass():
    block = ESMoE(kernels=[3, 5])
    assert block.channels is None
    block(torch.randn(1, 12, 8, 8))
    assert block.channels == 12


def test_it_drops_into_a_sequential():
    net = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), ESMoE(kernels=[3, 5, 7], top_k=1))
    assert net(torch.randn(2, 3, 32, 32)).shape == (2, 16, 16, 16)


def test_routing_reaches_backward():
    net = nn.Sequential(nn.Conv2d(3, 8, 3, 1, 1), ESMoE(kernels=[3, 5]))
    with esmoe.routing(net) as route:
        out = net(torch.randn(2, 3, 8, 8))
        aux = route.loss()
        (out.sum() + aux).backward()
    assert aux.item() > 0
    assert all(p.grad is not None for p in net[1].router.parameters())


def test_routing_forgets_the_step_before():
    net = nn.Sequential(ESMoE(kernels=[3, 5], channels=4))
    net(torch.randn(1, 4, 8, 8))
    with esmoe.routing(net) as route:
        assert route.loss().item() == 0.0


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ({"e": 3, "k": 1}, {"num_experts": 3, "top_k": 1}),
        ({"experts": 3}, {"num_experts": 3}),
        ({"experts": [3, 5]}, {"num_experts": 2, "kernels": [3, 5]}),
        ({"expert_kernel_sizes": [3, 5]}, {"kernels": [3, 5]}),
        ({"max_kernel_size": 5, "kernels": [9, 9]}, {"kernels": [5, 5]}),
    ],
)
def test_other_spellings_are_understood(given, expected):
    block = ESMoE(**given)
    for key, value in expected.items():
        assert getattr(block, key) == value


def test_a_mapping_stands_in_for_the_arguments():
    block = ESMoE({"kernels": [3, 5, 7], "top_k": 1})
    assert (block.num_experts, block.top_k) == (3, 1)


def test_an_unknown_argument_warns_and_is_ignored():
    with pytest.warns(UserWarning, match="nonsense"):
        block = ESMoE(kernels=[3, 5], nonsense=True)
    assert block.num_experts == 2


@pytest.mark.parametrize(
    ("given", "message"),
    [
        ({"num_experts": 0}, "positive"),
        ({"kernels": [3, 5], "top_k": 0}, "top_k"),
        ({"kernels": [3, 5], "expert": [esmoe.DWExpert]}, "disagree"),
        ({"dynamic_threshold": 2.0}, "dynamic_threshold"),
        ({"reduction": 0}, "reduction"),
    ],
)
def test_a_bad_value_still_raises(given, message):
    with pytest.raises(ValueError, match=message):
        ESMoE(**given)


def test_spec_records_what_a_config_has_to_carry():
    assert ESMoE(kernels=[3, 5, 7]).spec() == {
        "balance": "switch",
        "out_norm": False,
        "dense_training": False,
        "sparse_inference": True,
        "dynamic_threshold": 0.0,
    }
    assert ESMoE(kernels=[1, 1]).spec()["kernels"] == [1, 1]


def test_graft_puts_blocks_into_a_model_that_exists():
    model = nn.Sequential(nn.Conv2d(3, 8, 3, 1, 1), nn.ReLU())
    esmoe.graft(model, at="0", kernels=[3, 5])
    assert len(list(esmoe.blocks(model))) == 1
    assert model(torch.randn(1, 3, 8, 8)).shape == (1, 8, 8, 8)


@pytest.mark.parametrize("mode", ["after", "before", "replace"])
def test_graft_modes(mode):
    model = nn.Sequential(nn.Conv2d(3, 3, 3, 1, 1))
    esmoe.graft(model, at=lambda name, module: isinstance(module, nn.Conv2d), mode=mode, kernels=[3])
    assert model(torch.randn(1, 3, 8, 8)).shape == (1, 3, 8, 8)


def test_too_large_a_top_k_falls_back_to_every_expert():
    with pytest.warns(UserWarning, match="activates all"):
        assert ESMoE(kernels=[3, 5], top_k=4).top_k == 2


def test_graft_says_so_when_nothing_matches():
    with pytest.raises(ValueError, match="no submodule"):
        esmoe.graft(nn.Sequential(nn.ReLU()), at="missing")


def test_old_checkpoints_still_find_the_block():
    import esmoe.module as legacy

    assert legacy.ESMoE is ESMoE
    assert legacy.DWExpert is esmoe.DWExpert
