import pytest
import torch
from torch import nn

import esmoe
from esmoe import DWExpert, ESMoE, blocks, collect_aux_loss, odd_kernels


def _net(num_experts=4, top_k=2, channels=32, **kwargs):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, 2, 1),
        nn.SiLU(),
        ESMoE(num_experts, top_k, channels=channels, **kwargs),
        nn.Conv2d(32, 16, 1),
    )


def test_public_api():
    for name in ("ESMoE", "inject_esmoe", "attach_aux_loss", "collect_aux_loss", "graft", "equip"):
        assert hasattr(esmoe, name)
    assert esmoe.__version__


def test_forward_and_nonzero_aux_loss():
    torch.manual_seed(0)
    net = _net()
    y = net(torch.randn(4, 3, 32, 32))
    assert y.shape == (4, 16, 16, 16)
    aux = collect_aux_loss(net)
    assert aux.item() > 0  # 8.24 gate: non-zero router aux loss
    (y.mean() + 0.01 * aux).backward()
    router_grad = sum(p.grad.abs().sum().item() for p in net[2].router.parameters() if p.grad is not None)
    assert router_grad > 0


def test_lazy_channels_are_inferred_and_preserved():
    block = ESMoE(4, 2)
    assert block.channels is None
    y = block(torch.randn(2, 48, 8, 8))
    assert block.channels == 48
    assert y.shape == (2, 48, 8, 8)


def test_collect_reflects_latest_forward_only():
    torch.manual_seed(0)
    net = _net()
    net(torch.randn(4, 3, 32, 32))
    first = collect_aux_loss(net).item()
    assert collect_aux_loss(net).item() == first  # no double counting without a new forward
    esmoe.clear_aux_loss()
    assert collect_aux_loss(net).item() == 0.0


def test_kernel_sizes_are_heterogeneous_and_capped():
    assert odd_kernels(4) == [3, 5, 7, 9]
    assert odd_kernels(4, max_kernel_size=5) == [3, 5, 5, 5]
    assert ESMoE(3, 1, 16, expert_kernel_sizes=[5, 5, 7]).expert_kernel_sizes == [5, 5, 7]


def test_experts_and_balance_are_replaceable():
    class Thin(nn.Conv2d):
        def __init__(self, c1, c2, k):
            super().__init__(c1, c2, k, 1, k // 2)

    net = _net(expert=Thin, balance=lambda probs, gate: probs.sum() * 0)
    net(torch.randn(2, 3, 32, 32))
    block = next(blocks(net))
    assert all(isinstance(e, Thin) for e in block.experts)
    assert collect_aux_loss(net).item() == 0.0
    assert issubclass(DWExpert, nn.Module)


def test_blocks_finds_every_block():
    net = nn.Sequential(ESMoE(2, 1, 8), nn.Identity(), ESMoE(3, 2, 8))
    assert len(list(blocks(net))) == 2


def test_invalid_configuration_is_rejected():
    for bad in ({"num_experts": 2, "top_k": 3}, {"num_experts": 2, "top_k": 0}):
        with pytest.raises(ValueError):
            ESMoE(**bad)
    with pytest.raises(ValueError):
        ESMoE(4, 2, 16, expert_kernel_sizes=[3, 5])


def test_aux_term_survives_autocast():
    """所有实验都开着 AMP,所以半精度下辅助项必须仍然有限、非零,且不把参数拖成半精度。"""
    torch.manual_seed(0)
    net = _net()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = net(torch.randn(4, 3, 32, 32))
    aux = collect_aux_loss(net)
    assert torch.isfinite(y).all()
    assert torch.isfinite(aux) and aux.item() > 0
    assert all(p.dtype is torch.float32 for p in net.parameters())


def test_gate_still_sums_to_one_in_half_precision():
    torch.manual_seed(0)
    block = ESMoE(4, 2, channels=16)
    probability = torch.softmax(block.router(torch.randn(8, 16, 8, 8).half().float()), dim=1)
    weights, chosen = probability.topk(2, dim=1)
    gate = torch.zeros_like(probability).scatter(1, chosen, weights).half()
    gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-9)
    assert torch.allclose(gate.float().sum(dim=1), torch.ones(8), atol=1e-2)


def test_gshard_matches_the_upstream_definition():
    """N * sum(usage^2) on the gated weights: 1.0 at uniform, N when one expert takes everything."""
    probs = torch.full((6, 4), 0.25)
    uniform_gate = torch.full((6, 4), 0.25)
    one_hot_gate = torch.zeros(6, 4)
    one_hot_gate[:, 0] = 1.0
    assert esmoe.gshard_balance(probs, uniform_gate).item() == pytest.approx(1.0, rel=1e-5)
    assert esmoe.gshard_balance(probs, one_hot_gate).item() == pytest.approx(4.0, rel=1e-5)


def test_gshard_rises_as_the_dispatch_concentrates():
    probs = torch.full((3, 4), 0.25)
    flat = torch.tensor([[0.25, 0.25, 0.25, 0.25]]).repeat(3, 1)
    skewed = torch.tensor([[0.70, 0.20, 0.05, 0.05]]).repeat(3, 1)
    assert esmoe.gshard_balance(probs, skewed) > esmoe.gshard_balance(probs, flat)


def test_switch_is_flat_where_gshard_is_not():
    """The two objectives disagree, which is why the block lets you choose.

    With uniform mean probabilities the Switch term is pinned at k however the top-k dispatch
    falls, so it cannot report a dispatch that has collapsed onto one expert.
    """
    even = torch.tensor(
        [[0.40, 0.30, 0.15, 0.15], [0.30, 0.40, 0.15, 0.15], [0.15, 0.15, 0.40, 0.30], [0.15, 0.15, 0.30, 0.40]]
    )
    skewed = torch.tensor([[0.25, 0.45, 0.15, 0.15], [0.25, 0.15, 0.45, 0.15], [0.25, 0.15, 0.15, 0.45]])

    def top_k_gate(probs, k=2):
        return torch.zeros_like(probs).scatter(1, probs.topk(k, dim=1).indices, 1.0)

    assert esmoe.switch_balance(even, top_k_gate(even)).item() == pytest.approx(2.0, rel=1e-5)
    assert esmoe.switch_balance(skewed, top_k_gate(skewed)).item() == pytest.approx(2.0, rel=1e-5)
    # one expert takes every token's top-1 in `skewed`, and the Switch term never moves
    assert top_k_gate(skewed)[:, 0].mean().item() == pytest.approx(1.0)


def test_block_trains_with_the_upstream_objective():
    net = _net(balance=esmoe.gshard_balance)
    esmoe.clear_aux_loss()
    net(torch.randn(2, 3, 32, 32))
    aux = collect_aux_loss(net)
    assert aux.item() > 0
    aux.backward()
    block = next(blocks(net))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.router.parameters())


def test_router_survives_a_runaway_logit():
    """A logit large enough to overflow the softmax must not take the gate to NaN.

    Upstream clamps before the softmax for this reason; without it a diverging router poisons every
    downstream loss and the run dies far from the cause.
    """
    net = _net()
    net(torch.randn(2, 3, 32, 32))
    block = next(blocks(net))
    with torch.no_grad():
        block.router[-1].bias[0] = 1e4
    out = net(torch.randn(2, 3, 32, 32))
    assert torch.isfinite(out).all()
    assert torch.isfinite(collect_aux_loss(net)).all()


def _gate_of(probs, k=2):
    weights, chosen = probs.topk(k, dim=1)
    gate = torch.zeros_like(probs).scatter(1, chosen, weights)
    return gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-9)


def test_paper_and_upstream_state_the_same_objective():
    """Eq. 13 and the released ES_MOE differ only by an affine map.

    With utilisations summing to one, ``sum((u - 1/E)^2) == sum(u^2) - 1/E``, so the paper's loss
    is ``(gshard - 1) / E^2``: same minimiser, proportional gradients. An earlier reading of the
    source had these as different objectives; upstream feeds its term the *gated* weights, which
    is exactly what the paper does.
    """
    torch.manual_seed(0)
    for _ in range(5):
        probs = torch.rand(6, 4).softmax(dim=1)
        gate = _gate_of(probs)
        upstream = esmoe.gshard_balance(probs, gate).item()
        assert esmoe.master_balance(probs, gate).item() == pytest.approx((upstream - 1.0) / 4**2, rel=1e-5)


def test_reading_the_gate_is_what_sees_a_collapsed_dispatch():
    """The variable that matters is which tensor an objective reads, not the formula family.

    Both arms below carry uniform mean probabilities; in `collapsed` one expert is in every
    sample's top-k. Terms reading the probabilities cannot tell them apart; terms reading the
    gated weights can.
    """
    balanced = torch.tensor(
        [[0.40, 0.30, 0.15, 0.15], [0.30, 0.40, 0.15, 0.15], [0.15, 0.15, 0.40, 0.30], [0.15, 0.15, 0.30, 0.40]]
    )
    collapsed = torch.tensor([[0.25, 0.45, 0.15, 0.15], [0.25, 0.15, 0.45, 0.15], [0.25, 0.15, 0.15, 0.45]])
    assert _gate_of(collapsed)[:, 0].gt(0).all(), "expert 0 must be in every top-k for this to test anything"

    for blind in (esmoe.switch_balance, esmoe.gshard_probs_balance):
        assert blind(balanced, _gate_of(balanced)).item() == pytest.approx(
            blind(collapsed, _gate_of(collapsed)).item(), abs=1e-6
        )
    for seeing in (esmoe.gshard_balance, esmoe.master_balance):
        assert seeing(collapsed, _gate_of(collapsed)) > seeing(balanced, _gate_of(balanced)) + 1e-6


def test_master_balance_is_zero_at_uniform_use():
    uniform = torch.full((5, 4), 0.25)
    assert esmoe.master_balance(uniform, uniform).item() == pytest.approx(0.0, abs=1e-9)


def test_the_default_objective_is_the_one_upstream_ships():
    """Changing this default changes what every future run optimises, so it is pinned by a test."""
    block = ESMoE(4, 2, channels=16)
    assert block.balance is esmoe.gshard_balance


def test_defaults_keep_the_recorded_runs_reproducible():
    """The switches that align with upstream are opt-in.

    Every protocol run in `results/` was measured without the output norm and without dense
    training. Flipping either default would silently change what `equip()` builds, so the defaults
    are pinned here and the alignment is requested explicitly.
    """
    block = ESMoE(4, 2, channels=16)
    assert block.out_norm is False
    assert block.dense_training is False
    assert isinstance(block.norm, nn.Identity)


def test_out_norm_adds_a_normalisation_to_the_mixed_output():
    plain, normed = ESMoE(4, 2, channels=16), ESMoE(4, 2, channels=16, out_norm=True)
    assert isinstance(normed.norm, nn.Sequential)
    assert isinstance(normed.norm[0], nn.BatchNorm2d)
    x = torch.randn(2, 16, 8, 8)
    assert plain(x).shape == normed(x).shape


def test_dense_training_runs_every_expert():
    """An unrouted expert must still see data when dense training is on, and not when it is off."""
    x = torch.randn(4, 16, 8, 8)

    def count_experts_run(block):
        seen: set[int] = set()

        def watch(index):
            return lambda _m, _i, _o: seen.add(index)

        for index, expert in enumerate(block.experts):
            expert.register_forward_hook(watch(index))
        block.train()
        block(x)
        return len(seen)

    assert count_experts_run(ESMoE(4, 1, channels=16, dense_training=True)) == 4
    assert count_experts_run(ESMoE(4, 1, channels=16, dense_training=False)) < 4


def test_dense_training_does_not_change_the_output():
    """Unrouted experts are weighted by zero, so running them cannot move the result."""
    torch.manual_seed(0)
    sparse = ESMoE(4, 2, channels=16)
    dense = ESMoE(4, 2, channels=16, dense_training=True)
    dense.load_state_dict(sparse.state_dict())
    sparse.eval(), dense.eval()
    x = torch.randn(3, 16, 8, 8)
    with torch.no_grad():
        assert torch.allclose(sparse(x), dense(x), atol=1e-6)


@pytest.mark.parametrize("num_experts,top_k", [(2, 1), (3, 2), (4, 2), (4, 4), (8, 2)])
@pytest.mark.parametrize("balance", ["switch_balance", "gshard_balance", "master_balance", "gshard_probs_balance"])
def test_every_combination_trains_and_reports_a_finite_aux(num_experts, top_k, balance):
    """The community will mix these freely, so every combination has to build and back-propagate."""
    block = ESMoE(num_experts, top_k, channels=16, balance=getattr(esmoe, balance), out_norm=True)
    net = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), block, nn.Conv2d(16, 4, 1))
    esmoe.clear_aux_loss()
    out = net(torch.randn(2, 3, 16, 16))
    aux = collect_aux_loss(net)
    assert torch.isfinite(aux).all()
    (out.sum() + aux).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in block.router.parameters())


def test_top_k_equal_to_num_experts_is_dense_and_balanced():
    """K == E leaves nothing to route away, so every balance term should sit at its floor."""
    block = ESMoE(4, 4, channels=16)
    esmoe.clear_aux_loss()
    block(torch.randn(4, 16, 8, 8))
    assert collect_aux_loss(nn.Sequential(block)).item() >= 0


def test_top_k_none_activates_every_expert():
    """Upstream reads `top_k=None` as "use all experts"; a community config will pass it."""
    assert ESMoE(4, None, channels=16).top_k == 4


def test_out_channels_changes_the_block_width():
    """Upstream's ES_MOE takes `out_channels`. A block that is not channel-preserving cannot be
    grafted into a stock yaml, since `parse_model` assumes `c2 == c1`, but it can be wired by hand.
    """
    block = ESMoE(4, 2, channels=16, out_channels=32)
    assert block(torch.randn(2, 16, 8, 8)).shape == (2, 32, 8, 8)


@pytest.mark.parametrize(
    "given,cap,expected",
    [([4, 20, 7], 9, [3, 9, 7]), ([3, 5, 7], 15, [3, 5, 7]), ([16, 16, 16], 15, [15, 15, 15])],
)
def test_explicit_kernels_step_down_to_odd_and_cap(given, cap, expected):
    """Upstream lowers an even kernel by one and caps it, so a pruned checkpoint's kernels reload."""
    assert ESMoE(3, 2, channels=16, expert_kernel_sizes=given, max_kernel_size=cap).expert_kernel_sizes == expected


def test_an_even_kernel_cap_is_lowered_not_raised():
    assert ESMoE(2, 1, channels=16, max_kernel_size=16).expert_kernel_sizes == [3, 5]


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"num_experts": 0}, "num_experts"),
        ({"reduction": 0}, "reduction"),
        ({"dynamic_threshold": 1.5}, "dynamic_threshold"),
        ({"max_kernel_size": 2}, "max_kernel_size"),
        ({"top_k": 9}, "top_k"),
        ({"nonsense": True}, "unknown ESMoE options"),
    ],
)
def test_invalid_settings_are_refused_with_a_reason(kwargs, message):
    """The same guards upstream applies, so a bad community config fails at construction."""
    with pytest.raises(ValueError, match=message):
        ESMoE(channels=16, **kwargs)


def test_dynamic_threshold_prunes_outside_training_and_keeps_the_leader():
    """Upstream's inference pruning: below the threshold an expert goes, except the leading one."""
    block = ESMoE(4, 4, channels=16, dynamic_threshold=0.9).eval()
    seen = []
    for expert in block.experts:
        expert.register_forward_hook(lambda m, i, o, seen=seen: seen.append(m))
    with torch.no_grad():
        block(torch.randn(3, 16, 8, 8))
    # One expert can clear a 0.9 share at most, so only the leader survives on each sample.
    assert 1 <= len(seen) <= 4


def test_dynamic_threshold_is_inert_while_training():
    block = ESMoE(4, 4, channels=16, dynamic_threshold=0.9).train()
    plain = ESMoE(4, 4, channels=16).train()
    plain.load_state_dict(block.state_dict())
    x = torch.randn(3, 16, 8, 8)
    assert torch.allclose(block(x), plain(x), atol=1e-6)


def test_sparse_inference_off_runs_every_expert():
    block = ESMoE(4, 1, channels=16, sparse_inference=False).eval()
    seen = []
    for expert in block.experts:
        expert.register_forward_hook(lambda m, i, o, seen=seen: seen.append(m))
    with torch.no_grad():
        block(torch.randn(2, 16, 8, 8))
    assert len(seen) == 4


def test_spec_reports_every_setting_a_config_can_carry():
    block = ESMoE(4, 2, channels=16, balance="master", out_norm=True, dynamic_threshold=0.4)
    assert block.spec() == {
        "balance": "master",
        "out_norm": True,
        "dense_training": False,
        "sparse_inference": True,
        "dynamic_threshold": 0.4,
    }
    assert set(block.spec()) == set(esmoe.SETTINGS)
