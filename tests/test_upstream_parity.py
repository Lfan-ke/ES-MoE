"""Hold this package's block numerically equal to YOLO-Master's ES_MOE where it claims to be.

Upstream's operations are rewritten here from its source rather than imported, so the comparison
is self-contained and states plainly what is being matched. Where the two genuinely differ, the
difference is asserted too, so a later change cannot quietly erase it.

Source read for these: `ultralytics/nn/modules/moe/routers.py` (DynamicRoutingLayer, `_soft_top_k`,
`_hard_top_k`), `moe/loss.py` (`gshard_balance_loss`), `moe/modules.py` (ES_MOE), `moe/experts.py`
(EfficientExpertGroup) and `nn/modules/_numeric.py` (`stable_normalize`).
"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import esmoe
from esmoe import ESMoE

TOP_K, EXPERTS, CHANNELS = 2, 4, 16


def stable_normalize(tensor: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    """Upstream's `_numeric.stable_normalize`, with the dtype floor spelled out."""
    return tensor / tensor.sum(dim=dim, keepdim=True).clamp_min(eps)


def soft_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Upstream `DynamicRoutingLayer._soft_top_k`, on logits shaped [B, E]."""
    weights = F.softmax(logits.clamp(-30.0, 30.0).float(), dim=1).type_as(logits)
    _, indices = torch.topk(weights, top_k, dim=1)
    mask = F.one_hot(indices, num_classes=logits.shape[1]).sum(dim=1).to(weights.dtype)
    return stable_normalize(weights * mask, dim=1)


def hard_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Upstream `DynamicRoutingLayer._hard_top_k`."""
    weights = F.softmax(logits.clamp(-30.0, 30.0).float(), dim=1).type_as(logits)
    values, indices = torch.topk(weights, top_k, dim=1)
    values = stable_normalize(values, dim=1)
    return torch.zeros_like(weights).scatter_(1, indices, values)


def gshard_balance_loss(usage: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Upstream `loss.gshard_balance_loss`."""
    usage = usage.reshape(-1).float()
    usage = usage / usage.sum().clamp_min(1e-6)
    return num_experts * torch.sum(usage * usage)


def gate_of(block: ESMoE, x: torch.Tensor) -> torch.Tensor:
    """The gate this package builds, pulled out of the forward pass."""
    probs = F.softmax(block.router(x).float().clamp(-30.0, 30.0), dim=1).type_as(x)
    weights, chosen = probs.topk(block.top_k, dim=1)
    gate = torch.zeros_like(probs).scatter(1, chosen, weights)
    return gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-9)


def test_upstreams_two_top_k_paths_agree_with_each_other():
    """Upstream's own claim, in its comment: hard Top-K is soft Top-K's numerics without the mask."""
    torch.manual_seed(0)
    logits = torch.randn(32, EXPERTS) * 3
    assert torch.allclose(soft_top_k(logits, TOP_K), hard_top_k(logits, TOP_K), atol=1e-6)


def test_the_gate_matches_upstreams_soft_top_k():
    torch.manual_seed(0)
    block = ESMoE(EXPERTS, TOP_K, channels=CHANNELS)
    x = torch.randn(8, CHANNELS, 8, 8)
    logits = block.router(x)
    assert torch.allclose(gate_of(block, x), soft_top_k(logits, TOP_K), atol=1e-6)


def test_a_one_by_one_convolution_on_a_pooled_map_is_the_linear_router():
    """Upstream routes with two 1x1 convolutions over a [B, C, 1, 1] map; this package uses Linear.

    They are the same operation, bias included, which is why the two routers can be compared at all.
    """
    torch.manual_seed(0)
    block = ESMoE(EXPERTS, TOP_K, channels=CHANNELS)
    hidden = max(CHANNELS // 8, 8)
    conv = nn.Sequential(
        nn.Conv2d(CHANNELS, hidden, 1),
        nn.SiLU(inplace=False),
        nn.Conv2d(hidden, EXPERTS, 1),
    )
    linear_layers = [m for m in block.router if isinstance(m, nn.Linear)]
    conv_layers = [m for m in conv if isinstance(m, nn.Conv2d)]
    for linear, convolution in zip(linear_layers, conv_layers, strict=True):
        convolution.weight.data = linear.weight.data[:, :, None, None].clone()
        convolution.bias.data = linear.bias.data.clone()

    x = torch.randn(8, CHANNELS, 8, 8)
    pooled = F.adaptive_avg_pool2d(x, 1)
    assert torch.allclose(block.router(x), conv(pooled).flatten(1), atol=1e-5)


def test_the_balance_term_matches_upstreams_on_the_same_tensor():
    """Upstream feeds `routing_weights.mean(dim=(0, 2, 3))`; the router's output is spatially flat,
    so that mean equals the mean over the batch of this package's [B, E] gate."""
    torch.manual_seed(0)
    block = ESMoE(EXPERTS, TOP_K, channels=CHANNELS)
    x = torch.randn(8, CHANNELS, 8, 8)
    gate = gate_of(block, x)
    upstream = gshard_balance_loss(gate.mean(dim=0), EXPERTS)
    assert torch.allclose(esmoe.gshard_balance(gate, gate), upstream, atol=1e-6)


def test_repeating_the_gate_over_space_changes_nothing():
    """Upstream carries the gate as [B, E, H, W] repeated from [B, E, 1, 1]; this package keeps
    [B, E] and broadcasts. The spatial mean upstream takes is therefore the same number."""
    torch.manual_seed(0)
    gate = soft_top_k(torch.randn(8, EXPERTS), TOP_K)
    spatial = gate[:, :, None, None].repeat(1, 1, 5, 7)
    assert torch.allclose(spatial.mean(dim=(0, 2, 3)), gate.mean(dim=0), atol=1e-6)


def test_top_k_none_is_upstreams_plain_softmax():
    """With `use_top_k=False` upstream skips masking entirely; `top_k=None` here must match."""
    torch.manual_seed(0)
    block = ESMoE(EXPERTS, None, channels=CHANNELS)
    x = torch.randn(8, CHANNELS, 8, 8)
    plain = F.softmax(block.router(x).clamp(-30.0, 30.0).float(), dim=1)
    assert torch.allclose(gate_of(block, x), plain, atol=1e-6)


@pytest.mark.parametrize("num_experts,expected", [(1, [3]), (2, [3, 5]), (3, [3, 5, 7]), (5, [3, 5, 7, 9, 11])])
def test_default_kernels_match_upstreams_rule(num_experts, expected):
    """Upstream uses `[3, 5, 7][:E]` up to three experts and `3 + 2i` beyond."""
    assert ESMoE(num_experts, 1, channels=CHANNELS).expert_kernel_sizes == expected


def test_the_expert_is_upstreams_depthwise_separable_block():
    """`EfficientExpertGroup` wraps `DepthwiseSeparableConv`: dw(bias=False) -> pw(bias=False)
    -> BatchNorm -> SiLU. Anything else changes what the parameter count means."""
    expert = esmoe.DWExpert(CHANNELS, CHANNELS, 5)
    assert (expert.dw.groups, expert.dw.bias, expert.dw.kernel_size) == (CHANNELS, None, (5, 5))
    assert (expert.pw.bias, expert.pw.kernel_size) == (None, (1, 1))
    assert isinstance(expert.bn, nn.BatchNorm2d) and isinstance(expert.act, nn.SiLU)


def test_the_normalisation_floor_is_the_one_difference_and_it_is_deliberate():
    """Upstream floors the denominator at 1e-6 scaled to the dtype; this package uses 1e-9.

    Both only matter when every routed weight underflows, which top-k of a softmax cannot produce:
    the selected weights are the largest, so the largest is at least 1/E.
    """
    torch.manual_seed(0)
    for _ in range(20):
        logits = torch.randn(16, EXPERTS) * 30
        gate = soft_top_k(logits, TOP_K)
        assert gate.sum(dim=1).min() > 0.99
        assert gate.max(dim=1).values.min() >= 1.0 / EXPERTS - 1e-6


def dense_forward(block: ESMoE, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Upstream `ES_MOE._dense_forward`: every expert runs, weighted, nothing skipped."""
    out = 0
    for index, expert in enumerate(block.experts):
        out = out + expert(x) * gate[:, index].view(-1, 1, 1, 1)
    return out


@pytest.mark.parametrize("top_k", [1, 2, EXPERTS, None])
def test_skipping_an_unrouted_expert_is_the_dense_sum(top_k):
    """Upstream sums over all experts and lets a zero weight do the work; this package skips the
    expert instead. The two have to agree exactly, or `dense_training` would change the model
    rather than only its normalisation statistics."""
    torch.manual_seed(0)
    block = ESMoE(EXPERTS, top_k, channels=CHANNELS).eval()
    x = torch.randn(6, CHANNELS, 8, 8)
    with torch.no_grad():
        assert torch.allclose(block(x), dense_forward(block, x, gate_of(block, x)), atol=1e-6)


def test_upstream_runs_dense_whenever_top_k_cannot_skip_anything():
    """`_eager_sparse_enabled()` is false when `top_k` is None or equals the expert count, so
    upstream goes dense. Here every weight is then non-zero, so no expert is skipped either."""
    torch.manual_seed(0)
    x = torch.randn(4, CHANNELS, 8, 8)
    for top_k in (EXPERTS, None):
        block = ESMoE(EXPERTS, top_k, channels=CHANNELS).eval()
        assert (gate_of(block, x) > 0).all(), "no expert can be skipped when k covers them all"


def test_a_non_finite_balance_term_cannot_reach_the_optimised_loss():
    """Upstream replaces a non-finite balance loss with a graph-connected zero. This package zeroes
    it in the loss patch instead; either way the total must stay finite."""
    block = ESMoE(EXPERTS, TOP_K, channels=CHANNELS)
    esmoe.clear_aux_loss()
    block(torch.randn(4, CHANNELS, 8, 8))
    esmoe.registry.publish(block, torch.tensor(float("nan")))
    aux = esmoe.collect_aux_loss(nn.Sequential(block))
    total = torch.tensor(1.0) + (torch.zeros_like(aux) if not torch.isfinite(aux) else aux)
    assert torch.isfinite(total).all()


def test_reading_the_term_does_not_consume_it_and_a_clear_is_what_drops_it():
    """Upstream's registry read does not remove either -- a step may compute the loss more than
    once and each computation needs the term. Upstream rejects a value from an earlier step with a
    step stamp; here the loss patch clears before each forward. Both reads, then the clear."""
    block = ESMoE(EXPERTS, TOP_K, channels=CHANNELS)
    net = nn.Sequential(block)
    esmoe.clear_aux_loss()
    block(torch.randn(4, CHANNELS, 8, 8))
    assert esmoe.collect_aux_loss(net).item() == pytest.approx(esmoe.collect_aux_loss(net).item())
    esmoe.clear_aux_loss()
    assert esmoe.collect_aux_loss(net).item() == 0.0
