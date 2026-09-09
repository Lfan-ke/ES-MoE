--8<-- "limitations.md"

## Divergence from upstream and from the paper

This package reimplements the structure of YOLO-Master's `ES_MOE`. Three differences are worth stating after a line-by-line comparison.

**The balancing term is not the same one.** Upstream's `ES_MOE` optimises the GShard form `N * sum(usage^2)` at `balance_loss_coeff = 1.0`, which reaches the total loss through `moe_gain = 1.0`; this package defaults to the Switch form `E * sum(p_i f_i)` at `weight = 0.01`. On a real routing record the former puts 31x more balancing gradient on the mean probabilities. Both are selectable (`balance=esmoe.gshard_balance`), and the comparison's criteria are on the [judgment lines](JUDGMENT.md) page.

**The paper describes a diversity objective that the shipped block does not apply.** The paper (arXiv 2512.23273) says training carries an objective encouraging complementary expertise. Upstream does define `diversity_loss_coeff`, but it belongs to a different family of blocks, defaults to `0.0`, and `ES_MOE` never references it. Nothing in the default configuration pushes the experts apart - which is consistent with the absence of scale specialisation measured here across seven generations, and is not particular to this package.

**Inference sparsifies to a different degree.** Upstream carries a `dynamic_threshold` (0.4 by default) that prunes low-confidence experts at inference; this package only skips the experts outside top-k. Training semantics match; exported graphs and measured latency will not.

Checked and identical: the default heterogeneous kernels for four experts are `[3, 5, 7, 9]` in both; both routers are a two-layer bottleneck after global pooling (`reduction = 8`, floored at 8 channels); both clamp logits to `[-30, 30]` before the softmax.
