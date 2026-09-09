--8<-- "limitations.md"

## Divergence from upstream and from the paper

This package reimplements the structure of YOLO-Master's `ES_MOE`. Three differences are worth stating after a line-by-line comparison.

**The balancing term: the paper and upstream agree, and this package read the wrong tensor.** Upstream applies `N * sum(usage^2)` to whatever `DynamicRoutingLayer` returns, and in training that is the output of `_soft_top_k` -- **the gate, after the top-k mask and renormalisation**. The paper's equation 12 averages the same thing. The two are therefore one objective up to an affine map: `sum((mu - 1/E)^2) == sum(mu^2) - 1/E`, so `L_paper = (L_upstream - 1) / E^2`, with the same minimiser and proportional gradients.

This package first wrote `gshard_balance` against the **raw softmax**, which is neither. It now reads the gate, and a test pins the identity above. The probability-reading form survives as `gshard_probs_balance` to isolate one variable -- whether the objective reads the probabilities or the dispatch. That, not the choice between the Switch and GShard formulas, is what decides whether a routing collapse is visible to the term.

The coefficient still differs: upstream carries `balance_loss_coeff = 1.0` against this package's `weight = 0.01`. But upstream also divides by an EMA of the term itself and caps the sum at `aux_budget = 3.0`, so its steady-state contribution is about `1.0 * gain` and the coefficient mostly moves the transient. An earlier note here claimed a 31x difference in balancing gradient without accounting for that; it is corrected.

**The paper describes a diversity objective that the shipped block does not apply.** The paper (arXiv 2512.23273) says training carries an objective encouraging complementary expertise. Upstream does define `diversity_loss_coeff`, but it belongs to a different family of blocks, defaults to `0.0`, and `ES_MOE` never references it. Nothing in the default configuration pushes the experts apart - which is consistent with the absence of scale specialisation measured here across seven generations, and is not particular to this package.

**Inference sparsifies to a different degree.** Upstream carries a `dynamic_threshold` (0.4 by default) that prunes low-confidence experts at inference; this package only skips the experts outside top-k. Training semantics match; exported graphs and measured latency will not.

**The graft location matches the paper's final default; the count does not.** Section 3.1 of the paper places the block in both backbone and neck, but the ablation in section 4.3.1 (Table 5) reports Neck Only 58.2, Both 54.9 and **Backbone Only 62.1**, on which the authors state "we adopt backbone-only ES-MoE as our default configuration". This package grafts into the backbone only, which agrees with that default; an earlier note here said the paper used both placements and was written without that table.

The count does differ: upstream's yaml places one block after every backbone stage, four in all, where this package defaults to one at the backbone end. `graft(at="backbone_stages")` reproduces the upstream layout and yields four blocks on all seven generations. The stage boundaries are derived from the downsampling layers rather than fixed indices, because the generations disagree on how they downsample -- v9 uses `AConv`, v10 `SCDown`, the rest a stride-2 `Conv`.

Checked and identical: the default heterogeneous kernels for four experts are `[3, 5, 7, 9]` in both; both routers are a two-layer bottleneck after global pooling (`reduction = 8`, floored at 8 channels); both clamp logits to `[-30, 30]` before the softmax; and the paper's Soft Top-K (equation 8) matches this package's gating term for term.
