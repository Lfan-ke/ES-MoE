# Judgment lines (pre-registered)

A negative result only counts when the judgment line predates it. This page fixes the criteria and predictions **before** the runs they govern finish; the git commit time is the evidence.

## Known and unknown at declaration

Known when declared (evening of 2026-09-02, UTC+8): all three YOLOv8n arms (baseline / esmoe / esmoe-rewire), three seeds each, with their area buckets and router analysis; the overall metrics of YOLO11n baseline seed 0. **Unknown**: every YOLO11n esmoe / rewire run and the remaining baseline seeds; every YOLO12n and YOLO26n run. The lines and predictions below bind those unknown results.

## Lines

For each backbone × arm (same budget, data, augmentation and evaluation, three same-seed pairs):

- **Effective**: paired mAP50 mean > 0, at least 2/3 seeds positive, and paired mAP50-95 mean ≥ 0.
- **Ineffective**: paired mAP50 mean ≤ 0, or at most 1/3 seeds positive.
- **Insufficient evidence**: anything else (e.g. positive mean with negative mAP50-95).

Three seeds support no significance test: even 3/3 wins gives a sign-test p of 0.125, so "effective" is always phrased as a small, direction-consistent improvement, never a reliable per-run one.

## Predictions (declared before the results)

1. **The non-transfer of the default wiring is a wiring property, not a backbone property**: on YOLO11n / 12n / 26n the `rewire` arm's paired mAP50 mean beats the same backbone's default `esmoe` arm.
2. **The default wiring hurts large objects on every generation**: the default arms of 12n / 26n show a negative mean APl (the P5 lateral bypass exists in all four structures).
3. **YOLO11n's default arm stays below "effective" under the protocol budget** (it was negative at 640/20ep; if 800/120ep turns it positive, the earlier "does not transfer" conclusion gets corrected publicly).

Wrong predictions get recorded as wrong - that is what this page is for.

## Evidence chain

One JSON per run (`results/`, with git_ref, environment, budget, seed, metrics, artifact path); buckets and router analyses are recomputed from checkpoints by `scripts/buckets.py` / `scripts/routing.py`. The 36 `best.pt` files and training arguments live on the [`checkpoints` branch](https://github.com/Lfan-ke/ES-MoE/tree/checkpoints) (Git LFS).

## Verdicts (2026-09-04, after the full matrix)

All 36 runs — four backbone generations × three arms × three seeds — completed under the protocol budget (full VisDrone training, 800px, 120 epochs, patience 0, batch 32, same-seed pairing). Cell by cell against the lines above:

| backbone | default `esmoe` | `rewire` |
|:--:|:--:|:--:|
| YOLOv8n | effective (mAP50 +0.0025, 2/3; mAP50-95 +0.0004) | effective (+0.0036, 3/3; +0.0011) |
| YOLO11n | effective (+0.0013, 2/3; +0.0001) | effective (+0.0004, 2/3; +0.0002) |
| YOLO12n | ineffective (−0.0018, 0/3; −0.0021) | effective (+0.0001, 2/3; +0.0006) |
| YOLO26n | ineffective (−0.0034, 0/3; −0.0022) | ineffective (−0.0005, 1/3; +0.0008) |

As declared, three seeds support no significance test; every "effective" is a small, direction-consistent improvement. YOLO12n's `rewire` clears the line with a near-zero mean and is better read as parity.

### Predictions, settled

1. **Partly right.** The `rewire` arm beats the default arm on 12n (+0.0001 vs −0.0018) and 26n (−0.0005 vs −0.0034), and loses to it on 11n (+0.0004 vs +0.0013).
2. **Wrong.** The default arms' mean APl is +0.0068 on 12n and +0.0085 on 26n — not negative. "The default wiring hurts large objects" was a v8n-specific finding (APl 0/3, mean −0.0104) that did not transfer. What 26n's default arm consistently loses is small objects (APs 0/3, mean −0.0045).
3. **Wrong; corrected publicly.** YOLO11n's default arm is effective under the protocol budget (+0.0013, 2/3; mAP50-95 +0.0001). The earlier "does not transfer across backbones" conclusion, drawn from 640px/20ep runs, does not hold: the small-budget negative flipped positive at full budget, and the claim is withdrawn.

### What the full matrix does support

- **The default wiring's effect decays monotonically with backbone generation**: +0.0025 (v8n) → +0.0013 (11n) → −0.0018 (12n) → −0.0034 (26n). The newer the backbone end (SPPF → C2PSA → A2C2f → E2E head), the worse the same graft point fares. **Withdrawn at the second-round verdicts below**: with seven generations the order no longer holds by version number, and the grouping is by what the backbone ends in.
- **`rewire` pulls 12n and 26n back to near parity** and is the only 3/3 arm on v8n; only on 11n does it trail the default arm. The wiring — whether consumers read the block's output — moves the metric more than the backbone does, but not in one direction everywhere.
- **Where the damage lands depends on the backbone**: v8n loses large objects, 26n loses small ones, 12n is direction-unstable. No fixed "this block helps/hurts scale X" statement survives the four-generation matrix.
- **Routing collapses identically on all four generations**: no dead experts, one dominant expert, near-uniform mean probabilities, no scale specialisation; five of 26n's six checkpoints hand the lead to the `k = 9` expert. Rewiring changes none of it.

## Second pre-registration (2026-09-05, declared before any YOLOv5 / v9 / v10 result)

The four-generation matrix is settled. **Unknown**: all 27 runs of YOLOv5n / YOLOv9t / YOLOv10n (same protocol, three arms, three seeds). The judgment lines above carry over; the predictions:

1. **v5n's default arm has a positive paired mAP50 mean** — its backbone end is SPPF, the same family as v8n; the old end of the "monotonic decay with generation" line should sit above zero.
2. **v10n's default-arm mean is below v5n's** — v10 ends in an attention block (PSA), and the "newer end fares worse" ordering puts it under v5.
3. **The routing collapse repeats on all three** — no dead experts, one dominant expert, near-uniform mean probabilities, no scale specialisation. This is the mechanistic bet: a failure here would carry more information than the other two combined.

Wrong predictions get recorded as wrong.

### Addendum (2026-09-07, still before any v5 / v9 / v10 result)

With two MetaX C500 boxes available the queue is v9t nine runs, v10n nine runs, v5n nine runs. Two points belong on the record in advance:

1. **v5n is rerun as a whole generation**, completing the missing `rewire` arm and yielding a same-configuration replication on another host. `scripts/report.py` carries the hardware stack in its grouping key, so two hosts form two groups rather than folding into one cell as repeats.
2. **v10n gains seeds 3 and 4.** Box b's queue is shorter than box a's, and the spare hours extend **all three** v10n arms to five seeds. Extending every arm together, and deciding it before any v10n result exists, keeps this from being a cell chosen after the fact. The judgment lines are unchanged; five seeds put the best possible sign test at p = 0.031, still short of licensing a claim about any single run.

## Second-round verdicts (2026-09-09, after 33 runs)

YOLOv5n, v9t and v10n all finished under the protocol. v5n was rerun as a whole generation on the new hardware stack, and every v10n arm was extended to five seeds. Cell by cell:

| backbone | default `esmoe` | `rewire` |
|:--:|:--:|:--:|
| YOLOv5n (`metax3.3`) | effective (+0.0055, 3/3; mAP50-95 +0.0038) | effective (+0.0024, 2/3; +0.0018) |
| YOLOv5n (`metax3.7`) | effective (+0.0041, 3/3; +0.0026) | one seed only, not judged |
| YOLOv9t | effective (+0.0025, 2/3; +0.0009) | ineffective (−0.0013, 1/3; −0.0021) |
| YOLOv10n (5 seeds) | ineffective (−0.0002, 3/5; −0.0003) | ineffective (−0.0031, 1/5; −0.0016) |

YOLOv10n's default arm sits on zero with 3 of 5 seeds positive; the line calls that ineffective, and parity is the fairer reading.

### Predictions, settled

1. **Right.** v5n's default arm is positive on both stacks: +0.0055 (3/3) and +0.0041 (3/3).
2. **Right.** v10n's default arm (−0.0002) sits below v5n's (+0.0055).
3. **Right.** The collapse repeats across 22 checkpoints: no dead experts, a dominant expert holding 0.491–0.921 of the top-1 (0.71–0.82 by generation), mean-probability entropy at 89.8%–97.2% of its maximum, the leader changing with the seed, and the k = 3 expert never leading.

### Replication on another host

The same v5n configuration ran nine times on each of two C500 hosts: +0.0041 (`metax3.7`) against +0.0055 (`metax3.3`), agreeing in sign, wins and magnitude. The paired delta reproduces.

One self-check worth recording: v5n's `rewire` arm read +0.0074 on a single seed and +0.0024 once all three were in. "No mean below three seeds" exists for exactly that.

### Seven generations retire the earlier wording

At four generations this page said the effect "decays monotonically with backbone generation". With seven it no longer does by version number - v10n (−0.0002) falls below 11n (+0.0013). Grouped by what the backbone ends in, it is orderly:

| backbone end | backbones | default-arm mean |
|:--:|:--:|:--:|
| SPPF family | v5n / v8n / v9t | +0.0055 / +0.0025 / +0.0025 |
| attention | v10n (PSA) / 11n (C2PSA) | −0.0002 / +0.0013 |
| area attention | 12n (A2C2f) | −0.0018 |
| E2E head | 26n | −0.0034 |

The sign changes when the backbone end becomes an attention block, not as release numbers rise. The "monotonic with generation" wording is withdrawn.

## Third pre-registration (2026-09-09, declared before any `gshard` result)

Reading YOLO-Master's own `ES_MOE` shows it optimises a different balancing term from this toolkit's default:

| | formula | reads |
|:--:|:--:|:--:|
| upstream `ES_MOE` | `N · Σ usage²` (GShard) | mean routing probabilities |
| default `switch_balance` | `E · Σ p̄ᵢfᵢ` (Switch) | mean probabilities × realised load |

Both sit at their minimum in exactly the collapse we measured: with mean probabilities near uniform and one expert taking 62%–92% of the top-1, the Switch term is fixed at `k` and the GShard term at 1.0, and moving the top-k dispatch does not shift either. They separate only when the probability mass itself concentrates, where both rise.

`gshard_balance` is implemented and exported, and `scripts/train.py --balance gshard` selects it. **Unknown**: the `gshard` arms of v5n and v10n, three seeds each. The predictions:

1. **Switching to the upstream GShard objective will not undo the routing collapse**: the `gshard` arms still show a dominant expert above 0.6 top-1 share, mean-probability entropy above 90% of its maximum, and no dead experts. This is the mechanistic bet - if the collapse does lift, the choice of objective was the cause and this package's default should change.
2. **Accuracy will not move materially**: paired mAP50 means of the `gshard` arms sit within ±0.003 of their `switch` counterparts on v5n and v10n, the scale of the seed spread.

Wrong predictions get recorded as wrong.

### Correction (2026-09-09, still before any `gshard` result)

The claim above that both objectives "sit at their minimum" in the measured collapse was computed on a hand-made uniform matrix and **does not hold on the real records**; correcting it. Taking `yolov9t-esmoe-s0`: the mean probabilities are `[0.206, 0.213, 0.200, 0.381]`, not uniform, and the GShard term evaluates to 1.0919, above its 1.0 minimum. It does register the concentration.

The real gap is the coefficient. Upstream's `ES_MOE` carries `balance_loss_coeff = 1.0`, passes through `moe_gain = 1.0`, and lands as `total = native_loss + aux`; this package defaults to `attach_aux_loss(weight=0.01)`. On that same record the balancing gradient on the mean probabilities is 0.0200 here against 0.6159 upstream, a factor of **31**.

The two predictions keep their content but change their reason: the collapse is unlikely to lift on a change of objective because this package applies the balancing term at one hundredth of upstream's weight, not because both objectives are blind. **A third prediction follows**:

3. **Raising the weight to upstream's 1.0 moves the routing**: under `--aux-weight 1.0` the dominant expert's top-1 share falls relative to 0.01, and by more than switching the objective does. If raising the weight moves nothing either, the balancing term is not the culprit and the router is.

### Fourth pre-registration (2026-09-09, declared before any `master` result)

Reading the paper itself (arXiv 2512.23273, section 3.5) shows that **the balancing term the paper specifies is not the one the released code implements**. Paper equation (13):

$$ \mathcal{L}_{LB} = \frac{1}{E}\sum_{i=1}^{E}\left(\mu_i - \frac{1}{E}\right)^2 $$

where μᵢ averages **Ω_train** - the weights after the top-K mask and renormalisation (paper equation 8) - over the batch and spatial positions. Upstream's `ES_MOE` instead applies the GShard form `N·Σusage²` to the **raw router probabilities**. On one pair of inputs:

| objective | balanced dispatch | collapsed dispatch | separates |
|:--:|:--:|:--:|:--:|
| `switch_balance` (this package's default) | 2.00000 | 2.00000 | no |
| `gshard_balance` (upstream code) | 1.00000 | 1.00000 | no |
| `master_balance` (paper eq. 13) | 0.00000 | 0.00383 | **yes** |

Both inputs carry uniform mean probabilities and differ only in whether the top-k dispatch collapses onto one expert. The first two read the probabilities and cannot tell them apart; the paper's reads the gated weights and can. **The collapse measured across seven generations sits inside the blind spot of the first two.**

`master_balance` is implemented and `--balance master` selects it. **Unknown**: the `master` arms of v5n and v10n, two seeds each. The prediction:

4. **The paper's term measurably lowers the dominant expert's top-1 share**: the `master` arm's dominant share falls below the same backbone's `switch` arm, and by more than the `gshard` arm does. This is the strongest mechanistic bet of the four - if this one does not move either, the collapse is independent of the balancing term's form and the router or the graft point is responsible, and the explanations offered in the first three rounds give way.

The paper states only λ_LB > 0 without a value, so the `master` arm keeps this package's default weight of 0.01 and stays separate from the coefficient comparison.

### Correction (evening of 2026-09-09, still before any `master` or structural result)

The table above in the fourth round -- "only the paper's objective sees a collapsed dispatch" -- **rests on a false premise**, corrected here.

Reading upstream line by line: `ES_MOE._compute_load_balancing_loss` applies the GShard form to whatever `DynamicRoutingLayer` returns, and in training that is the output of `_soft_top_k(...)` -- **the gate, after the top-k mask and renormalisation**. The paper's equation 12 averages the same tensor. Therefore

$$\mathcal{L}_{\text{paper}}=\frac{\mathcal{L}_{\text{upstream}}-1}{E^2}$$

**The paper and the released code state one objective, differing only by an affine map**, with the same minimiser and proportional gradients. The earlier claim that they differ was wrong.

What was wrong is this package: `gshard_balance` had been written against the **raw softmax**, which is neither. It now reads the gate; the probability-reading form is kept as `gshard_probs_balance`. Consequently:

- The `gshard` row of that table measured `gshard_probs` and **is relabelled accordingly**; the five finished records have had their `balance` field corrected in place.
- The fourth prediction is **not withdrawn**: it asks whether an objective reading the gate lifts the collapse, which remains open. The `master` arm is running.
- The dividing line is restated: **whether the term reads the probabilities or the gate**, not Switch versus GShard.

### Fifth pre-registration (evening of 2026-09-09, declared before any structural result)

The same line-by-line comparison found three structural differences that no experiment had covered. All three are switches, defaulting to the old behaviour so the 77 records already in `results/` stay reproducible, and each becomes its own arm:

| arm | difference from upstream | switch |
|:--:|:--:|:--:|
| `stages` | upstream puts one block after every backbone stage, four in all; this package puts one | `--at backbone_stages` |
| `norm` | upstream normalises the mixed output with `BatchNorm+SiLU` (the paper's eq. 2 `Norm`) | `--out-norm` |
| `dense` | upstream runs every expert in training, the unrouted ones weighted zero, so their normalisation statistics keep moving | `--dense-training` |

The predictions:

5. **Block count is the dominant term: the `stages` arm's paired mAP50 mean beats the single-block arm**, by more than any change of balancing objective achieves. Four blocks carry four times the capacity and graft surface of one; if that does not move the metric either, the "graft location and capacity" explanation should retire.
6. **The `dense` arm's dominant expert holds a smaller top-1 share than its sparse counterpart.** In this package an unrouted expert's normalisation statistics freeze; upstream keeps them moving. If the collapse owes anything to that, this arm should reduce the share.
7. **The `norm` arm moves accuracy by less than ±0.003**, the scale of the seed spread. It is a question of structural completeness, not an expected source of accuracy.

Wrong predictions get recorded as wrong.

### Correction (late 2026-09-09, still before any structural-arm result)

The previous section relabelled five records' `balance` as `gshard_probs`. **That relabelling was also wrong**, and so is every result from the `--balance`, `--out-norm` and `--dense-training` switches. Both are corrected here.

The trainer rebuilds the model from `model.yaml`. An objective or a switch set on the blocks after `YOLO(cfg)` returns lands on an object that is then discarded — no error, no trace. So none of the three switches reached the model that trained: a run asking for `master` optimised the constructor default, a run asking for the output norm trained without one, and the record repeated the command line, which said something else.

The evidence is the weights, not the logs:

```python
model = torch.load("best.pt", map_location="cpu", weights_only=False)["model"]
[b.balance.__name__ for b in esmoe.blocks(model)]  # ['switch_balance']
```

A block pickled before a setting existed carries no attribute for it, and the absence is itself the answer. Therefore:

- Those five records trained `switch_balance`: same configuration and same seed as the default arm, run a second time. They are relabelled `switch` and join the default arm as repeats — a direct measure of run-to-run spread on one card, which is exactly what three-seed intervals lack.
- The two runs labelled `master` optimised **GShard read off the gate** (the md5 of `esmoe/module.py` matches the commit that switched to the gate, and `esmoe_aux` sits at 0.0101, which is $N\sum u^2 = 1$ under a uniform gate times the 0.01 weight). They are recorded as `gshard`.
- The fourth round's prediction stands and is still unanswered: whether an objective that reads the gate relieves the collapse. These two runs answer it, not the two that were thought to.

**The fix, and the audit.** `graft()` now writes the settings into the config (`[4, 2, null, {balance: …, out_norm: …, dense_training: …}]`), and `ESMoE` takes them as an options mapping and resolves an objective by name — so a rebuild cannot drop them. `scripts/train.py` records the block settings read off the trained model instead of repeating the request, and refuses to start when the two disagree; `tests/test_ultralytics.py` trains a model to hold that path.

The audit does not rely on memory: `scripts/blockspec.py` reads each checkpoint's block settings on the host that holds it, and `scripts/backfill.py --settings` rewrites the records to match, naming the change inside the record. All 120 records here were checked against their checkpoints and **none needed correcting** — the published data is sound; what was wrong were the new arms, none of them merged yet.

**The general lesson**: a `--flag` in a record does not mean the flag reached the model. Read the record off the trained model; do not repeat the command line.

## Fourth-round verdict (2026-09-10, the gate-reading objective has run)

The fourth round asked whether an objective that can see the collapse relieves it. Two v5n runs (seeds 0 and 1) are in, and the answer is **no** -- and it moved the other way.

The same `scripts/routing.py` over the 548 VisDrone validation images:

| objective | checkpoints | top-two usage, of 2.0 | lowest usage | dead experts | probability entropy |
|:--:|:--:|:--:|:--:|:--:|:--:|
| Switch (every earlier record) | 46 | 1.265–1.777 (mean 1.468) | 0.0365 | 0 of 46 | 90%–97% |
| GShard on the gate | 2 | 1.610–1.898 (mean 1.754) | 0.0091 | 1 of 2 | 97%–98% |

Both checkpoints sit in the upper half of the Switch range or above it, and seed 1's 1.898 is the most concentrated dispatch among all 48 checkpoints. Seed 0 holds the **first dead expert in the project** (usage 0.0091); none of the 46 earlier checkpoints has one. Meanwhile the probability entropy is *higher*, 97%–98% against 90%–97%: **flatter probabilities, more concentrated dispatch**, which is exactly the dissociation this line of work has been chasing.

The paired accuracy is +0.0018 over two seeds (1/2 wins), inside the run-to-run spread and therefore not evidence of anything.

**The confound was stated backwards here, and this corrects it.** The first version read: the two terms differ in scale at one weight, so this arm carried about half the pressure. That compares the terms' **values** at their optimum, which says nothing about pressure. What decides how much a weight buys is the **gradient** the term puts on the router's logits. Measured at the mean routing probabilities each checkpoint actually converged to (`scripts/pressure.py`, output in `results/pressure.md`):

| objective | gradient relative to Switch, across the v5n checkpoints |
|:--:|:--:|
| `gshard`, reading the gate | 0.59x – 1.48x |
| `master`, the paper's eq. 13 | about 1/11 to 1/134, which is the `1/E^2` factor |

**At the same `weight=0.01` the gate-reading term pushes as hard as the arm it is compared against**, not half as hard. The "it had less pressure" escape therefore does not hold, and the verdict is firmer for it: **under comparable balancing pressure, reading the gate did not relieve the collapse and the dispatch came out more concentrated.**

What does remain is the sample: two seeds against forty-six, one backbone against seven.

**The fourth prediction is not borne out, and is recorded as such.** `master` genuinely does sit an order of magnitude lower, so answering "is the paper's scale enough" needs its weight raised by `1/E^2` and a run of its own, which was not made.

## Run-to-run spread at one configuration (2026-09-09, not planned)

The six mislabelled runs of the previous section trained exactly what the default arm trains: same configuration, same seed, same card, run a second time. They are therefore not extra samples but a direct measure of something nothing here had measured -- **how far apart two runs of one configuration land.**

`scripts/report.py` computes the table at the end of `results/summary.md`:

| stack | epochs | repeats | mean gap | largest gap |
|:--:|:--:|:--:|:--:|:--:|
| metaxc500 / metax3.3 | 120 | 5 | 0.0045 | 0.0130 |
| metaxc500 / metax3.3 | 1 | 1 | 0.0000 | 0.0000 |
| 4090d / torch2.6 | 20 | 1 | 0.0000 | 0.0000 |

That is the denominator for every conclusion here. Across seven generations the default arm's mean paired delta is v5n +0.0055, v8n +0.0025, v9t +0.0025, v10n −0.0002, 11n +0.0013, 12n −0.0018, 26n −0.0034 -- **six of the seven inside the mean gap between two runs of one configuration (0.0045)**; only v5n's +0.0055 clears it, and it still sits well inside the largest gap observed, 0.0130.

Three things follow, and all three are stated rather than softened:

1. **`deterministic=True` does not hold on this accelerator.** The trainer fixes the seed and sets the deterministic flag; on the 4090 a repeat reproduces bit for bit (one pair, 20 epochs), on the MetaX C500 it does not (five pairs, 120 epochs). The divergence accumulates with training: at one epoch the gap is zero.
2. **The generation with the largest positive effect sits on the noisier machine.** v5n's +0.0055 and +0.0041 both come from the MetaX cards; the four measured on the 4090 (v8n, 11n, 12n, 26n) rest on a single same-configuration repeat so far.
3. **This does not overturn the consistency of the signs, but it does overturn reading the magnitudes.** "Positive on this generation, negative on that one" is carried by the sign counts across seeds. "The block is worth +0.0055" is not: two runs of one configuration already differ by 0.0130.

Accuracy claims here therefore stop at direction, with the sample size and interval stated alongside; anywhere an earlier judgment line reasoned from a magnitude, it is downgraded to direction. **This is the most useful negative result of the project, and it is a by-product of an implementation defect -- without those six runs being mislabelled, nobody would have paid for a repeat of an identical configuration.**

## Fifth-round verdicts (2026-09-10, the three structural arms have run)

Three arms at three seeds each, paired against a baseline of the same seed on the same card:

| arm | relation to upstream | paired mAP50 | 95% CI | wins | mAP50-95 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| `norm` (v5n) | always on upstream, off here | **+0.0038** | [−0.0033, +0.0109] | **3/3** | +0.0022 (3/3) |
| `dense` (v5n) | always on upstream, off here | **+0.0031** | [−0.0006, +0.0068] | **3/3** | +0.0034 (3/3) |
| `stages` (v10n, four blocks) | upstream's layout, one block here | **−0.0071** | [−0.0162, +0.0019] | **0/3** | −0.0048 (0/3) |

**In one line: both of upstream's block-internal defaults are positive and 3/3; upstream's block-count layout is negative and 0/3.** Of the things this package had not copied, the good ones were inside the block and the bad one was the count.

### Predictions settled

**Fifth (the `stages` arm beats the single-block arm, by more than any balancing objective achieved) — refuted, and the other way round.** Four blocks did not lift the metric; they are the only cell of twenty-one that is 0/3 with both metrics negative. The prediction itself said "if it cannot move the metric either, then the graft-location-and-capacity explanation should retire". By its own terms, **it retires.** The parameter count is higher (3.15M against 3.03M), so this is not a capacity shortfall.

The confound is measured, not assumed: four blocks sum four auxiliary terms, so at one weight `esmoe_aux` goes from 0.020 to 0.080, exactly fourfold. Separating "block count" from "fourfold balancing pressure" needs the four-block arm at `weight=0.0025`; that arm is queued and the verdict waits on it.

**Sixth (the `dense` arm's leading expert holds a smaller top-1 share) — partly, and not on the quantity that was registered.** Per seed: −0.277, +0.095, −0.017, a mean of −0.066 at **2/3**, carried almost entirely by seed 0. On the share the top two experts take instead, dense is lower on 3/3 (1.392/1.403/1.555 against 1.431/1.610/1.646). **What was registered was the top-1 share, and on that quantity this is not established**; "dense spreads the dispatch" holds on the other measure.

**Seventh (`norm` moves accuracy by less than ±0.003) — refuted, in the positive direction.** +0.0038 is outside the band. More interesting is what came with it: the `norm` arm's leading expert holds a **larger** top-1 share than the sparse arm by 0.089, on 3/3 seeds. **More concentrated, and more accurate.**

### One thing that matters more than the three predictions

Put "how concentrated the dispatch is" beside "how much the pairing moved", over the 58 runs that have both a routing analysis and a paired delta:

$$r = +0.160$$

The leading expert's top-1 share runs from 0.49 to 0.92 and the paired mAP50 delta from −0.0113 to +0.0109, and the two are **almost unrelated — with what relationship there is pointing the wrong way.** `norm` is more concentrated and more accurate; `dense` is more spread and more accurate. Both are instances of the same thing.

**This unsettles the premise of the whole line of questioning.** Four rounds here asked how to relieve routing collapse, assuming collapse costs accuracy. Under these conditions that assumption has no support: **collapse and accuracy are unrelated in this data.** Whether a balancing term belongs, and how large it should be, therefore cannot be argued from "it prevents collapse"; it has to be settled by what the term does to the metric — and that effect currently sits inside the run-to-run spread.

The correlation is recomputed by `scripts/routing.py --summarise` at the top of `results/routing.md`; it is not transcribed.

### Results from the alignment arms

Filled in as the runs land, from the same numbers as `results/summary.md`.

<div id="esmoe-alignment"></div>
