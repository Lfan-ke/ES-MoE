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

$$ \mathcal{L}_{LB} = rac{1}{E}\sum_{i=1}^{E}\left(\mu_i - rac{1}{E}ight)^2 $$

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
