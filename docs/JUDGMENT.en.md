# Judgment lines (pre-registered)

The task book requires negative results to come with judgment lines defined in advance. This page fixes the criteria and predictions **before** the runs they govern finish; the git commit time of this page is the evidence.

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

One JSON per run (`results/`, with git_ref, environment, budget, seed, metrics, artifact path); buckets and router analyses are recomputed from checkpoints by `scripts/buckets.py` / `scripts/routing.py`.

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

- **The default wiring's effect decays monotonically with backbone generation**: +0.0025 (v8n) → +0.0013 (11n) → −0.0018 (12n) → −0.0034 (26n). The newer the backbone end (SPPF → C2PSA → A2C2f → E2E head), the worse the same graft point fares.
- **`rewire` pulls 12n and 26n back to near parity** and is the only 3/3 arm on v8n; only on 11n does it trail the default arm. The wiring — whether consumers read the block's output — moves the metric more than the backbone does, but not in one direction everywhere.
- **Where the damage lands depends on the backbone**: v8n loses large objects, 26n loses small ones, 12n is direction-unstable. No fixed "this block helps/hurts scale X" statement survives the four-generation matrix.
- **Routing collapses identically on all four generations**: no dead experts, one dominant expert, near-uniform mean probabilities, no scale specialisation; five of 26n's six checkpoints hand the lead to the `k = 9` expert. Rewiring changes none of it.
