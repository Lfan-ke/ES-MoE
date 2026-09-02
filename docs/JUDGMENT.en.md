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
