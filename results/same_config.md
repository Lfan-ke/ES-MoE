# Same configuration: YOLO-Master's fork against official ultralytics + esmoe

## metrics/mAP50(B)

| seed | card | metrics | A | A0 | B | C | A - B | A - A0 | B - C | (A - A0) - (B - C) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | metaxc500/metax3.3 | measured | 0.3637 | 0.3542 | 0.3739 | 0.3597 | -0.0102 | +0.0095 | +0.0142 | -0.0047 |

| difference | seeds | mean | 95% CI | positive |
|:--:|:--:|:--:|:--:|:--:|
| A - B | 1 | -0.0102 | - | 0/1 |
| A - A0 | 1 | +0.0095 | - | 1/1 |
| B - C | 1 | +0.0142 | - | 1/1 |
| (A - A0) - (B - C) | 1 | -0.0047 | - | 0/1 |

## metrics/mAP50-95(B)

| seed | card | metrics | A | A0 | B | C | A - B | A - A0 | B - C | (A - A0) - (B - C) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | metaxc500/metax3.3 | measured | 0.2098 | 0.2044 | 0.2168 | 0.2065 | -0.0070 | +0.0054 | +0.0103 | -0.0049 |

| difference | seeds | mean | 95% CI | positive |
|:--:|:--:|:--:|:--:|:--:|
| A - B | 1 | -0.0070 | - | 0/1 |
| A - A0 | 1 | +0.0054 | - | 1/1 |
| B - C | 1 | +0.0103 | - | 1/1 |
| (A - A0) - (B - C) | 1 | -0.0049 | - | 0/1 |

## What each run actually trained with

| seed | arm | run | amp at end | batch at end | epochs replayed | hours |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | A | yolo-master-n-upstream-e120-s0-p800c-fork-20260911060342 | False | 32 | 1 | 10.80 |
| 0 | A0 | yolo-master-n-baseline-e120-s0-p800c-fork-20260911165202 | False | 32 | 1 | 6.95 |
| 0 | B | yolo-master-n-esmoe-upstream-w1-e120-s0-p800c-20260911234908 | True | 32 | 0 | 4.59 |
| 0 | C | yolo-master-n-baseline-e120-s0-p800c-20260911165332 | True | 32 | 0 | 4.28 |
