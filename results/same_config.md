# Same configuration: YOLO-Master's fork against official ultralytics + esmoe

## metrics/mAP50(B)

| seed | card | metrics | A | A0 | B | C | A - B | A - A0 | B - C | (A - A0) - (B - C) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | metaxc500/metax3.3 | measured | 0.3637 | 0.3542 | 0.3739 | 0.3597 | -0.0102 | +0.0095 | +0.0142 | -0.0047 |
| 1 | metaxc500/metax3.3 | measured | 0.3703 | 0.3534 | 0.3724 | 0.3589 | -0.0021 | +0.0170 | +0.0135 | +0.0035 |

| difference | seeds | mean | 95% CI | positive |
|:--:|:--:|:--:|:--:|:--:|
| A - B | 2 | -0.0061 | [-0.0578, +0.0456] | 0/2 |
| A - A0 | 2 | +0.0132 | [-0.0340, +0.0605] | 2/2 |
| B - C | 2 | +0.0138 | [+0.0095, +0.0182] | 2/2 |
| (A - A0) - (B - C) | 2 | -0.0006 | [-0.0522, +0.0510] | 1/2 |

## metrics/mAP50-95(B)

| seed | card | metrics | A | A0 | B | C | A - B | A - A0 | B - C | (A - A0) - (B - C) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | metaxc500/metax3.3 | measured | 0.2098 | 0.2044 | 0.2168 | 0.2065 | -0.0070 | +0.0054 | +0.0103 | -0.0049 |
| 1 | metaxc500/metax3.3 | measured | 0.2146 | 0.2027 | 0.2163 | 0.2048 | -0.0017 | +0.0120 | +0.0115 | +0.0005 |

| difference | seeds | mean | 95% CI | positive |
|:--:|:--:|:--:|:--:|:--:|
| A - B | 2 | -0.0043 | [-0.0380, +0.0293] | 0/2 |
| A - A0 | 2 | +0.0087 | [-0.0327, +0.0501] | 2/2 |
| B - C | 2 | +0.0109 | [+0.0036, +0.0183] | 2/2 |
| (A - A0) - (B - C) | 2 | -0.0022 | [-0.0363, +0.0318] | 1/2 |

## What each run actually trained with

| seed | arm | run | amp at end | batch at end | epochs replayed | hours |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | A | yolo-master-n-upstream-e120-s0-p800c-fork-20260911060342 | False | 32 | 1 | 10.80 |
| 0 | A0 | yolo-master-n-baseline-e120-s0-p800c-fork-20260911165202 | False | 32 | 1 | 6.95 |
| 0 | B | yolo-master-n-esmoe-upstream-w1-e120-s0-p800c-20260911234908 | True | 32 | 0 | 4.59 |
| 0 | C | yolo-master-n-baseline-e120-s0-p800c-20260911165332 | True | 32 | 0 | 4.28 |
| 1 | A | yolo-master-n-upstream-e120-s1-p800d-fork-20260911060354 | False | 32 | 1 | 10.79 |
| 1 | A0 | yolo-master-n-baseline-e120-s1-p800d-fork-20260911165201 | False | 32 | 1 | 6.97 |
| 1 | B | yolo-master-n-esmoe-upstream-w1-e120-s1-p800d-20260911235018 | True | 32 | 0 | 4.74 |
| 1 | C | yolo-master-n-baseline-e120-s1-p800d-20260911165331 | True | 32 | 0 | 4.28 |
