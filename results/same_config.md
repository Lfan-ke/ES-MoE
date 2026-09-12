# Same configuration: YOLO-Master's fork against official ultralytics + esmoe

## metrics/mAP50(B)

| precision | seed | card | metrics | A | A0 | B | C | A - B | A - A0 | B - C | (A - A0) - (B - C) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| mixed | 0 | metaxc500/metax3.3 | measured | 0.3637 | 0.3542 | 0.3739 | 0.3597 | -0.0102 | +0.0095 | +0.0142 | -0.0047 |
| mixed | 1 | metaxc500/metax3.3 | measured | 0.3703 | 0.3534 | 0.3724 | 0.3589 | -0.0021 | +0.0170 | +0.0135 | +0.0035 |
| mixed | 2 | metaxc500/metax3.3 | measured | 0.3615 | 0.3514 | 0.3698 | 0.3598 | -0.0084 | +0.0101 | +0.0100 | +0.0001 |

| precision | difference | seeds | mean | 95% CI | positive |
|:--:|:--:|:--:|:--:|:--:|:--:|
| mixed | A - B | 3 | -0.0069 | [-0.0175, +0.0037] | 0/3 |
| mixed | A - A0 | 3 | +0.0122 | [+0.0019, +0.0225] | 3/3 |
| mixed | B - C | 3 | +0.0126 | [+0.0070, +0.0181] | 3/3 |
| mixed | (A - A0) - (B - C) | 3 | -0.0004 | [-0.0105, +0.0098] | 2/3 |

## metrics/mAP50-95(B)

| precision | seed | card | metrics | A | A0 | B | C | A - B | A - A0 | B - C | (A - A0) - (B - C) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| mixed | 0 | metaxc500/metax3.3 | measured | 0.2098 | 0.2044 | 0.2168 | 0.2065 | -0.0070 | +0.0054 | +0.0103 | -0.0049 |
| mixed | 1 | metaxc500/metax3.3 | measured | 0.2146 | 0.2027 | 0.2163 | 0.2048 | -0.0017 | +0.0120 | +0.0115 | +0.0005 |
| mixed | 2 | metaxc500/metax3.3 | measured | 0.2094 | 0.2010 | 0.2139 | 0.2069 | -0.0045 | +0.0084 | +0.0070 | +0.0014 |

| precision | difference | seeds | mean | 95% CI | positive |
|:--:|:--:|:--:|:--:|:--:|:--:|
| mixed | A - B | 3 | -0.0044 | [-0.0110, +0.0022] | 0/3 |
| mixed | A - A0 | 3 | +0.0086 | [+0.0005, +0.0167] | 3/3 |
| mixed | B - C | 3 | +0.0096 | [+0.0038, +0.0154] | 3/3 |
| mixed | (A - A0) - (B - C) | 3 | -0.0010 | [-0.0094, +0.0074] | 2/3 |

## What each run actually trained with

| precision | seed | arm | run | amp asked | amp at end | batch at end | epochs replayed | hours |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| mixed | 0 | A | yolo-master-n-upstream-e120-s0-p800c-fork-20260911060342 | True | False | 32 | 1 | 10.80 |
| mixed | 0 | A0 | yolo-master-n-baseline-e120-s0-p800c-fork-20260911165202 | True | False | 32 | 1 | 6.95 |
| mixed | 0 | B | yolo-master-n-esmoe-upstream-w1-e120-s0-p800c-20260911234908 | True | True | 32 | 0 | 4.59 |
| mixed | 0 | C | yolo-master-n-baseline-e120-s0-p800c-20260911165332 | True | True | 32 | 0 | 4.28 |
| mixed | 1 | A | yolo-master-n-upstream-e120-s1-p800d-fork-20260911060354 | True | False | 32 | 1 | 10.79 |
| mixed | 1 | A0 | yolo-master-n-baseline-e120-s1-p800d-fork-20260911165201 | True | False | 32 | 1 | 6.97 |
| mixed | 1 | B | yolo-master-n-esmoe-upstream-w1-e120-s1-p800d-20260911235018 | True | True | 32 | 0 | 4.74 |
| mixed | 1 | C | yolo-master-n-baseline-e120-s1-p800d-20260911165331 | True | True | 32 | 0 | 4.28 |
| mixed | 2 | A | yolo-master-n-upstream-e120-s2-p800e-fork-20260911100545 | True | False | 32 | 1 | 10.68 |
| mixed | 2 | A0 | yolo-master-n-baseline-e120-s2-p800e-fork-20260911204651 | True | False | 32 | 1 | 7.02 |
| mixed | 2 | B | yolo-master-n-esmoe-upstream-w1-e120-s2-p800e-20260912034809 | True | True | 32 | 0 | 4.67 |
| mixed | 2 | C | yolo-master-n-baseline-e120-s2-p800e-20260911204822 | True | True | 32 | 0 | 4.29 |
