## Runs

| run | variant | seed | mAP50 | mAP50-95 | params | wall_s |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| yolov8n-baseline-e20-s0-20260829080043 | baseline | 0 | 0.0933 | 0.0419 | 3012798 | 362.8 |
| yolov8n-baseline-e20-s1-20260829081257 | baseline | 1 | 0.0892 | 0.0393 | 3012798 | 365.5 |
| yolov8n-baseline-e20-s2-20260829082514 | baseline | 2 | 0.0903 | 0.0402 | 3012798 | 389.9 |
| yolov8n-esmoe-e20-s0-20260829080650 | esmoe-e4k2w0.01 | 0 | 0.0950 | 0.0424 | 3327330 | 362.7 |
| yolov8n-esmoe-e20-s0-e2k1w0.01-20260829083842 | esmoe-e2k1w0.01 | 0 | 0.0903 | 0.0399 | 3161888 | 363.0 |
| yolov8n-esmoe-e20-s0-e4k1w0.01-20260829084450 | esmoe-e4k1w0.01 | 0 | 0.0928 | 0.0415 | 3327330 | 391.4 |
| yolov8n-esmoe-e20-s0-e4k2w0.0-20260829090326 | esmoe-e4k2w0.0 | 0 | 0.0938 | 0.0418 | 3327330 | 402.0 |
| yolov8n-esmoe-e20-s0-e8k2w0.01-20260829085727 | esmoe-e8k2w0.01 | 0 | 0.0934 | 0.0414 | 3781094 | 355.0 |
| yolov8n-esmoe-e20-s1-20260829081907 | esmoe-e4k2w0.01 | 1 | 0.0907 | 0.0399 | 3327330 | 362.7 |
| yolov8n-esmoe-e20-s2-20260829083149 | esmoe-e4k2w0.01 | 2 | 0.0932 | 0.0406 | 3327330 | 378.9 |

## Across seeds

| variant | seeds | mAP50 | mAP50-95 |
|:--:|:--:|:--:|:--:|
| baseline | 3 | 0.0909 ± 0.0022 | 0.0405 ± 0.0013 |
| esmoe-e4k2w0.01 | 3 | 0.0930 ± 0.0022 | 0.0410 ± 0.0013 |
| esmoe-e2k1w0.01 | 1 | 0.0903 | 0.0399 |
| esmoe-e4k1w0.01 | 1 | 0.0928 | 0.0415 |
| esmoe-e4k2w0.0 | 1 | 0.0938 | 0.0418 |
| esmoe-e8k2w0.01 | 1 | 0.0934 | 0.0414 |

## Paired against baseline - metrics/mAP50(B)

| variant | seed | baseline | variant | delta |
|:--:|:--:|:--:|:--:|:--:|
| esmoe-e4k2w0.01 | 0 | 0.0933 | 0.0950 | +0.0017 |
| esmoe-e2k1w0.01 | 0 | 0.0933 | 0.0903 | -0.0031 |
| esmoe-e4k1w0.01 | 0 | 0.0933 | 0.0928 | -0.0006 |
| esmoe-e4k2w0.0 | 0 | 0.0933 | 0.0938 | +0.0005 |
| esmoe-e8k2w0.01 | 0 | 0.0933 | 0.0934 | +0.0000 |
| esmoe-e4k2w0.01 | 1 | 0.0892 | 0.0907 | +0.0015 |
| esmoe-e4k2w0.01 | 2 | 0.0903 | 0.0932 | +0.0029 |

| variant | seeds | mean delta | wins |
|:--:|:--:|:--:|:--:|
| esmoe-e4k2w0.01 | 3 | +0.0021 | 3/3 |
| esmoe-e2k1w0.01 | 1 | -0.0031 | 0/1 |
| esmoe-e4k1w0.01 | 1 | -0.0006 | 0/1 |
| esmoe-e4k2w0.0 | 1 | +0.0005 | 1/1 |
| esmoe-e8k2w0.01 | 1 | +0.0000 | 1/1 |

## Paired against baseline - metrics/mAP50-95(B)

| variant | seed | baseline | variant | delta |
|:--:|:--:|:--:|:--:|:--:|
| esmoe-e4k2w0.01 | 0 | 0.0419 | 0.0424 | +0.0006 |
| esmoe-e2k1w0.01 | 0 | 0.0419 | 0.0399 | -0.0020 |
| esmoe-e4k1w0.01 | 0 | 0.0419 | 0.0415 | -0.0004 |
| esmoe-e4k2w0.0 | 0 | 0.0419 | 0.0418 | -0.0000 |
| esmoe-e8k2w0.01 | 0 | 0.0419 | 0.0414 | -0.0005 |
| esmoe-e4k2w0.01 | 1 | 0.0393 | 0.0399 | +0.0006 |
| esmoe-e4k2w0.01 | 2 | 0.0402 | 0.0406 | +0.0005 |

| variant | seeds | mean delta | wins |
|:--:|:--:|:--:|:--:|
| esmoe-e4k2w0.01 | 3 | +0.0005 | 3/3 |
| esmoe-e2k1w0.01 | 1 | -0.0020 | 0/1 |
| esmoe-e4k1w0.01 | 1 | -0.0004 | 0/1 |
| esmoe-e4k2w0.0 | 1 | -0.0000 | 0/1 |
| esmoe-e8k2w0.01 | 1 | -0.0005 | 0/1 |

## Determinism (repeated runs of an identical config)

| variant | seed | mAP50 first | mAP50 repeat | identical |
|:--:|:--:|:--:|:--:|:--:|
| esmoe-e4k2w0.01 | 0 | 0.0950 | 0.0950 | yes |
