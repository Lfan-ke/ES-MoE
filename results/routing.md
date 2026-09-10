# Router behaviour on VisDrone val

Per checkpoint: the share of images on which each expert is the top-1 choice, the share on which it is in the top-2, the mean routing probability, and the correlation of that probability with the mean object size and the object count of the image.

## Does concentration cost accuracy?

Over the 58 runs that have both a paired delta and a routing analysis, the leading expert's top-1 share runs 0.49 to 0.92 and the paired mAP50 delta -0.0113 to +0.0109. Their correlation is **r = +0.160**: on this evidence a concentrated dispatch does not cost accuracy, which is worth holding against the premise that a balancing term is what the block needs.

## yolo11n-esmoe-e120-s0-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3335 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.168 | 0.233 | +0.23 | -0.46 |
| 1 | 5 | 0.653 | 0.911 | 0.309 | +0.26 | -0.31 |
| 2 | 7 | 0.257 | 0.465 | 0.210 | -0.31 | +0.51 |
| 3 | 9 | 0.089 | 0.456 | 0.247 | +0.17 | -0.40 |

## yolo11n-esmoe-e120-s1-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3486 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.027 | 0.135 | 0.230 | -0.02 | -0.04 |
| 1 | 5 | 0.190 | 0.527 | 0.229 | -0.30 | +0.47 |
| 2 | 7 | 0.133 | 0.500 | 0.238 | +0.31 | -0.18 |
| 3 | 9 | 0.650 | 0.838 | 0.303 | +0.05 | -0.29 |

## yolo11n-esmoe-e120-s2-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2719 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.009 | 0.620 | 0.264 | +0.26 | -0.46 |
| 1 | 5 | 0.084 | 0.354 | 0.246 | -0.06 | +0.11 |
| 2 | 7 | 0.786 | 0.849 | 0.352 | +0.18 | -0.22 |
| 3 | 9 | 0.120 | 0.177 | 0.138 | -0.21 | +0.29 |

## yolo11n-esmoe-rewire-e120-s0-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3233 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.210 | 0.226 | +0.27 | -0.62 |
| 1 | 5 | 0.788 | 0.943 | 0.338 | +0.18 | -0.18 |
| 2 | 7 | 0.153 | 0.392 | 0.194 | -0.29 | +0.48 |
| 3 | 9 | 0.058 | 0.454 | 0.241 | +0.20 | -0.41 |

## yolo11n-esmoe-rewire-e120-s1-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3112 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.179 | 0.207 | +0.14 | -0.37 |
| 1 | 5 | 0.031 | 0.316 | 0.211 | +0.01 | -0.28 |
| 2 | 7 | 0.319 | 0.628 | 0.241 | -0.27 | +0.47 |
| 3 | 9 | 0.650 | 0.878 | 0.341 | +0.24 | -0.28 |

## yolo11n-esmoe-rewire-e120-s2-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2885 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.009 | 0.509 | 0.251 | +0.47 | -0.45 |
| 1 | 5 | 0.797 | 0.901 | 0.361 | +0.20 | -0.21 |
| 2 | 7 | 0.119 | 0.363 | 0.227 | -0.12 | +0.29 |
| 3 | 9 | 0.075 | 0.226 | 0.162 | -0.32 | +0.21 |

## yolo12n-esmoe-e120-s0-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2869 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.703 | 0.914 | 0.348 | +0.12 | -0.08 |
| 1 | 5 | 0.100 | 0.297 | 0.221 | -0.23 | +0.29 |
| 2 | 7 | 0.089 | 0.133 | 0.149 | -0.24 | +0.24 |
| 3 | 9 | 0.108 | 0.655 | 0.282 | +0.39 | -0.48 |

## yolo12n-esmoe-e120-s1-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.334 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.013 | 0.108 | 0.215 | +0.07 | -0.11 |
| 1 | 5 | 0.564 | 0.814 | 0.298 | -0.49 | +0.60 |
| 2 | 7 | 0.042 | 0.161 | 0.184 | +0.38 | -0.50 |
| 3 | 9 | 0.381 | 0.918 | 0.302 | +0.26 | -0.29 |

## yolo12n-esmoe-e120-s2-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2781 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.011 | 0.427 | 0.224 | +0.07 | -0.06 |
| 1 | 5 | 0.907 | 0.953 | 0.405 | -0.15 | +0.15 |
| 2 | 7 | 0.038 | 0.529 | 0.227 | +0.59 | -0.42 |
| 3 | 9 | 0.044 | 0.091 | 0.143 | -0.15 | +0.07 |

## yolo12n-esmoe-rewire-e120-s0-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3312 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.002 | 0.571 | 0.236 | +0.08 | +0.04 |
| 1 | 5 | 0.064 | 0.177 | 0.210 | +0.33 | -0.29 |
| 2 | 7 | 0.119 | 0.359 | 0.210 | -0.28 | +0.33 |
| 3 | 9 | 0.816 | 0.892 | 0.344 | -0.05 | -0.04 |

## yolo12n-esmoe-rewire-e120-s1-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3062 of 1.3863, distinct top-2 pairs seen: 5 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.009 | 0.329 | 0.226 | +0.24 | -0.14 |
| 1 | 5 | 0.124 | 0.256 | 0.178 | -0.30 | +0.32 |
| 2 | 7 | 0.863 | 0.920 | 0.366 | +0.18 | -0.17 |
| 3 | 9 | 0.004 | 0.496 | 0.230 | +0.21 | -0.45 |

## yolo12n-esmoe-rewire-e120-s2-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2758 of 1.3863, distinct top-2 pairs seen: 5 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.790 | 0.920 | 0.364 | +0.14 | -0.10 |
| 1 | 5 | 0.055 | 0.204 | 0.206 | -0.17 | +0.30 |
| 2 | 7 | 0.099 | 0.126 | 0.145 | -0.20 | +0.15 |
| 3 | 9 | 0.057 | 0.750 | 0.285 | +0.33 | -0.43 |

## yolo26n-esmoe-e120-s0-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3339 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.009 | 0.336 | 0.230 | +0.17 | -0.08 |
| 1 | 5 | 0.031 | 0.213 | 0.206 | +0.57 | -0.39 |
| 2 | 7 | 0.237 | 0.553 | 0.228 | -0.29 | +0.37 |
| 3 | 9 | 0.723 | 0.898 | 0.336 | -0.01 | -0.16 |

## yolo26n-esmoe-e120-s1-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3069 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.084 | 0.655 | 0.278 | +0.32 | -0.42 |
| 1 | 5 | 0.374 | 0.520 | 0.272 | -0.18 | +0.34 |
| 2 | 7 | 0.051 | 0.104 | 0.145 | -0.26 | +0.27 |
| 3 | 9 | 0.491 | 0.721 | 0.305 | +0.24 | -0.31 |

## yolo26n-esmoe-e120-s2-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3395 of 1.3863, distinct top-2 pairs seen: 5 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.188 | 0.222 | +0.11 | -0.06 |
| 1 | 5 | 0.166 | 0.613 | 0.234 | -0.34 | +0.44 |
| 2 | 7 | 0.015 | 0.285 | 0.206 | +0.46 | -0.50 |
| 3 | 9 | 0.819 | 0.914 | 0.339 | +0.04 | -0.13 |

## yolo26n-esmoe-rewire-e120-s0-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3129 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.035 | 0.491 | 0.239 | +0.15 | -0.22 |
| 1 | 5 | 0.064 | 0.126 | 0.171 | -0.20 | +0.15 |
| 2 | 7 | 0.027 | 0.443 | 0.225 | +0.51 | -0.56 |
| 3 | 9 | 0.874 | 0.940 | 0.366 | -0.10 | +0.21 |

## yolo26n-esmoe-rewire-e120-s1-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3065 of 1.3863, distinct top-2 pairs seen: 5 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.380 | 0.232 | +0.12 | -0.19 |
| 1 | 5 | 0.086 | 0.184 | 0.170 | -0.17 | +0.20 |
| 2 | 7 | 0.896 | 0.929 | 0.364 | +0.03 | +0.02 |
| 3 | 9 | 0.018 | 0.507 | 0.234 | +0.28 | -0.44 |

## yolo26n-esmoe-rewire-e120-s2-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3394 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.193 | 0.224 | +0.13 | -0.27 |
| 1 | 5 | 0.217 | 0.560 | 0.236 | -0.33 | +0.46 |
| 2 | 7 | 0.111 | 0.376 | 0.216 | +0.49 | -0.47 |
| 3 | 9 | 0.671 | 0.870 | 0.324 | -0.08 | -0.02 |

## yolov10n-esmoe-e120-s0-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3466 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.265 | 0.226 | -0.03 | -0.09 |
| 1 | 5 | 0.159 | 0.418 | 0.240 | -0.17 | +0.06 |
| 2 | 7 | 0.316 | 0.584 | 0.243 | -0.26 | +0.36 |
| 3 | 9 | 0.525 | 0.734 | 0.291 | +0.35 | -0.34 |

## yolov10n-esmoe-e120-s1-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3308 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.018 | 0.442 | 0.257 | +0.46 | -0.44 |
| 1 | 5 | 0.248 | 0.544 | 0.256 | -0.08 | +0.25 |
| 2 | 7 | 0.117 | 0.256 | 0.184 | -0.28 | +0.36 |
| 3 | 9 | 0.617 | 0.759 | 0.303 | +0.15 | -0.33 |

## yolov10n-esmoe-e120-s2-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.293 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.350 | 0.210 | -0.03 | -0.07 |
| 1 | 5 | 0.060 | 0.292 | 0.187 | -0.15 | +0.23 |
| 2 | 7 | 0.892 | 0.942 | 0.399 | +0.10 | -0.02 |
| 3 | 9 | 0.047 | 0.416 | 0.205 | +0.02 | -0.20 |

## yolov10n-esmoe-e120-s3-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3373 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.002 | 0.144 | 0.225 | +0.09 | -0.23 |
| 1 | 5 | 0.097 | 0.367 | 0.205 | -0.29 | +0.39 |
| 2 | 7 | 0.832 | 0.916 | 0.336 | +0.12 | -0.07 |
| 3 | 9 | 0.069 | 0.573 | 0.234 | +0.17 | -0.32 |

## yolov10n-esmoe-e120-s4-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2983 of 1.3863, distinct top-2 pairs seen: 5 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.235 | 0.225 | +0.30 | -0.46 |
| 1 | 5 | 0.847 | 0.925 | 0.369 | +0.04 | +0.10 |
| 2 | 7 | 0.110 | 0.270 | 0.169 | -0.26 | +0.27 |
| 3 | 9 | 0.044 | 0.569 | 0.237 | +0.28 | -0.47 |

## yolov10n-esmoe-norm-e120-s0-p800-best

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3316 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.024 | 0.339 | 0.246 | +0.16 | -0.09 |
| 1 | 5 | 0.829 | 0.914 | 0.326 | +0.09 | +0.07 |
| 2 | 7 | 0.080 | 0.571 | 0.250 | +0.29 | -0.40 |
| 3 | 9 | 0.068 | 0.175 | 0.178 | -0.30 | +0.19 |

## yolov10n-esmoe-rewire-e120-s0-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3054 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.002 | 0.429 | 0.218 | -0.09 | +0.12 |
| 1 | 5 | 0.088 | 0.394 | 0.209 | -0.23 | +0.37 |
| 2 | 7 | 0.066 | 0.250 | 0.190 | +0.10 | -0.24 |
| 3 | 9 | 0.845 | 0.927 | 0.383 | +0.12 | -0.13 |

## yolov10n-esmoe-rewire-e120-s1-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3155 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.020 | 0.571 | 0.230 | -0.12 | +0.09 |
| 1 | 5 | 0.079 | 0.217 | 0.191 | -0.18 | +0.27 |
| 2 | 7 | 0.816 | 0.894 | 0.369 | +0.17 | -0.06 |
| 3 | 9 | 0.086 | 0.318 | 0.210 | -0.00 | -0.28 |

## yolov10n-esmoe-rewire-e120-s2-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3121 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.025 | 0.153 | 0.215 | -0.01 | -0.12 |
| 1 | 5 | 0.089 | 0.531 | 0.214 | -0.23 | +0.31 |
| 2 | 7 | 0.841 | 0.916 | 0.368 | -0.09 | +0.19 |
| 3 | 9 | 0.044 | 0.400 | 0.202 | +0.47 | -0.62 |

## yolov10n-esmoe-rewire-e120-s3-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3006 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.002 | 0.438 | 0.222 | +0.23 | -0.25 |
| 1 | 5 | 0.093 | 0.292 | 0.181 | -0.24 | +0.29 |
| 2 | 7 | 0.890 | 0.923 | 0.380 | +0.04 | -0.03 |
| 3 | 9 | 0.015 | 0.347 | 0.217 | +0.30 | -0.43 |

## yolov10n-esmoe-rewire-e120-s4-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3337 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.213 | 0.230 | +0.09 | -0.33 |
| 1 | 5 | 0.755 | 0.909 | 0.332 | +0.11 | -0.08 |
| 2 | 7 | 0.203 | 0.460 | 0.214 | -0.25 | +0.39 |
| 3 | 9 | 0.042 | 0.418 | 0.224 | +0.24 | -0.45 |

## yolov10n-esmoe-rewire-stages-e120-s0-p800-best

548 images through block 0 (32 channels), kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.352 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.708 | 0.810 | 0.310 | -0.21 | +0.32 |
| 1 | 5 | 0.122 | 0.256 | 0.204 | +0.08 | -0.12 |
| 2 | 7 | 0.157 | 0.370 | 0.242 | +0.22 | -0.37 |
| 3 | 9 | 0.013 | 0.564 | 0.243 | -0.03 | +0.16 |

548 images through block 1 (64 channels), kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2674 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.416 | 0.487 | 0.244 | +0.18 | -0.16 |
| 1 | 5 | 0.566 | 0.611 | 0.260 | -0.28 | +0.28 |
| 2 | 7 | 0.002 | 0.259 | 0.248 | +0.36 | -0.41 |
| 3 | 9 | 0.016 | 0.642 | 0.248 | +0.36 | -0.41 |

548 images through block 2 (128 channels), kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3498 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.332 | 0.662 | 0.257 | +0.27 | -0.21 |
| 1 | 5 | 0.606 | 0.781 | 0.303 | -0.32 | +0.51 |
| 2 | 7 | 0.053 | 0.265 | 0.222 | +0.18 | -0.59 |
| 3 | 9 | 0.009 | 0.292 | 0.218 | -0.03 | -0.22 |

548 images through block 3 (256 channels), kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.317 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.276 | 0.211 | +0.05 | -0.20 |
| 1 | 5 | 0.057 | 0.330 | 0.196 | -0.21 | +0.47 |
| 2 | 7 | 0.161 | 0.489 | 0.231 | +0.28 | -0.25 |
| 3 | 9 | 0.783 | 0.905 | 0.362 | -0.10 | -0.05 |

## yolov10n-esmoe-rewire-stages-e120-s1-p800-best

548 images through block 0 (32 channels), kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3319 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.161 | 0.391 | 0.220 | +0.14 | -0.15 |
| 1 | 5 | 0.752 | 0.814 | 0.340 | -0.23 | +0.35 |
| 2 | 7 | 0.084 | 0.219 | 0.212 | +0.24 | -0.39 |
| 3 | 9 | 0.004 | 0.577 | 0.229 | +0.19 | -0.42 |

548 images through block 1 (64 channels), kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3559 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.661 | 0.734 | 0.290 | -0.10 | +0.38 |
| 1 | 5 | 0.279 | 0.504 | 0.234 | -0.03 | -0.08 |
| 2 | 7 | 0.057 | 0.454 | 0.238 | +0.25 | -0.58 |
| 3 | 9 | 0.004 | 0.308 | 0.238 | +0.25 | -0.57 |

548 images through block 2 (128 channels), kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3577 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.305 | 0.586 | 0.261 | +0.34 | -0.37 |
| 1 | 5 | 0.068 | 0.279 | 0.226 | +0.15 | -0.55 |
| 2 | 7 | 0.004 | 0.332 | 0.218 | -0.28 | +0.19 |
| 3 | 9 | 0.624 | 0.803 | 0.296 | -0.28 | +0.59 |

548 images through block 3 (256 channels), kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2896 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.164 | 0.195 | +0.06 | -0.23 |
| 1 | 5 | 0.073 | 0.531 | 0.218 | +0.09 | -0.16 |
| 2 | 7 | 0.905 | 0.954 | 0.405 | +0.01 | +0.10 |
| 3 | 9 | 0.022 | 0.350 | 0.181 | -0.21 | +0.17 |

## yolov10n-esmoe-rewire-stages-e120-s2-p800-best

548 images through block 0 (32 channels), kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3745 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.093 | 0.770 | 0.261 | -0.12 | +0.29 |
| 1 | 5 | 0.655 | 0.792 | 0.281 | -0.26 | +0.10 |
| 2 | 7 | 0.184 | 0.268 | 0.232 | +0.15 | -0.08 |
| 3 | 9 | 0.068 | 0.170 | 0.226 | +0.20 | -0.23 |

548 images through block 1 (64 channels), kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3824 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.699 | 0.863 | 0.271 | -0.25 | +0.54 |
| 1 | 5 | 0.031 | 0.620 | 0.246 | -0.21 | +0.21 |
| 2 | 7 | 0.013 | 0.122 | 0.239 | -0.03 | -0.08 |
| 3 | 9 | 0.257 | 0.394 | 0.243 | +0.31 | -0.52 |

548 images through block 2 (128 channels), kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3351 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.179 | 0.693 | 0.267 | -0.16 | +0.30 |
| 1 | 5 | 0.588 | 0.783 | 0.296 | -0.17 | +0.43 |
| 2 | 7 | 0.069 | 0.157 | 0.199 | -0.03 | -0.44 |
| 3 | 9 | 0.164 | 0.367 | 0.238 | +0.38 | -0.43 |

548 images through block 3 (256 channels), kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2818 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.411 | 0.201 | +0.25 | -0.30 |
| 1 | 5 | 0.053 | 0.237 | 0.203 | +0.07 | -0.16 |
| 2 | 7 | 0.046 | 0.400 | 0.182 | -0.23 | +0.40 |
| 3 | 9 | 0.901 | 0.953 | 0.414 | +0.04 | -0.07 |

## yolov5n-esmoe-dense-e120-s0-p800h2-best

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3124 of 1.3863, distinct top-2 pairs seen: 5 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.002 | 0.547 | 0.230 | +0.42 | -0.46 |
| 1 | 5 | 0.000 | 0.060 | 0.218 | +0.37 | -0.41 |
| 2 | 7 | 0.547 | 0.626 | 0.269 | -0.38 | +0.48 |
| 3 | 9 | 0.451 | 0.766 | 0.283 | +0.31 | -0.43 |

## yolov5n-esmoe-dense-e120-s1-p800h2-best

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2917 of 1.3863, distinct top-2 pairs seen: 5 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.161 | 0.203 | +0.01 | -0.03 |
| 1 | 5 | 0.027 | 0.436 | 0.212 | +0.10 | -0.14 |
| 2 | 7 | 0.847 | 0.916 | 0.388 | +0.18 | -0.32 |
| 3 | 9 | 0.126 | 0.487 | 0.197 | -0.31 | +0.52 |

## yolov5n-esmoe-dense-e120-s2-p800h2-best

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3096 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.004 | 0.666 | 0.244 | +0.14 | -0.30 |
| 1 | 5 | 0.075 | 0.182 | 0.191 | -0.10 | +0.07 |
| 2 | 7 | 0.080 | 0.263 | 0.208 | -0.12 | +0.38 |
| 3 | 9 | 0.841 | 0.889 | 0.356 | +0.13 | -0.22 |

## yolov5n-esmoe-e120-s0-p800h2-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2538 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.002 | 0.525 | 0.229 | +0.37 | -0.44 |
| 1 | 5 | 0.007 | 0.219 | 0.218 | +0.18 | -0.19 |
| 2 | 7 | 0.825 | 0.905 | 0.387 | +0.20 | -0.21 |
| 3 | 9 | 0.166 | 0.350 | 0.167 | -0.29 | +0.32 |

## yolov5n-esmoe-e120-s1-p800h2-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2969 of 1.3863, distinct top-2 pairs seen: 5 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.221 | 0.201 | -0.03 | +0.04 |
| 1 | 5 | 0.027 | 0.170 | 0.188 | +0.05 | -0.21 |
| 2 | 7 | 0.752 | 0.885 | 0.376 | +0.22 | -0.29 |
| 3 | 9 | 0.221 | 0.725 | 0.235 | -0.33 | +0.53 |

## yolov5n-esmoe-e120-s2-p800h2-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2445 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.734 | 0.243 | +0.22 | -0.31 |
| 1 | 5 | 0.005 | 0.126 | 0.224 | +0.14 | -0.14 |
| 2 | 7 | 0.137 | 0.228 | 0.146 | -0.20 | +0.27 |
| 3 | 9 | 0.858 | 0.912 | 0.387 | +0.16 | -0.22 |

## yolov5n-esmoe-master-e120-s0-p800h2-best

548 images, kernels [3, 5, 7, 9], top-2, dead experts: [0], mean entropy 1.344 of 1.3863, distinct top-2 pairs seen: 5 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.009 | 0.203 | -0.16 | -0.21 |
| 1 | 5 | 0.000 | 0.381 | 0.213 | -0.19 | -0.10 |
| 2 | 7 | 0.272 | 0.715 | 0.265 | +0.29 | -0.35 |
| 3 | 9 | 0.728 | 0.894 | 0.319 | -0.25 | +0.46 |

## yolov5n-esmoe-master-e120-s1-p800h2-best

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3597 of 1.3863, distinct top-2 pairs seen: 4 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.058 | 0.215 | +0.03 | -0.44 |
| 1 | 5 | 0.000 | 0.044 | 0.195 | -0.20 | -0.01 |
| 2 | 7 | 0.122 | 0.905 | 0.264 | +0.32 | -0.22 |
| 3 | 9 | 0.878 | 0.993 | 0.327 | -0.30 | +0.39 |

## yolov5n-esmoe-norm-e120-s0-p800h2-best

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3489 of 1.3863, distinct top-2 pairs seen: 5 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.035 | 0.175 | 0.228 | -0.06 | +0.05 |
| 1 | 5 | 0.009 | 0.173 | 0.207 | +0.08 | +0.07 |
| 2 | 7 | 0.892 | 0.960 | 0.332 | -0.12 | +0.19 |
| 3 | 9 | 0.064 | 0.692 | 0.234 | +0.21 | -0.41 |

## yolov5n-esmoe-norm-e120-s1-p800h2-best

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3193 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.376 | 0.221 | +0.04 | -0.03 |
| 1 | 5 | 0.036 | 0.451 | 0.226 | -0.01 | +0.01 |
| 2 | 7 | 0.057 | 0.235 | 0.186 | -0.25 | +0.28 |
| 3 | 9 | 0.907 | 0.938 | 0.367 | +0.17 | -0.20 |

## yolov5n-esmoe-norm-e120-s2-p800h2-best

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3152 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.414 | 0.212 | -0.10 | +0.05 |
| 1 | 5 | 0.029 | 0.244 | 0.203 | +0.10 | -0.15 |
| 2 | 7 | 0.069 | 0.405 | 0.205 | -0.17 | +0.38 |
| 3 | 9 | 0.901 | 0.936 | 0.380 | +0.09 | -0.20 |

## yolov5n-esmoe-rewire-e120-s0-p800h2-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3447 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.029 | 0.086 | 0.216 | -0.08 | +0.08 |
| 1 | 5 | 0.016 | 0.184 | 0.204 | -0.06 | +0.10 |
| 2 | 7 | 0.805 | 0.943 | 0.329 | -0.05 | +0.29 |
| 3 | 9 | 0.150 | 0.786 | 0.251 | +0.15 | -0.39 |

## yolov5n-esmoe-rewire-e120-s1-p800h2-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2799 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.208 | 0.197 | -0.07 | +0.05 |
| 1 | 5 | 0.029 | 0.464 | 0.203 | +0.02 | -0.09 |
| 2 | 7 | 0.921 | 0.945 | 0.417 | +0.15 | -0.19 |
| 3 | 9 | 0.049 | 0.383 | 0.183 | -0.24 | +0.37 |

## yolov5n-esmoe-rewire-e120-s2-p800h2-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3355 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.022 | 0.549 | 0.237 | -0.23 | +0.25 |
| 1 | 5 | 0.047 | 0.303 | 0.214 | +0.37 | -0.32 |
| 2 | 7 | 0.179 | 0.323 | 0.217 | -0.24 | +0.42 |
| 3 | 9 | 0.752 | 0.825 | 0.332 | +0.09 | -0.26 |

## yolov8n-esmoe-e120-s0-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3023 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.055 | 0.202 | +0.39 | -0.45 |
| 1 | 5 | 0.011 | 0.551 | 0.220 | +0.47 | -0.42 |
| 2 | 7 | 0.246 | 0.474 | 0.218 | -0.30 | +0.48 |
| 3 | 9 | 0.743 | 0.920 | 0.360 | +0.10 | -0.31 |

## yolov8n-esmoe-e120-s1-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2586 of 1.3863, distinct top-2 pairs seen: 5 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.213 | 0.187 | +0.11 | -0.23 |
| 1 | 5 | 0.011 | 0.381 | 0.191 | +0.12 | -0.14 |
| 2 | 7 | 0.920 | 0.943 | 0.437 | +0.11 | -0.14 |
| 3 | 9 | 0.069 | 0.462 | 0.185 | -0.27 | +0.38 |

## yolov8n-esmoe-e120-s2-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3032 of 1.3863, distinct top-2 pairs seen: 5 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.170 | 0.195 | +0.32 | -0.20 |
| 1 | 5 | 0.004 | 0.053 | 0.175 | +0.26 | -0.35 |
| 2 | 7 | 0.381 | 0.876 | 0.293 | -0.45 | +0.49 |
| 3 | 9 | 0.615 | 0.901 | 0.337 | +0.26 | -0.30 |

## yolov8n-esmoe-rewire-e120-s0-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2849 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.239 | 0.197 | -0.04 | -0.01 |
| 1 | 5 | 0.038 | 0.402 | 0.204 | +0.02 | -0.08 |
| 2 | 7 | 0.046 | 0.409 | 0.192 | -0.16 | +0.30 |
| 3 | 9 | 0.916 | 0.951 | 0.408 | +0.10 | -0.13 |

## yolov8n-esmoe-rewire-e120-s1-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3135 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.015 | 0.173 | 0.214 | -0.06 | +0.20 |
| 1 | 5 | 0.057 | 0.571 | 0.221 | -0.17 | +0.42 |
| 2 | 7 | 0.885 | 0.931 | 0.376 | -0.04 | -0.01 |
| 3 | 9 | 0.044 | 0.325 | 0.190 | +0.21 | -0.41 |

## yolov8n-esmoe-rewire-e120-s2-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.292 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.002 | 0.411 | 0.206 | +0.14 | -0.12 |
| 1 | 5 | 0.036 | 0.179 | 0.197 | +0.02 | -0.09 |
| 2 | 7 | 0.100 | 0.473 | 0.206 | -0.19 | +0.26 |
| 3 | 9 | 0.861 | 0.938 | 0.391 | +0.09 | -0.08 |

## yolov9t-esmoe-e120-s0-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2878 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.005 | 0.195 | 0.206 | +0.12 | -0.22 |
| 1 | 5 | 0.170 | 0.476 | 0.213 | -0.28 | +0.44 |
| 2 | 7 | 0.016 | 0.420 | 0.200 | +0.37 | -0.46 |
| 3 | 9 | 0.808 | 0.909 | 0.381 | +0.07 | -0.15 |

## yolov9t-esmoe-e120-s1-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.2605 of 1.3863, distinct top-2 pairs seen: 5 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.036 | 0.164 | +0.07 | -0.24 |
| 1 | 5 | 0.179 | 0.790 | 0.265 | -0.29 | +0.57 |
| 2 | 7 | 0.027 | 0.230 | 0.171 | +0.16 | -0.32 |
| 3 | 9 | 0.794 | 0.943 | 0.400 | +0.12 | -0.19 |

## yolov9t-esmoe-e120-s2-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.331 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.004 | 0.110 | 0.210 | -0.01 | -0.19 |
| 1 | 5 | 0.182 | 0.555 | 0.230 | -0.33 | +0.34 |
| 2 | 7 | 0.779 | 0.901 | 0.344 | +0.04 | +0.03 |
| 3 | 9 | 0.035 | 0.434 | 0.216 | +0.50 | -0.53 |

## yolov9t-esmoe-rewire-e120-s0-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3339 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.000 | 0.084 | 0.204 | +0.04 | -0.32 |
| 1 | 5 | 0.244 | 0.527 | 0.242 | -0.35 | +0.59 |
| 2 | 7 | 0.212 | 0.533 | 0.237 | +0.19 | -0.22 |
| 3 | 9 | 0.544 | 0.856 | 0.316 | +0.18 | -0.33 |

## yolov9t-esmoe-rewire-e120-s1-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3183 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.016 | 0.234 | 0.212 | -0.12 | +0.04 |
| 1 | 5 | 0.825 | 0.942 | 0.360 | -0.22 | +0.22 |
| 2 | 7 | 0.025 | 0.073 | 0.168 | -0.16 | +0.07 |
| 3 | 9 | 0.133 | 0.752 | 0.259 | +0.47 | -0.36 |

## yolov9t-esmoe-rewire-e120-s2-p800-best.pt

548 images, kernels [3, 5, 7, 9], top-2, dead experts: none, mean entropy 1.3481 of 1.3863, distinct top-2 pairs seen: 6 of 6.

| expert | kernel | top-1 share | top-2 share | mean prob | corr. size | corr. count |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 3 | 0.139 | 0.639 | 0.257 | +0.11 | -0.36 |
| 1 | 5 | 0.491 | 0.611 | 0.266 | -0.41 | +0.58 |
| 2 | 7 | 0.343 | 0.626 | 0.276 | +0.34 | -0.33 |
| 3 | 9 | 0.027 | 0.124 | 0.202 | +0.06 | -0.22 |

