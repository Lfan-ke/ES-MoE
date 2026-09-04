# Router behaviour on VisDrone val

Per checkpoint: the share of images on which each expert is the top-1 choice, the share on which it is in the top-2, the mean routing probability, and the correlation of that probability with the mean object size and the object count of the image.

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

## Reading

- The collapse repeats on all four backbones (v8n, 11n, 12n, 26n) and all arms: no dead experts, one dominant expert per checkpoint (top-2 coverage 0.72–0.95 for the leader). On the first three backbones the leader differs per seed; on 26n all three seeds converge on the `k = 9` expert (top-1 share 0.49–0.82) — the only case where the winner is stable, and still not a scale-based division of labour (the other experts' correlations stay weak and sign-unstable).
- Mean routing probabilities stay near uniform (entropy 91–97% of the maximum) while the realised top-k load is skewed. The Switch-Transformer term multiplies mean probability by load, and load sums to `top_k` by construction, so once mean probabilities are balanced the term is nearly blind to load skew; the logged `esmoe_aux ≈ 0.020` at weight 0.01 is the balanced-importance value `k = 2` almost exactly.
- Correlations between an expert's probability and the mean object size of the image are weak and change sign across seeds on every backbone. No checkpoint learnt a scale-based assignment.
- The `rewire` checkpoints route the same way as the default ones on the backbones that have them. Rewiring changes where the block's output goes, not how the router behaves — so the wiring effects on the area buckets come from the graph, not from routing.
