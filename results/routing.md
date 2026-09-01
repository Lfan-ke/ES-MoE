# Router behaviour on VisDrone val

Per checkpoint: the share of images on which each expert is the top-1 choice, the share on which it is in the top-2, the mean routing probability, and the correlation of that probability with the mean object size and the object count of the image.

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

## Reading

- No expert is dead, but routing is dominated by one expert per checkpoint: it is the top-1 choice on 74%, 92% and 62% of images. Which expert dominates differs between seeds (kernel 9, 7, 9), so the specialisation is not a property of the kernel sizes.
- The smallest kernel (3) is never the top-1 choice in any seed and sits in the top-2 on 5–21% of images.
- Mean routing probabilities stay near uniform (entropy 91–94% of the maximum) while the realised top-k load is skewed. The Switch-Transformer term multiplies mean probability by load, and load sums to `top_k` by construction, so once mean probabilities are balanced the term is nearly blind to load skew; the logged `esmoe_aux ≈ 0.020` at weight 0.01 is the balanced-importance value `k = 2` almost exactly.
- Correlations between an expert's probability and the mean object size of the image are weak and change sign across seeds (expert 2: −0.30, +0.11, −0.45). The router has not learnt a scale-based assignment.
- Read together with the area buckets, the block behaves less like a mixture of receptive fields than like one dominant depthwise branch plus a rotating second one, which is consistent with a small, seed-dependent effect.
