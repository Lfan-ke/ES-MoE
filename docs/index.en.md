<div class="es-hero" markdown>
<div class="es-hero__eyebrow">top-2 of 4 experts</div>
<h1 class="es-hero__claim">A <em>sparse mixture of experts</em> for Ultralytics YOLO</h1>
<p class="es-hero__lede">One call to add it to official ultralytics, a router loss that reaches <code>backward()</code>, and a run record behind every number on this site.</p>
<div class="es-router"><span></span><span></span><span></span><span></span></div>
<div class="es-router__label">router picks 2 of 4 per image</div>

<ul class="es-proof">
<li><strong>One call to add it</strong><span><code>equip</code> grafts the block into a config, renumbers layer references and wires the loss, on seven official generations from YOLOv5 to YOLO26 and on YOLO-Master's fork.</span></li>
<li><strong>The aux loss reaches backward()</strong><span>An <code>esmoe_aux</code> column in the training log, asserted by unit tests and by ten real-training checks in <code>scripts/verify.py</code>.</span></li>
<li><strong>Acts the same as YOLO-Master's block</strong><span>One yolo-master-n, one protocol: the block adds +0.0104 on the fork and +0.0095 on official ultralytics with this package (FP32, 3/3 each), and the gap between the two is equivalent.</span></li>
</ul>
</div>

<div class="es-stats" markdown>
<div><strong>142</strong><span>full-protocol runs, 600 card-hours</span></div>
<div><strong>8</strong><span>backbones run through the full protocol</span></div>
<div><strong>319</strong><span>tests, including bit-for-bit parity with upstream and the paper</span></div>
<div><strong>11</strong><span>pre-registrations committed before their results</span></div>
</div>

<div class="es-cards">
<a href="tutorial/"><strong>Tutorial</strong><span>install, the three calls, grafting and renumbering, a comparison you can defend</span></a>
<a href="API/"><strong>API</strong><span>every argument of equip, graft, attach_aux_loss and ESMoE</span></a>
<a href="design/"><strong>ES-MoE and YOLO</strong><span>what the block adds, how it computes in training and inference, and YOLO-Master item by item</span></a>
<a href="experiments/"><strong>Experiments</strong><span>dataset, protocol, pipeline, eight rounds and the same-configuration comparison, all interactive</span></a>
<a href="charts/"><strong>Effect chart</strong><span>per-seed paired deltas across seven generations</span></a>
<a href="JUDGMENT/"><strong>Judgment lines</strong><span>criteria and predictions first, verdicts after</span></a>
</div>

[Open the quick start in Colab](https://colab.research.google.com/github/Lfan-ke/ES-MoE/blob/main/notebooks/quickstart.ipynb): install, equip, train and watch the `esmoe_aux` column, all on a free GPU.

## Install

    pip install esmoe

The distribution, the import and the CLI are all `esmoe`.

## Use

    import esmoe

    model = esmoe.equip("yolo11n.yaml", weight=0.01)   # register + graft + build + wire
    model.train(data="coco8.yaml", epochs=10)

The separate steps, the CLI and the hand-written config are covered in the [tutorial](tutorial.md).

## Compatibility

| backbone | build + forward | grafted config | aux loss in training | protocol runs |
|:--:|:--:|:--:|:--:|:--:|
| YOLOv5 | yes | yes | yes | yes |
| YOLOv8 | yes | yes | yes | yes |
| YOLOv9 | yes | yes | yes | yes |
| YOLOv10 | yes | yes | yes | yes |
| YOLO11 | yes | yes | yes | yes |
| YOLO12 | yes | yes | yes | yes |
| YOLO26 | yes | yes | yes | yes |
| YOLO-Master (fork) | yes | yes | yes | no |

Verified by `tests/test_ultralytics.py` on ultralytics 8.4.101 and 8.4.132, which report loss items in two different shapes; both are handled. The training column is backed by real 1-epoch VisDrone runs on four generations and by the 120-epoch protocol runs on all seven, each logging a non-zero `train/esmoe_aux`; graft and forward run on every row in CI. The YOLO-Master row runs against the fork's vendored ultralytics: `scripts/fork_smoke.py` grafts their `yolo-master-n.yaml`, trains one epoch with a non-zero `esmoe_aux`, and builds their own `ES_MOE` config alongside ours. The same-configuration comparison trains upstream's own blocks on the fork and this package's blocks on official ultralytics, hence the last column.

## Shipped default

`ESMoE(num_experts=4, top_k=2)` with `attach_aux_loss(weight=0.01)`. Under one budget it beat the 2-, 4- and 8-expert and top-1 variants, and on the full VisDrone training set three seeds confirmed it: 3/3 paired wins, +0.0021 mAP50, for 10.4% more parameters. The argument is on [Selection](SELECTION.md).
