<div class="es-hero" markdown>
<div class="es-hero__eyebrow">top-2 of 4 experts</div>
<h1 class="es-hero__claim">A <em>sparse mixture of experts</em> for torch models and YOLO</h1>
<p class="es-hero__lede">Torch is all it needs, one call adds it to official ultralytics, and a second implementation on Keras 3 runs it on tensorflow and jax. Every number on this site has a run record behind it.</p>
<div class="es-router"><span></span><span></span><span></span><span></span></div>
<div class="es-router__label">router picks 2 of 4 per image</div>

<ul class="es-proof">
<li><strong>Drops into any torch model</strong><span>It sizes itself on the first forward pass, so an <code>nn.Sequential</code> takes it as written; <code>kernels</code> sets the experts and their receptive fields, <code>top_k</code> how many run.</span></li>
<li><strong>One call to add it to YOLO</strong><span><code>equip</code> grafts the block into a config, renumbers layer references and wires the loss, on seven official generations from YOLOv5 to YOLO26 and on YOLO-Master's fork.</span></li>
<li><strong>The aux loss reaches backward()</strong><span>An <code>esmoe_aux</code> column in the training log, asserted by unit tests and by ten real-training checks in <code>scripts/verify.py</code>.</span></li>
</ul>
</div>

<div class="es-stats">
<div><strong>142</strong><span>full-protocol runs, 600 card-hours</span></div>
<div><strong>7</strong><span>official generations, plus yolo-master-n</span></div>
<div><strong>409</strong><span>tests, including numerical parity with upstream and the paper</span></div>
<div><strong>3</strong><span>seeds or more, paired seed by seed</span></div>
</div>

<div class="es-cards">
<a href="torch/"><strong>Torch</strong><span>the two knobs, the auxiliary loss, grafting into a model that exists</span></a>
<a href="tutorial/"><strong>Ultralytics YOLO</strong><span>install, the three calls, grafting and renumbering, a comparison you can defend</span></a>
<a href="tensorflow/"><strong>TensorFlow</strong><span>inside tf.function, differentiated by a GradientTape</span></a>
<a href="jax/"><strong>Jax</strong><span>stateless_call under jax.jit and jax.grad</span></a>
<a href="keras/"><strong>Keras 3</strong><span>one implementation on tensorflow, jax and torch</span></a>
<a href="API/"><strong>API</strong><span>every argument of equip, graft, attach_aux_loss and ESMoE</span></a>
<a href="design/"><strong>ES-MoE and YOLO</strong><span>what the block adds, how it computes in training and inference, and YOLO-Master item by item</span></a>
<a href="experiments/"><strong>Experiments</strong><span>dataset, protocol, pipeline, eight rounds and the same-configuration comparison, a figure for each finding</span></a>
<a href="charts/"><strong>Effect chart</strong><span>per-seed paired deltas across seven generations</span></a>
<a href="JUDGMENT/"><strong>Judgment lines</strong><span>criteria and predictions first, verdicts after</span></a>
</div>

[Open the quick start in Colab](https://colab.research.google.com/github/Lfan-ke/ES-MoE/blob/main/notebooks/quickstart.ipynb): install, equip, train and watch the `esmoe_aux` column, all on a free GPU.

## Install

    uv add esmoe            # the block, for any torch model
    uv add esmoe[yolo]      # ... and the Ultralytics integration

`esmoe[ultralytics]` is that extra under its other name, `esmoe[keras]` adds Keras 3, and `esmoe[tensorflow]` or `esmoe[jax]` bring the backend with it. The distribution, the import and the command line are all `esmoe`.

## Use

    import esmoe
    from torch import nn

    net = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), esmoe.ESMoE(kernels=[3, 5, 7], top_k=1))

In YOLO:

    model = esmoe.equip("yolo11n.yaml", weight=0.01)   # register + graft + build + wire
    model.train(data="coco8.yaml", epochs=10)

The separate steps, the command line and the hand-written config are in [In Ultralytics YOLO](tutorial.md); the plain torch path is in [In a torch model](torch.md).

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

Verified by `tests/test_ultralytics.py` on ultralytics 8.4.101 and the latest release (the CI matrix), which report loss items in two different shapes; both are handled. The training column is backed by real 1-epoch VisDrone runs on four generations and by the 120-epoch protocol runs on all seven, each logging a non-zero `train/esmoe_aux`; graft and forward run on every row in CI. The YOLO-Master row runs against the fork's vendored ultralytics: `scripts/fork_smoke.py` grafts their `yolo-master-n.yaml`, trains one epoch with a non-zero `esmoe_aux`, and builds their own `ES_MOE` config alongside ours. The same-configuration comparison trains upstream's own blocks on the fork and trains `yolo-master-n` with this package's blocks on official ultralytics, hence the last column.

## Default configuration

`ESMoE(num_experts=4, top_k=2)` with `attach_aux_loss(weight=0.01)`. Under one budget of a 25% train split, 640 px and 20 epochs, it beat the 2-expert, 8-expert, top-1 and aux-off variants; on the full VisDrone training set, at the same 20 epochs and 640 px, three seeds confirmed it: 3/3 paired wins, +0.0021 mAP50, and 10.4% more parameters on YOLOv8n. Under the protocol budget (800 px, 120 epochs) the default wiring on YOLOv8n is +0.0025, 2/3. The argument is on [Selection](SELECTION.md).
