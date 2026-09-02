<div class="es-hero" markdown>
<div class="es-hero__eyebrow">top-2 of 4 experts</div>
<h1 class="es-hero__claim">A <em>sparse mixture of experts</em> for Ultralytics YOLO</h1>
<p class="es-hero__lede">One call to add it, a router loss that reaches <code>backward()</code>, and a run record behind every number on this site.</p>
<div class="es-router"><span></span><span></span><span></span><span></span></div>
<div class="es-router__label">router picks 2 of 4 per image</div>

<ul class="es-proof">
<li><strong>One call to add it</strong><span><code>equip</code> grafts the block into a config, renumbers the head and wires the loss.</span></li>
<li><strong>The aux loss reaches backward()</strong><span>An <code>esmoe_aux</code> column in <code>results.csv</code>, asserted by unit tests rather than by a config key.</span></li>
<li><strong>Measured under the repository protocol</strong><span>YOLOv8n, imgsz 800, 120 epochs, three seeds: mAP50 +0.0025, mAP50-95 +0.0004; large objects lose, small-object recall gains. On YOLO11n it does not hold.</span></li>
</ul>
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

| backbone | build + forward | grafted config | aux loss in training |
|:--:|:--:|:--:|:--:|
| YOLOv8 | yes | yes | yes |
| YOLO11 | yes | yes | yes |
| YOLO12 | yes | yes | yes |
| YOLO26 | yes | yes | yes |
| YOLO-Master (fork) | yes | yes | yes |

Verified by `tests/test_ultralytics.py` on ultralytics 8.4.101 and 8.4.132, plus a real 1-epoch training run per generation logging a non-zero `train/esmoe_aux`. Graft and forward are also exercised on yolov5n, yolov9t and yolov10n in CI. The YOLO-Master row runs on the fork's vendored ultralytics: `scripts/fork_smoke.py` grafts their `yolo-master-n.yaml`, trains one epoch with a non-zero `esmoe_aux`, and builds their own `ES_MOE` config alongside ours.

## Evidence

- [Tutorial](tutorial.md): from install to a comparison you can defend.
- [Selection](SELECTION.md): which configuration ships as the default, and why.
- [Limitations](limitations.md): what the numbers do not say. Read this before quoting any of them.
- [Baseline and increment](BASELINE.md): locked references and what counts as new work.
