<div class="es-hero" markdown>
<div class="es-hero__eyebrow">top-2 of 4 experts</div>
<h1 class="es-hero__claim">ES-MoE you can <em>install</em>, with evidence you can check.</h1>
<p class="es-hero__lede">A sparse mixture of experts for Ultralytics YOLO. One call to add it, a router loss that really
trains, and a run record behind every number on this site.</p>
<div class="es-router"><span></span><span></span><span></span><span></span></div>
<div class="es-router__label">router picks 2 of 4 per image</div>

<ul class="es-proof">
<li><strong>One call to add it</strong><span><code>equip</code> grafts the block into a config, renumbers the head and wires the loss.</span></li>
<li><strong>The aux loss reaches backward()</strong><span>An <code>esmoe_aux</code> column in <code>results.csv</code>, asserted by unit tests rather than by a config key.</span></li>
<li><strong>Measured, then re-measured</strong><span>On YOLOv8n: 3/3 seeds, +0.0021 mAP50 at 20 epochs, +0.0013 at 50. On YOLO11n: a wash. The evidence says where it works.</span></li>
</ul>
</div>

[Open the quick start in Colab](https://colab.research.google.com/github/Lfan-ke/ES-MoE/blob/main/notebooks/quickstart.ipynb)
- install, equip, train and watch the `esmoe_aux` column, all on a free GPU.

## Install

    pip install esmoe

The distribution, the import and the CLI are all `esmoe`.

## Use

    import esmoe

    model = esmoe.equip("yolo11n.yaml", weight=0.01)   # register + graft + build + wire
    model.train(data="coco8.yaml", epochs=10)

The steps are also available on their own - `inject_esmoe()`, `graft(base, out=..., at=...)`,
`attach_aux_loss(model, weight=...)` - and from the shell:

    esmoe graft yolo11n.yaml -o yolo11n-esmoe.yaml -e 4 -k 2 --at backbone_end

`attach_aux_loss` adds an `esmoe_aux` column to the trainer's loss table, so the auxiliary term is
visible in `results.csv` as a back-propagated number rather than a configuration key.

## Compatibility

| backbone | build + forward | grafted config | aux loss in training |
|:--:|:--:|:--:|:--:|
| YOLOv8 | yes | yes | yes |
| YOLO11 | yes | yes | yes |
| YOLO12 | yes | yes | yes |
| YOLO26 | yes | yes | yes |

Verified by `tests/test_ultralytics.py` on ultralytics 8.4.101 and 8.4.132, plus a real 1-epoch
training run per generation logging a non-zero `train/esmoe_aux`.

## Evidence

- [Selection](SELECTION.md) - which configuration ships as the default and why.
- [Limitations](limitations.md) - what the numbers do not say. Read this before quoting any of them.
- [Baseline and increment](BASELINE.md) - locked references and what counts as new work.
- [Mid-term report](MIDTERM.md) - question, budget, evidence, open points.
