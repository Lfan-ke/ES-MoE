# Try it

Drop in an image and see what up to three models make of it. A model with ES-MoE blocks also lists, block by block, what each expert scored and which ones were chosen. Everything runs in your own browser.

<div id="esmoe-demo" data-models="../../models/" data-assets="../../assets/demo/"></div>

## Two groups

| Group | Model | What it is |
|:--:|:--:|:--:|
| COCO, 80 classes | YOLO11n, no blocks | Official weights, the 80 everyday classes |
| COCO, 80 classes | YOLO-Master-EsMoE-N | Upstream's released weights, four blocks of three experts all active, carried onto this package's blocks |
| VisDrone, 10 classes | Baseline, no blocks | Arm C of the same-configuration comparison |
| VisDrone, 10 classes | Four blocks, this package | Arm B |
| VisDrone, 10 classes | Four blocks, upstream recipe | Arm A |

The three VisDrone models know ten classes seen from a drone, so an everyday photograph turns up almost nothing there, and the COCO models the other way round. The protocol behind the three arms is on [Experiments](experiments.md).

## Reading the gate panel

A block's router scores the whole image, keeps the top k experts, and outputs their weighted sum. Each row is one expert, labelled with its kernel size; the highlighted rows are the ones this image selected. The number is the probability after the softmax, and an expert outside the top k contributes nothing to this image.

What the block is made of, and how training differs from inference, is on [ES-MoE and YOLO](design.md); the cell-by-cell verdicts are on [Judgment lines](JUDGMENT.md).

## Notes

- `scripts/demo_export.py` exports each model from a checkpoint on the `checkpoints` branch, carrying every block's router probabilities out as extra outputs and checking them against the PyTorch forward.
- The first run downloads the model, 10 to 12 MB each, and the browser caches it afterwards.
- WebGPU is used where the browser has it and WASM otherwise; the numbers agree either way.
- The Netron link under each panel opens that model's graph; Wetron reads local files, so download the model file beside it and drop it in.
- Sample images: `bus.jpg` and `zidane.jpg` come from the [ultralytics](https://github.com/ultralytics/ultralytics) repository (AGPL-3.0); the aerial one is [Buses is depot at Bishan, Singapore](https://commons.wikimedia.org/wiki/File:Buses_is_depot_at_Bishan,_Singapore_(Unsplash).jpg) from Wikimedia Commons (CC0).

