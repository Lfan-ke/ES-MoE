# 试玩

上传一张图片，看最多三个模型在同一张图上的检测结果。带 ES-MoE 块的模型还会列出每个块里各专家得了多少分、哪几个被选中。推理在你自己的浏览器里完成。

<div id="esmoe-demo" data-models="../models/" data-assets="../assets/demo/"></div>

## 两组模型

| 组 | 模型 | 说明 |
|:--:|:--:|:--:|
| COCO 80 类 | YOLO11n，无块 | 官方权重，日常照片里的 80 类 |
| COCO 80 类 | YOLO-Master-EsMoE-N | 上游发布的权重，四块、每块三个专家全开，装在本包的块上 |
| VisDrone 10 类 | 无块基线 | 同配置对照的 C 臂 |
| VisDrone 10 类 | 四块，本包 | 同配置对照的 B 臂 |
| VisDrone 10 类 | 四块，上游配方 | 同配置对照的 A 臂 |

VisDrone 的三个模型只认无人机视角的 10 类，日常照片在这一组里基本检不到目标；COCO 组反之。三臂的训练协议与判定见[实验](experiments.md)。

## 门控面板怎么读

块的路由器对整张图打分，取前 k 个专家，输出是这几个专家的加权和。面板里每行是一个专家，前面标着它的卷积核大小；高亮的行是这张图上被选中的。分值是 softmax 之后的概率，没被选中的专家对这张图没有贡献。

块的结构、训练与推理的区别见[ES-MoE 与 YOLO](design.md)，逐格判定见[判读线](JUDGMENT.md)。

## 说明

- 模型由 `scripts/demo_export.py` 从 `checkpoints` 分支的权重导出成 ONNX，导出时把每个块的路由概率一并作为输出，并逐项核对与 PyTorch 前向一致。
- 首次运行要下载模型，每个约 10 到 12 MB，之后留在浏览器缓存里。
- 浏览器支持 WebGPU 时用它，否则回落到 WASM，两者数值一致。
- 每块面板底部的 Netron 与 Wetron 链接可以直接打开这份模型的结构图。
- 样例图：`bus.jpg`、`zidane.jpg` 取自 [ultralytics](https://github.com/ultralytics/ultralytics) 仓库（AGPL-3.0）；航拍那张取自 Wikimedia Commons 的 [Buses is depot at Bishan, Singapore](https://commons.wikimedia.org/wiki/File:Buses_is_depot_at_Bishan,_Singapore_(Unsplash).jpg)（CC0）。

