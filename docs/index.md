<div class="es-hero" markdown>
<div class="es-hero__eyebrow">4 选 2 专家路由</div>
<h1 class="es-hero__claim">给 Ultralytics YOLO 装上<em>稀疏专家混合</em></h1>
<p class="es-hero__lede">一行接入，路由的负载均衡损失进入反向传播。站内每个数字都对应 <code>results/</code> 里的一条实验记录。</p>
<div class="es-router"><span></span><span></span><span></span><span></span></div>
<div class="es-router__label">每张图从 4 个专家里选 2 个</div>

<ul class="es-proof">
<li><strong>一个调用装上</strong><span><code>equip</code> 把块接进配置、重编号 head、接好损失。</span></li>
<li><strong>辅助损失进入反向传播</strong><span><code>results.csv</code> 中 <code>esmoe_aux</code> 自成一列，由单测断言；配置里写了键不算。</span></li>
<li><strong>按仓库协议量过</strong><span>YOLOv8n、imgsz 800、120 epoch、三个 seed：mAP50 +0.0025，mAP50-95 +0.0004；大目标变差、小目标召回变好。YOLO11n 上不成立。</span></li>
</ul>
</div>

[在 Colab 里打开快速上手](https://colab.research.google.com/github/Lfan-ke/ES-MoE/blob/main/notebooks/quickstart.ipynb)：安装、接入、训练，在日志里看到 `esmoe_aux` 列，全程在免费 GPU 上完成。

## 安装

    pip install esmoe

分发名、导入名与命令行名都是 `esmoe`。

## 使用

    import esmoe

    model = esmoe.equip("yolo11n.yaml", weight=0.01)   # 注册 + 接入 + 构建 + 接损失
    model.train(data="coco8.yaml", epochs=10)

分步接入、命令行与手写配置见[教程](tutorial.md)。

## 兼容性

| 主干 | 构建与前向 | 配置接入 | 训练中的辅助损失 |
|:--:|:--:|:--:|:--:|
| YOLOv8 | 是 | 是 | 是 |
| YOLO11 | 是 | 是 | 是 |
| YOLO12 | 是 | 是 | 是 |
| YOLO26 | 是 | 是 | 是 |

由 `tests/test_ultralytics.py` 在 ultralytics 8.4.101 与 8.4.132 上验证，两者的 loss items 形态不同，均已处理；另有每代主干各一次真实 1-epoch 训练，日志中 `train/esmoe_aux` 非零。

## 出厂配置

`ESMoE(num_experts=4, top_k=2)` 配 `attach_aux_loss(weight=0.01)`。在统一预算下，它胜过 2 / 4 / 8 专家与 top-1 的各个变体，并在全量 VisDrone 上以三个 seed 确认：配对 3/3 胜，mAP50 +0.0021，参数增加 10.4%。论证见[选型](SELECTION.md)。

## 证据与边界

- [教程](tutorial.md)：从安装到一次站得住的对照实验
- [选型](SELECTION.md)：为什么是 4 专家 top-2 与 0.01 的权重
- [已知局限](limitations.md)：引用任何数字之前先读
- [基线与增量](BASELINE.md)：锁定的基线，以及什么算本轮新增
