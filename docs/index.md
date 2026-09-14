<div class="es-hero" markdown>
<div class="es-hero__eyebrow">4 选 2 专家路由</div>
<h1 class="es-hero__claim">给 Ultralytics YOLO 装上<em>稀疏专家混合</em></h1>
<p class="es-hero__lede">一个调用接进官方 ultralytics，路由的平衡项进入反向传播。站内每个数字都对应 <code>results/</code> 里的一条实验记录。</p>
<div class="es-router"><span></span><span></span><span></span><span></span></div>
<div class="es-router__label">每张图从 4 个专家里选 2 个</div>

<ul class="es-proof">
<li><strong>一个调用装上</strong><span><code>equip</code> 把块接进配置、重编号层引用、接好损失；YOLOv5 至 YOLO26 七代官方主干与 YOLO-Master 分支都能用。</span></li>
<li><strong>辅助损失进入反向传播</strong><span>训练表里 <code>esmoe_aux</code> 自成一列，由单元测试与 <code>scripts/verify.py</code> 的十项真训练检查断言。</span></li>
<li><strong>与 YOLO-Master 作用一致</strong><span>同一个 yolo-master-n、同一份协议：块在上游分支上加 +0.0104，在官方 ultralytics 加本包上加 +0.0095（FP32，各 3/3），两者之差落在等效档。</span></li>
</ul>
</div>

<div class="es-stats">
<div><strong>142</strong><span>次满协议训练，600 卡时</span></div>
<div><strong>7</strong><span>代官方主干，另有 yolo-master-n</span></div>
<div><strong>335</strong><span>项测试，含与上游、论文的数值对照</span></div>
<div><strong>11</strong><span>次先于结果提交的预登记</span></div>
</div>

<div class="es-cards">
<a href="tutorial/"><strong>教程</strong><span>安装、三个调用、嫁接与重编号、对照实验怎么做</span></a>
<a href="API/"><strong>API</strong><span>equip、graft、attach_aux_loss 与 ESMoE 的全部参数</span></a>
<a href="design/"><strong>ES-MoE 与 YOLO</strong><span>块比普通 YOLO 多了什么，训练与推理怎么算，与 YOLO-Master 逐项对照</span></a>
<a href="experiments/"><strong>实验</strong><span>数据集、协议、流程、八轮实验与同配置对照，每个结论配一张图</span></a>
<a href="charts/"><strong>效果图</strong><span>七代主干上逐 seed 的配对差</span></a>
<a href="JUDGMENT/"><strong>判读线</strong><span>每一轮先写判据与预测，再写判定</span></a>
</div>

[在 Colab 里打开快速上手](https://colab.research.google.com/github/Lfan-ke/ES-MoE/blob/main/notebooks/quickstart.ipynb)：安装、接入、训练，在日志里看到 `esmoe_aux` 列，全程在免费 GPU 上完成。

## 安装

    pip install esmoe

分发名、导入名与命令行名都是 `esmoe`。

## 使用

    import esmoe

    model = esmoe.equip("yolo11n.yaml", weight=0.01)   # 注册 + 嫁接 + 构建 + 接损失
    model.train(data="coco8.yaml", epochs=10)

分步调用、命令行与手写配置见[教程](tutorial.md)。

## 兼容性

| 主干 | 构建与前向 | 嫁接进配置 | 训练中的辅助损失 | 满协议运行 |
|:--:|:--:|:--:|:--:|:--:|
| YOLOv5 | 是 | 是 | 是 | 是 |
| YOLOv8 | 是 | 是 | 是 | 是 |
| YOLOv9 | 是 | 是 | 是 | 是 |
| YOLOv10 | 是 | 是 | 是 | 是 |
| YOLO11 | 是 | 是 | 是 | 是 |
| YOLO12 | 是 | 是 | 是 | 是 |
| YOLO26 | 是 | 是 | 是 | 是 |
| YOLO-Master（分支） | 是 | 是 | 是 | 否 |

由 `tests/test_ultralytics.py` 在 ultralytics 8.4.101 与最新发布版上验证（CI 矩阵），两者的 loss items 形态不同，均已处理。训练一列由四代主干上真实的 1-epoch VisDrone 训练与七代主干上 120 epoch 的协议运行共同支撑，日志里 `train/esmoe_aux` 均非零；每一行的嫁接与前向都在 CI 里跑。YOLO-Master 一行跑在该分支自带的 ultralytics 上：`scripts/fork_smoke.py` 嫁接进它的 `yolo-master-n.yaml`、真实训练一轮且 `esmoe_aux` 非零，它自己的 `ES_MOE` 配置与本块共存。分支上的同配置对照训练的是上游自己的块，本包的块在官方 ultralytics 上训练 `yolo-master-n`，所以最后一列为否。

## 默认配置

`ESMoE(num_experts=4, top_k=2)` 配 `attach_aux_loss(weight=0.01)`。在 25% 训练集、640 像素、20 epoch 的统一预算下，它胜过 2 专家、8 专家、top-1 与关掉辅助损失的变体；在全量 VisDrone 上以同样的 20 epoch、640 像素、三个 seed 确认：配对 3/3 胜，mAP50 +0.0021，YOLOv8n 上参数增加 10.4%。协议预算（800 像素、120 epoch）下，YOLOv8n 默认接法为 +0.0025、2/3。论证见[选型](SELECTION.md)。
