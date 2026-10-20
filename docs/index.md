<div class="es-hero" markdown>
<div class="es-hero__eyebrow">稀疏专家混合</div>
<h1 class="es-hero__claim">一个<em>专家混合</em>块，装进你已有的模型</h1>
<p class="es-hero__lede">PyTorch、TensorFlow、Jax、Flax、Keras 各有原生实现，写法随各家习惯；Ultralytics YOLO 一个调用接进去。路由的平衡项真正进入反向传播，不是配置里的一个数。</p>
<div class="es-router"><span></span><span></span><span></span><span></span></div>
<div class="es-router__label">每张图从 4 个专家里选 2 个</div>

<ul class="es-proof">
<li><strong>装进任意模型</strong><span>通道在首次前向推断，<code>nn.Sequential</code> 里直接放得下；也可以按子模块名插进现成的模型，不碰它的源码。</span></li>
<li><strong>一个调用接进 YOLO</strong><span><code>equip</code> 把块接进配置、重编号层引用、接好损失；YOLOv5 至 YOLO26 七代官方主干与 YOLO-Master 分支都能用。</span></li>
<li><strong>五个框架，一套参数</strong><span><code>kernels</code> 定专家与感受野，<code>top_k</code> 定几选几；五处实现读参数的是同一段代码，换框架不用改写法。</span></li>
</ul>
</div>

<div class="es-stats">
<div><strong>5</strong><span>个框架，PyTorch 到 Keras</span></div>
<div><strong>7</strong><span>代 YOLO 主干，v5 到 26</span></div>
<div><strong>1</strong><span>个调用接进训练</span></div>
<div><strong>任意</strong><span>模型都能插块</span></div>
</div>

<div class="es-cards">
<a href="torch/"><strong>Torch 集成</strong><span>两个旋钮、辅助损失、往现成模型里插块</span></a>
<a href="tutorial/"><strong>YOLO 集成</strong><span>安装、三个调用、嫁接与重编号、对照实验怎么做</span></a>
<a href="tensorflow/"><strong>TensorFlow 集成</strong><span>进 tf.function、用 GradientTape 求梯度</span></a>
<a href="jax/"><strong>Jax 集成</strong><span>init 与 apply 的纯函数，配 jax.jit 与 jax.grad</span></a>
<a href="flax/"><strong>Flax 集成</strong><span>nnx.Module，配 nnx.jit 与 nnx.grad</span></a>
<a href="keras/"><strong>Keras3 集成</strong><span>一份实现跑在 tensorflow、jax、torch 上</span></a>
<a href="API/"><strong>API 文档</strong><span>ESMoE 的全部参数，以及各后端与集成的入口</span></a>
<a href="design/"><strong>设计</strong><span>块内部怎么算，包分几层，加一个后端要写什么</span></a>
<a href="experiments/"><strong>实验</strong><span>默认配置是怎么选出来的，每个结论配一张图</span></a>
<a href="charts/"><strong>效果图</strong><span>七代主干上逐 seed 的配对差</span></a>
<a href="JUDGMENT/"><strong>判读线</strong><span>每一轮先写判据与预测，再写判定</span></a>
</div>

[在 Colab 里打开快速上手](https://colab.research.google.com/github/Lfan-ke/ES-MoE/blob/main/notebooks/quickstart.ipynb)：安装、接入、训练，在日志里看到 `esmoe_aux` 列，全程在免费 GPU 上完成。

## 安装

    uv add esmoe            # 块本身，只依赖 torch
    uv add esmoe[yolo]      # 再加 Ultralytics 集成

`esmoe[ultralytics]` 是同一个 extra 的另一个名字；`esmoe[tensorflow]`、`esmoe[jax]`、`esmoe[flax]`、`esmoe[keras]` 各带一个后端。分发名、导入名与命令行名都是 `esmoe`。

## 使用

    import esmoe
    from torch import nn

    net = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), esmoe.ESMoE(kernels=[3, 5, 7], top_k=1))

接进 YOLO：

    model = esmoe.equip("yolo11n.yaml", weight=0.01)   # 注册 + 嫁接 + 构建 + 接损失
    model.train(data="coco8.yaml", epochs=10)

分步调用、命令行与手写配置见[在 YOLO 里](tutorial.md)，纯 torch 的用法见[在 torch 模型里](torch.md)。

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
