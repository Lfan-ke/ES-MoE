<div class="es-hero" markdown>
<div class="es-hero__eyebrow">4 选 2 专家路由</div>
<h1 class="es-hero__claim">装得上的 ES-MoE，<em>验得了</em>的证据。</h1>
<p class="es-hero__lede">面向 Ultralytics YOLO 的即插即用专家稀疏混合模块。它装在官方包旁边，而不是用 fork 顶替它；本站的每一处结论都对应仓库里的一条实验记录。</p>
<div class="es-router"><span></span><span></span><span></span><span></span></div>
<div class="es-router__label">每张图从 4 个专家里选 2 个</div>

<ul class="es-proof">
<li><strong>装在官方包旁边</strong><span><code>pip install esmoe</code> - 不 fork、不打补丁、可干净卸载。</span></li>
<li><strong>辅助损失真的进了 backward()</strong><span><code>results.csv</code> 里的 <code>esmoe_aux</code> 列，由单测断言，而不是配置里有个键。</span></li>
<li><strong>量过，又重量了一遍</strong><span>YOLOv8n 上 20 epoch 3/3 seed、+0.0021 mAP50，50 epoch +0.0013；YOLO11n 上则打平。证据说明它在哪儿有用。</span></li>
</ul>
</div>

[在 Colab 里打开快速上手](https://colab.research.google.com/github/Lfan-ke/ES-MoE/blob/main/notebooks/quickstart.ipynb)
- 安装、接入、训练，并在日志里看到 `esmoe_aux` 列，全程在免费 GPU 上完成。

## 安装

    pip install esmoe

分发名、import 名与命令行名统一都是 `esmoe`。

## 使用

    import esmoe

    model = esmoe.equip("yolo11n.yaml", weight=0.01)   # 注册 + 接入 + 构建 + 接损失
    model.train(data="coco8.yaml", epochs=10)

各步骤也可单独使用 - `inject_esmoe()`、`graft(base, out=..., at=...)`、`attach_aux_loss(model, weight=...)` - 也可以走命令行：

    esmoe graft yolo11n.yaml -o yolo11n-esmoe.yaml -e 4 -k 2 --at backbone_end

`attach_aux_loss` 会在 trainer 的损失表里加一项 `esmoe_aux`，于是这项非零、参与反向传播的辅助损失会出现在 `results.csv` 里，而不是只存在于配置中。

手写时，接入后的配置层就是一行：

    [-1, 1, ESMoE, [4, 2]]   # num_experts, top_k

模块保持通道数不变，并在首次前向时推断宽度 - 这正是官方 `parse_model` 无需打补丁就能为它定尺寸的原因。

## 兼容性

| 主干 | 构建与前向 | 配置接入 | 训练中的 aux loss |
|:--:|:--:|:--:|:--:|
| YOLOv8 | 是 | 是 | 是 |
| YOLO11 | 是 | 是 | 是 |
| YOLO12 | 是 | 是 | 是 |
| YOLO26 | yes | yes | yes |

由 `tests/test_ultralytics.py` 在 ultralytics 8.4.101 与 8.4.132 上验证（两者的 loss items 形态不同，均已处理），另有每代主干各一次真实 1-epoch 训练，日志中 `train/esmoe_aux` 非零。

## 出厂配置

`ESMoE(num_experts=4, top_k=2)` 配 `attach_aux_loss(weight=0.01)`。它在统一预算下胜过 2/4/8 专家与 top-1 变体，并在全量 VisDrone 上以三个 seed 确认：配对 3/3 胜，mAP50 +0.0021，参数 +10.4%。论证见[选型](SELECTION.md)。

## 证据与边界

- [教程](tutorial.md) - 从安装到一次站得住的对照实验
- [选型](SELECTION.md) - 为什么是 4 专家 top-2 与 0.01 的权重
- [已知局限](limitations.md) - 引用任何数字之前请先读
- [基线与增量](BASELINE.md) - 锁定的基线与"什么算本轮新增"
