# esmoe

面向 Ultralytics YOLO 的即插即用 ES-MoE（专家稀疏混合）模块。它**装在官方 `ultralytics` 旁边**，而不是用一个 fork 顶替它；并且随包给出预算公平的证据与一条真正进入优化器的辅助损失。

[在 Colab 里打开快速上手](https://colab.research.google.com/github/Lfan-ke/ES-MoE/blob/main/notebooks/quickstart.ipynb)
- 安装、接入、训练，并在日志里看到 `esmoe_aux` 列，全程在免费 GPU 上完成。

## 安装

    pip install esmoe

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

由 `tests/test_ultralytics.py` 在 ultralytics 8.4.101 与 8.4.132 上验证（两者的 loss items 形态不同，均已处理），另有每代主干各一次真实 1-epoch 训练，日志中 `train/esmoe_aux` 非零。

## 出厂配置

`ESMoE(num_experts=4, top_k=2)` 配 `attach_aux_loss(weight=0.01)`。它在统一预算下胜过 2/4/8 专家与 top-1 变体，并在全量 VisDrone 上以三个 seed 确认：配对 3/3 胜，mAP50 +0.0021，参数 +10.4%。论证见[选型](SELECTION.md)。

## 证据与边界

- [教程](tutorial.md) - 从安装到一次站得住的对照实验
- [选型](SELECTION.md) - 为什么是 4 专家 top-2 与 0.01 的权重
- [已知局限](limitations.md) - 引用任何数字之前请先读
- [基线与增量](BASELINE.md) - 锁定的基线与"什么算本轮新增"
