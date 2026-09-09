# 发布说明

当前版本 **0.1.5**。

## 新增

- **`gshard_balance`：上游 `ES_MOE` 优化的那个平衡项。** YOLO-Master 的 `ES_MOE` 用的是 GShard 式 `N · Σ usage²`（`balance_loss_coeff = 1.0`），本包默认是 Switch 式 `E · Σ p̄ᵢfᵢ`（`weight = 0.01`）。两者现在都可选：`ESMoE(..., balance=esmoe.gshard_balance)`，或 `scripts/train.py --balance gshard`。在一条真实路由记录上，上游那套对平均概率的平衡梯度是本包默认的 31 倍——这个差距比公式之别更能解释观察到的路由行为，对照实验的判据已先于结果登记在判读线页。

## 修复

- **路由统计混入了 warmup 前向。** 在加速卡上 ultralytics 会在第一个真实批次前跑一次空前向，路由钩子把那一行也收了进去——549 行对 548 张图。CPU 上不触发，所以此前一直没暴露。
- **同臂并行训练互相截断配置。** 嫁接出的 `configs/*.yaml` 原先不含 seed，两条并行车道跑同一条臂时写同一个文件，一个把另一个正在读的截断，读的那个死在 `KeyError: 'backbone'`。此前没遇到，只是因为并行的两条车道恰好总是不同臂。
- **`report.py` 跨硬件折算。** 分组键不含硬件，同一配置在两台机器上的运行被当成重复样本折进同一格。现在硬件栈进入键，换机重跑各成一组。
- **`buckets.py` 在部分加速卡上无法评测。** `val()` 在推理模式里融合 conv+bn，有的构建拒绝对 inference tensor 取 view。现在提前融合，数值不变。

## 反馈与迭代

0.1.4 的多数条目来自首轮使用反馈：评测口径改用 COCO 式 32²/96² 分档与 maxDets=500（`scripts/buckets.py`，口径来源已在文档注明）；`--patience` 与 `IMGSZ` 是为对齐仓库复现协议（imgsz 800、120 epoch、patience 0）而加；半精度有限值测试对应「先检查 FP32/AMP 下损失与梯度是否一致有限」的要求。上游侧的反馈同样闭环：`OptimizedMOE` 追踪守卫的修复已被 YOLO-Master 合并（#241）。

## 修复

- `scripts/report.py` 的分组键补进 imgsz。此前同 epoch 不同分辨率的记录会被平均进同一行，正是文档承诺不会发生的事。

## 此前的 0.1.3

- **导出的模型不再忽略路由。** 块会跳过门控为零的专家，而这是一个依赖数据的判断：追踪器只记录示例输入走过的那条路由，导出的图便对此后所有输入沿用同一批专家。在一个路由随输入变化的块上，用某个输入导出的 ONNX 与 PyTorch 在另一条路由的输入上相差 0.2，现在相差 1e-7。追踪期间块会跑满所有专家，运行时仍走捷径，导出之外没有变慢。

    用 0.1.0 至 0.1.2 导出过模型的，请重新导出。

## 新增

- `scripts/verify.py`：单测做不到的正确性检查——真实训练一轮并确认辅助项为正、`weight=0` 时损失表不变、checkpoint 往返、断点续训、多个块一起训练、`val` 与 `predict`，以及 ONNX 导出。
- 一项回归测试：导出一个路由随输入符号变化的块，把两条分支都与 PyTorch 对照。

## 更早的小版本

0.1.0 是首个版本，给出 `inject_esmoe`、`graft`、`attach_aux_loss`、`collect_aux_loss` 四个入口，并附选型与三 seed 证据。0.1.1 修好 `equip()` 不带 `out` 时把配置当字典交给 `YOLO()` 的问题，补上 Colab 快速上手。0.1.2 修好 8.4.13x 之后 `loss_names` 在 `on_train_start` 时为空、导致训练日志表头错位的问题，并把包从 `src/` 移到仓库根目录。

## 安装

    pip install esmoe
