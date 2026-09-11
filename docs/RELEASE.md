# 发布说明

当前版本 **0.1.5**。

## 新增

- **与上游 `ES_MOE` 的参数对齐。** 块现在接上游构造函数的全部参数：`out_channels`、`top_k=None`（等于用全部专家）、`sparse_inference`（上游的 `use_sparse_inference`）、`dynamic_threshold`（上游 0.4，本包默认 0.0 不剪——`results/` 里每条记录都是这样量出来的）。偶数核逐一降为奇数再按 `max_kernel_size` 截断，剪枝过的 checkpoint 因此装得回去；`num_experts` / `reduction` / `dynamic_threshold` / `max_kernel_size` 在构造时就按同样的边界校验。
- **四个平衡目标，默认仍是 Switch。** `switch_balance`（默认，与 0.1.4 和 `results/` 里的全部记录一致）、`gshard_balance`（对齐上游：读 top-k 掩码重归一后的门控）、`master_balance`（论文式 13，与 `gshard` 只差仿射 `(L−1)/E²`）、`gshard_probs_balance`（读原始概率，用来隔离「读哪个张量」这一个变量）。默认值由数据定：读门控的目标对没进 top-k 的专家梯度恒为零，实测 6 个 checkpoint 里 5 个出现死专家，Switch 在 66 个里一个没有（判读线第六轮）。命令行 `--balance {switch,gshard,master,gshard_probs}`。
- **多卡与 `compile=True` 同用。** 进程组里没被路由到的专家以零权重留在计算图中，DDP 不再依赖 `find_unused_parameters`。ultralytics 在 `compile=True` 时用的 `find_unused_parameters=False`、`static_graph=True` 已用两个 gloo 进程验过，块在 TorchDynamo 下能编译且与 eager 一致（`tests/test_distributed.py`）。单进程行为不变。
- **自定义平衡项与专家写进配置。** `graft(balance=fn, expert=cls)` 把自定义的函数与类写成 `模块:限定名`，训练器与每个 DDP 子进程重建模型时都从这个名字导入回同一个对象；按名导入不回来的（lambda、嵌套函数、`__main__` 里定义的）在嫁接时就拒绝。内置专家有了短名 `dw`，`esmoe.EXPERTS` 与 `esmoe.BALANCES` 并列。
- **改宽的块嫁接进官方 yaml。** `graft(out_channels=N)` 在块后写一个官方 `Index` 层，块把输出包成单元素列表交给它；官方 `parse_model` 读这一层声明的宽度，下游因此按真实宽度建。`N` 是字面值，不随 width 倍率缩放。
- **命令行补齐。** `esmoe graft` 新增 `--out-channels`、`--balance`、`--expert`、`--out-norm`、`--dense-training`。
- **上游的块布局与块内结构。** `graft(at="backbone_stages")` 在主干每个 stage 后各放一块，七代主干均得四块（stage 边界由下采样层反推）；`out_norm` 补上加权求和后的 `BatchNorm + SiLU`（论文式 2 的 `Norm`）；`dense_training` 让训练期跑满专家，未选的权重为 0 但归一化统计量继续更新。三者默认关着，以保 `results/` 里既有的运行可原样复现。
- **`scripts/blockspec.py`**：从任一 checkpoint 读回当时真正生效的块设置。**`scripts/backfill.py`**：补齐并复核记录里的配置 hash、数据集样本数、GPU-hours、产物校验和，`--settings` 按权重改写记录并把改动写进记录本身。**`scripts/queue.sh`** 入库，与 `scripts/train.py` 的运行命名由测试对表。
- `scripts/report.py` 的配对差值附 95% 置信区间；`scripts/routing.py` 逐块分析而非只看第一块。

## 修复

- **块设置进不了训练的那个模型。** 训练器照 `model.yaml` 重建模型，`YOLO(cfg)` 之后设到块上的目标函数与开关随那个被丢弃的实例一起消失，不报错也不留痕——`--balance`、`--out-norm`、`--dense-training` 因此全部失效，而记录照命令行写。现在 `graft()` / `equip()` 把设置写进配置，`ESMoE` 接受 options 映射并按名解析目标函数；`scripts/train.py` 记录的块配置从训练完的模型上读，请求与实际不符即在开跑前退出。**用 0.1.4 的 `equip()` + `configure()` 设过这些开关的，训练出来的是构造函数默认值，请按 `scripts/blockspec.py` 复核。**
- **`dynamic_threshold` 无法追踪。** 掩码原先用 `scatter_` 塞一个 Python 布尔量，追踪器没有对应的算子，`torch.jit.trace` 与建立在它之上的导出全部失败。改成张量运算，逐输入重算。
- **路由统计混入了 warmup 前向。** 在加速卡上 ultralytics 会在第一个真实批次前跑一次空前向，路由钩子把那一行也收了进去——549 行对 548 张图。CPU 上不触发，所以此前一直没暴露。
- **同臂并行训练互相截断配置。** 嫁接出的 `configs/*.yaml` 原先不含 seed，两条并行车道跑同一条臂时写同一个文件，一个把另一个正在读的截断，读的那个死在 `KeyError: 'backbone'`。
- **`report.py` 跨硬件折算。** 分组键不含硬件，同一配置在两台机器上的运行被当成重复样本折进同一格。现在硬件栈与块配置都进入键。
- **`buckets.py` 在部分加速卡上无法评测。** `val()` 在推理模式里融合 conv+bn，有的构建拒绝对 inference tensor 取 view。现在提前融合，数值不变。
- **路由器 logits 未钳位。** 混合精度下跑飞的 logit 到 softmax 已是 inf，整个门控变 NaN。现按上游做法夹到 `[-30, 30]` 后走 fp32 softmax。

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
