# 发布说明

当前版本 **0.1.4**。

## 新增

- **支持 DDP。** ultralytics 用一份生成的文件启动多卡 worker，文件里只导入训练器所在的模块，worker 进程从未导入 esmoe，建模时报 `KeyError: 'ESMoE'`。现在 `attach_aux_loss` 让 `model.train()` 选用一个住在 `esmoe.trainer` 里的训练器子类：worker 一导入它就注册块、从环境变量恢复辅助损失权重、给 `BaseModel.loss` 打补丁。`scripts/verify.py` 新增两项检查：真实的 worker 文件在干净解释器里训一轮；两个 gloo 进程各算各的辅助项、路由器梯度经 all-reduce 后一致。边界：稀疏分发需要 `find_unused_parameters=True`，`compile=True` 会把它关掉，该组合不可用。
- **`graft(..., rewire=True)`。** 重编号只移动引用，不改变引用对象，因此 head 里按序号点名旧主干末层的分支（YOLOv8 的 P5 侧向 `[-1, 9] Concat`）插入之后仍读 SPPF，块只经自顶向下路径间接影响 P5。`rewire` 让所有下游改指块。默认关闭，保持既有记录可比；两种接法的同预算对照尚未完成。
- `scripts/train.py --patience`（默认 0，禁用早停）与 `sweep.sh` 的 `IMGSZ` / `PATIENCE` / `ARMS`，用于按仓库复现协议跑（imgsz 800、120 epoch）。
- `scripts/buckets.py`：COCO 式面积分档（maxDets 500）；`scripts/routing.py`：验证集上的专家使用分布与路由是否随目标尺度变化。
- 两项半精度测试：bf16 autocast 下辅助项有限非零、参数不被拖成半精度；半精度门控重归一后仍和为 1。

## 反馈与迭代

0.1.4 的多数条目来自课题社区的首轮反馈：评测口径按导师给出的 COCO 式 32²/96² 分档与 maxDets=500 落地（`scripts/buckets.py`，口径来源已在文档注明）；`--patience` 与 `IMGSZ` 是为对齐仓库复现协议（imgsz 800、120 epoch、patience 0）而加；半精度有限值测试对应「先检查 FP32/AMP 下损失与梯度是否一致有限」的要求。上游侧的反馈同样闭环：`OptimizedMOE` 追踪守卫的修复已被 YOLO-Master 合并（#241）。

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
