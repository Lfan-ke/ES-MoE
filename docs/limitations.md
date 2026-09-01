# 已知局限

`results/` 里这些证据的边界，写在这里。

## 基准范围

- 数字来自 VisDrone2019-DET、imgsz 640、从零训练（无预训练权重）：候选选型用 25% 子集，确认实验用全量训练集。**这不是 COCO 数字**，也不能读作对 ES-MoE-N 锚点（2.68M / 8.7 GFLOPs / 42.7 mAP，那是 COCO 指标）的复现。
- 短预算（从零训数十 epoch）离收敛很远。这里测到的差距只界定同预算下两个块的排序，不预测收敛后的差距。
- 全部数字来自单机单卡。DDP 的机制已验证（`scripts/verify.py`：真实的 worker 文件在干净解释器里跑通；两个 gloo 进程各算各的辅助项、路由器梯度经 all-reduce 一致），但没有多卡训练出的精度数字。
- 稀疏分发会把某一批没路由到的专家留在图外，DDP 因此需要 `find_unused_parameters=True`；ultralytics 默认就这么建，但开 `compile=True` 时会关掉它，那种组合下不能用本块。

## 方法范围

- `ESMoE` 保持通道数，映射为 `c1 -> c1`。上游 `ES_MOE` 还支持 `c1 -> c2`，该路径**刻意未复现**：官方 `parse_model` 对第三方模块假定 `c2 == ch[f]`。
- 通道在首次前向时推断。若模型在任何前向之前被 script、导出或 `state_dict` 加载，此时尚无专家权重可载入。
- `attach_aux_loss` 会 patch 任务模型的类，并把权重同时保存在进程作用域：trainer 会重建模型，EMA 副本又在所有 callback 之前生成。因此一个进程一次只训练一种辅助损失设置；在从未调用 `attach_aux_loss` 的进程里加载 checkpoint，训练时不带辅助项。
- 负载均衡项采用 Switch-Transformer 形式（`num_experts * sum(importance * load)`），**未**做幅度的 EMA 归一化，这一点与 YOLO-Master 的 mixture 控制器不同。

## 协议与评测口径

- 早期记录（imgsz 640、20/50/100 epoch）不在仓库的复现协议下，只用于同预算的相对比较与参数筛选。协议合规的一组（imgsz 800、120 epoch、`patience=0`、完整 val 548 张）已补齐：YOLOv8n 配对 mAP50 +0.0025（2/3），mAP50-95 +0.0004；下结论以这一组为准。
- 训练记录里的指标由 ultralytics 自带评测器给出，`max_det` 取其默认值 300；`results/buckets.md` 另以 COCO 口径、maxDets=500 重评了协议合规的六个 checkpoint，两套数字不要混用，也都不能与 VisDrone 官方榜单直接比较。
- 面积分档（small < 32²、medium 32²–96²、large ≥ 96²，按原图 GT 框算）是本项目采用的 COCO 式定义，不是 VisDrone 官方定义。三个 seed 的配对结果：大目标一致变差（APl −0.0104、ARl −0.0129，0/3 胜），小目标召回一致变好（ARs +0.0026，3/3 胜）但不足以抬高 AP。

## 报告口径

- seed 逐个列出，并给出均值 ± 样本标准差。三个 seed 下标准差只是粗估，不主张任何显著性检验。全量数据上有一个 seed 几乎打平（mAP50 +0.0001），因此该效应方向一致，但**单次运行不可靠**。
- 除参数增加外，该块每轮墙钟在 imgsz 640 下约多 5%（829 s 对 790 s，RTX 4090 D），在 imgsz 800 下约多 9%（85 分钟对 78 分钟，RTX 4090）。
