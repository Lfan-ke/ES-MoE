# 已知局限

`results/` 里这些证据的边界，写在这里。

## 基准范围

- 数字来自 VisDrone2019-DET、imgsz 640、从零训练（无预训练权重）：候选选型用 25% 子集，确认实验用全量训练集。**这不是 COCO 数字**，也不能读作对 ES-MoE-N 锚点（2.68M / 8.7 GFLOPs / 42.7 mAP，那是 COCO 指标）的复现。
- 短预算（从零训数十 epoch）离收敛很远。这里测到的差距只界定同预算下两个块的排序，不预测收敛后的差距。
- 单机单卡 RTX 4090 D，没有多卡运行，DDP 下辅助损失的行为未经验证。

## 方法范围

- `ESMoE` 保持通道数，映射为 `c1 -> c1`。上游 `ES_MOE` 还支持 `c1 -> c2`，该路径**刻意未复现**：官方 `parse_model` 对第三方模块假定 `c2 == ch[f]`。
- 通道在首次前向时推断。若模型在任何前向之前被 script、导出或 `state_dict` 加载，此时尚无专家权重可载入。
- `attach_aux_loss` 会 patch 任务模型的类，并把权重同时保存在进程作用域：trainer 会重建模型，EMA 副本又在所有 callback 之前生成。因此一个进程一次只训练一种辅助损失设置；在从未调用 `attach_aux_loss` 的进程里加载 checkpoint，训练时不带辅助项。
- 负载均衡项采用 Switch-Transformer 形式（`num_experts * sum(importance * load)`），**未**做幅度的 EMA 归一化，这一点与 YOLO-Master 的 mixture 控制器不同。

## 协议与评测口径

- 训练协议是 imgsz 640、20/50/100 epoch、batch 32、AMP 开启，`patience` 未逐条记录。仓库既有的复现协议是 imgsz 800、120 epoch、`patience=0`、完整 val 548 张，本轮数字**不在那个协议下**，只能用于同预算的相对比较与参数筛选。
- 指标由 ultralytics 自带评测器给出，`max_det` 取其默认值 300，不是 VisDrone 官方口径的 500，因此这里的数字与 VisDrone 官方榜单不可直接比较。
- 未做 small / medium / large 面积分档，全部结论只覆盖总体 mAP50 与 mAP50-95。

## 报告口径

- seed 逐个列出，并给出均值 ± 样本标准差。三个 seed 下标准差只是粗估，不主张任何显著性检验。全量数据上有一个 seed 几乎打平（mAP50 +0.0001），因此该效应方向一致，但**单次运行不可靠**。
- 除参数增加外，该块每轮墙钟约多 5%（全量单次运行 829 s 对 790 s，单卡 RTX 4090 D）。
