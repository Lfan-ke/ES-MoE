# 实验结果

每一行对应 [`results/`](https://github.com/Lfan-ke/ES-MoE/tree/main/results) 里的一条实验记录，本页由 `scripts/report.py` 从这些记录生成。分组键是主干、块配置与预算三者，两个不同预算不会落进同一个均值。要看的是配对表而不是两组均值，引用任何数字之前先读[已知局限](limitations.md)；逐格判定见[判读线](JUDGMENT.md)。

协议矩阵每一轮的 `best.pt` 与完整训练参数存放在 [`checkpoints` 分支](https://github.com/Lfan-ke/ES-MoE/tree/checkpoints)（Git LFS，与主分支隔离，数量以该分支 README 为准）：分档与路由的每个数字都能从那里的 checkpoint 重算。

--8<-- "results/summary.md"

## 与 YOLO-Master 同配置对照

同一个模型、同一份协议，在上游分支与官方 ultralytics 加本包上各训一遍，四臂按 seed 分卡：A 是上游分支跑它自己的 `yolo-master-n`（四个 `ES_MOE`），A0 是同一分支去掉四个块，B 是官方 8.4.101 加本包四个 `ESMoE` 并按 `recipe="upstream"` 训练，C 是官方去块。每个 seed 的四臂在同一张卡上，配对差都在卡内求。

指标由 `scripts/measure.py` 统一重测：按各 run 自己的配置重建模型、装回 `last.pt` 的 EMA 权重、同一套验证参数各测一遍，不让两个框架各自的验证口径混进比较；训练器当时测到的值保留在记录的 `metrics_by_trainer` 里。判读按[判读线](JUDGMENT.md)第七轮。

--8<-- "results/same_config.md"

## 面积分档

--8<-- "results/buckets.md"

## 路由行为

--8<-- "results/routing.md"

## 平衡项的实际压力

各目标在同一权重下加到路由 logits 上的梯度，按各 checkpoint 实际收敛到的平均路由概率量出，由 `scripts/pressure.py` 生成。

--8<-- "results/pressure.md"

## 交付核验

下表由 `scripts/closure.py` 从记录、git 历史与已发布的产物逐条推出，可重跑复核。

--8<-- "results/closure.md"
