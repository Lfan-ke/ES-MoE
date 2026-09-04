# 实验结果

每一行对应 [`results/`](https://github.com/Lfan-ke/ES-MoE/tree/main/results) 里的一条实验记录，本页由 `scripts/report.py` 从这些记录生成。分组键是主干、块配置与预算三者，两个不同预算不会落进同一个均值。要看的是配对表而不是两组均值，引用任何数字之前先读[已知局限](limitations.md)；逐格判定见[判读线](JUDGMENT.md)。

协议矩阵 36 轮的 `best.pt` 与完整训练参数存放在 [`checkpoints` 分支](https://github.com/Lfan-ke/ES-MoE/tree/checkpoints)（Git LFS，与主分支隔离）：分档与路由的每个数字都能从那里的 checkpoint 重算。

--8<-- "results/summary.md"

## 面积分档

--8<-- "results/buckets.md"

## 路由行为

--8<-- "results/routing.md"
