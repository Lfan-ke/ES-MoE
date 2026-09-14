# 实验结果

每一行对应 [`results/`](https://github.com/Lfan-ke/ES-MoE/tree/main/results) 里的一条实验记录，本页由 `scripts/report.py` 从这些记录生成。分组键是主干、块配置与预算三者，两个不同预算不会落进同一个均值。要看的是配对表而不是两组均值，实验口径与交互式图表见[实验](experiments.md)；逐格判定见[判读线](JUDGMENT.md)。

协议矩阵每一次运行的 `best.pt` 与完整训练参数存放在 [`checkpoints` 分支](https://github.com/Lfan-ke/ES-MoE/tree/checkpoints)（Git LFS，与主分支隔离，数量以该分支 README 为准）：分档与路由的每个数字都能从那里的检查点重算。

<figure class="es-fig" id="fig-1">
<figcaption><b>图 1</b><span>七代主干的配对差</span><em>下表各格的逐 seed 值与均值</em></figcaption>
<div class="esmoe-figure" data-figure="generations" data-metrics="both"></div>
<small>数据：<code>results/summary.md</code></small>
</figure>

--8<-- "results/summary.md"

## 同配置对照

同一个模型、同一份协议，在上游分支与官方 ultralytics 加本包上各训一遍，四臂按 seed 分卡：A 是上游分支跑它自己的 `yolo-master-n`（四个 `ES_MOE`），A0 是同一分支去掉四个块，B 是官方 8.4.101 加本包四个 `ESMoE` 并按 `recipe="upstream"` 训练，C 是官方去块。每个 seed 的四臂在同一张卡上，配对差都在卡内求。

指标由 `scripts/measure.py` 统一重测：按各次运行自己的配置重建模型、装回 `last.pt` 的 EMA 权重、同一套验证参数各测一遍，不让两个框架各自的验证口径混进比较；训练器当时测到的值保留在记录的 `metrics_by_trainer` 里。判读按[判读线](JUDGMENT.md)第七轮（混合精度）与第八轮（FP32）。

<figure class="es-fig" id="fig-2">
<figcaption><b>图 2</b><span>四个差与 95% 区间</span><em>大点为均值，小点为 seed，灰带为噪声底</em></figcaption>
<div class="esmoe-figure" data-figure="same-config-differences" data-metrics="both"></div>
<small>数据：<code>results/same_config.md</code></small>
</figure>

--8<-- "results/same_config.md"

## 面积分档

<div class="es-duo">
<figure class="es-fig" id="fig-3">
<figcaption><b>图 3</b><span>七代主干</span><em>面积分档的配对差</em></figcaption>
<div class="esmoe-figure" data-figure="buckets-generations" data-arms="both"></div>
</figure>
<figure class="es-fig" id="fig-4">
<figcaption><b>图 4</b><span>同配置对照</span><em>A − A0 与 B − C</em></figcaption>
<div class="esmoe-figure" data-figure="buckets-same-config"></div>
</figure>
</div>

--8<-- "results/buckets.md"

## 路由行为

<div class="es-duo">
<figure class="es-fig" id="fig-5">
<figcaption><b>图 5</b><span>分派集中度与配对差</span><em>大点为四块布局</em></figcaption>
<div class="esmoe-figure" data-figure="routing-scatter"></div>
</figure>
<figure class="es-fig" id="fig-6">
<figcaption><b>图 6</b><span>各平衡项下的死专家</span><em>按检查点计</em></figcaption>
<div class="esmoe-figure" data-figure="dead-experts"></div>
</figure>
</div>

--8<-- "results/routing.md"

## 平衡压力

各平衡目标在同一权重下加到路由 logits 上的梯度，按各检查点实际收敛到的平均路由概率量出，由 `scripts/pressure.py` 生成。

<figure class="es-fig" id="fig-7">
<figcaption><b>图 7</b><span>平衡项的梯度</span><em>同一权重下，各检查点的中位数</em></figcaption>
<div class="esmoe-figure" data-figure="pressure"></div>
<small>数据：<code>results/pressure.md</code></small>
</figure>

--8<-- "results/pressure.md"

## 发布核验

下表由 `scripts/closure.py` 从记录、git 历史与已发布的产物逐条推出，可重跑复核。

--8<-- "results/closure.md"
