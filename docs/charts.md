# 效果图

七代主干矩阵里默认接法与改接两臂的逐 seed 配对差：横轴主干，纵轴同 seed 配对的差值，零线以上表示嫁接块在该 seed 上占优。

<figure class="es-fig" id="fig-1">
<figcaption><b>图 1</b><span>七代主干的配对差</span><em>散点为 seed，横条为均值，竖线为极差</em></figcaption>
<div id="esmoe-effect" class="esmoe-chart"></div>
<small>数据：<code>results/summary.md</code></small>
</figure>

横条是均值，至少三个 seed 才画（v10n 默认接法有五个 seed）；竖线是极差，散点是逐个 seed。上方可切 mAP50 与 mAP50-95，悬停看该格明细，右上角另存 PNG。

## 读法

- 看竖线，不只看横条。三个 seed 常横跨零线，均值为正也如此；判读线事先写明，三个 seed 撑不起显著性检验，3/3 全胜的符号检验 p 也只有 0.125。
- 均值走向是主要结论：默认接法在 SPPF 系末端（v5n、v8n、v9t）为正，末端换成注意力块后（v10n、11n）贴零，到区域注意力与 E2E 头（12n、26n）转负。
- 不满三个 seed 的臂不画均值，只留散点。
- 两臂参数量相同（YOLOv8n 上都是 3,327,330），差别只在接线。

## 分档

<figure class="es-fig" id="fig-2">
<figcaption><b>图 2</b><span>按目标大小拆开</span><em>COCO 面积分档的配对差，柱为三个 seed 的均值</em></figcaption>
<div class="esmoe-figure" data-figure="buckets-generations" data-arms="both"></div>
<small>数据：<code>results/buckets.md</code></small>
</figure>

分档没有一致的尺度利害：v8n 默认接法下大目标三个 seed 都变差（APl −0.0104），改接后翻正；26n 上受损的是小目标（APs −0.0045，0/3）。

## 对照臂

<figure class="es-fig es-fig--tall" id="fig-3">
<figcaption><b>图 3</b><span>对照臂</span><em>上游块内的设置、四块布局与平衡项，各对同 seed 基线</em></figcaption>
<div class="esmoe-figure" data-figure="alignment"></div>
<small>数据：<code>results/summary.md</code></small>
</figure>

这些臂比较的是同一主干上的块配置，放不进按代际排的图 1。输出归一化与训练期跑满专家在 v5n 上都是 3/3；四块布局在 v10n、v5n 上都是 0/3。逐轮判定见[实验](experiments.md)与[判读线](JUDGMENT.md)。

## 重算

    uv run python scripts/report.py   # 配对表
    uv run python scripts/charts.py   # 图与数据

数据由 `scripts/charts.py` 从 `results/*.json` 现算，写进 `docs/javascripts/data.js`，与[实验结果](results.md)、[判读线](JUDGMENT.md)同源。`docs/assets/` 下的 SVG 是同一份数据的静态版本，供 README 与 wiki 使用。
