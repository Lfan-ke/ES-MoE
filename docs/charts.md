# 效果图

横轴主干，纵轴同 seed 配对的差值。零线以上表示嫁接块在该 seed 上占优。

<div id="esmoe-effect" class="esmoe-chart"></div>

方块是三个 seed 的均值，竖线是极差，散点是逐个 seed。上方可切 mAP50 与 mAP50-95，悬停看该格明细，右上角另存 PNG。

## 读法

- 看竖线，不只看方块。三个 seed 常横跨零线，均值为正也如此。
- 均值走向是主要结论：默认接法的效应随主干代际单调衰减。
- 不满三个 seed 的臂不画均值，只留散点。
- 两臂参数量相同（3,327,330），差别只在接线。

## 重算

    uv run python scripts/report.py   # 配对表
    uv run python scripts/charts.py   # 图与数据

数据源同[实验结果](results.md)与[判读线](JUDGMENT.md)。同一脚本另写 `docs/assets/effect.svg`，供 README 与 wiki 用。
