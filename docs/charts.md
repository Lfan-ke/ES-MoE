# 效应图

下图把 `results/` 里每条协议运行的配对结果画出来：横轴是主干，纵轴是同 seed 配对下的差值，零线以上表示该 seed 上嫁接块占优。方块是三个 seed 的均值，竖线是三个 seed 的极差，散点是逐个 seed 的取值。图上方可切换 mAP50 与 mAP50-95，悬停可看该格的均值、胜数与逐 seed 取值，右上角可另存为 PNG。

<div id="esmoe-effect" class="esmoe-chart"></div>

数据由 `scripts/charts.py` 从 `results/*.json` 重算并写进 `docs/javascripts/data.js`，与[实验结果](results.md)、[判读线](JUDGMENT.md)两页的表格同源，图表不会与表格对不上。

## 怎么读这张图

- **竖线比方块更值得看**。三个 seed 的极差常常横跨零线，即使均值为正；画上极差正是为了不让均值单独说话。判读线声明过：三个 seed 不支持显著性检验，即使 3/3 胜，符号检验 p = 0.125。
- **均值连线的走向是本项目的主要结论**：默认接法的效应随主干代际单调衰减，从 YOLOv5n 的正值一路降到 YOLO26n 的负值。
- **未跑满三个 seed 的臂不画均值**，只留散点，避免与已判定的格子混为一谈。
- **两臂的差距是接线造成的，不是主干造成的**。`rewire` 与默认臂参数量完全相同（3,327,330），差别只在消费者是否读到块的输出。

## 重算

    uv run python scripts/report.py    # 先重建配对表
    uv run python scripts/charts.py    # 再重画图与数据模块

同一份脚本还会写出 `docs/assets/effect.svg`。README 与 wiki 用的是这张静态图——GitHub 在那两处会去掉脚本，交互图渲染不出来。
