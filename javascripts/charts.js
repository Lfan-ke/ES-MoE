// Interactive view of the paired-effect data. Numbers come from data.js, which scripts/charts.py
// regenerates from results/*.json, so this file never carries a hard-coded result.
(function () {
  const ARMS = {
    "default graft": "#2f6f9f",
    rewire: "#c2662d",
  };
  const ORDER = ["yolov5n", "yolov8n", "yolov9t", "yolov10n", "yolo11n", "yolo12n", "yolo26n"];

  // The site is bilingual; the chart follows the page rather than staying English throughout.
  const ZH = (document.documentElement.lang || "").toLowerCase().startsWith("zh");
  const T = ZH
    ? { "default graft": "ES-MoE 默认", rewire: "ES-MoE 改接", axis: "配对差值", seed: "seed ",
        mean: "均值，共 ", wins: "胜", seeds: "逐 seed", png: "存为 PNG" }
    : { "default graft": "ES-MoE default", rewire: "ES-MoE rewired", axis: "paired delta", seed: "seed ",
        mean: "mean of ", wins: "wins", seeds: "seeds", png: "PNG" };
  const SYSTEM = window.matchMedia("(prefers-color-scheme: dark)");
  const SEEDS = 3; // the protocol's seed count; fewer means the arm is still running

  // Three states: explicit light, explicit dark, or whatever the system says.
  function dark() {
    const scheme =
      document.body.getAttribute("data-md-color-scheme") ||
      document.documentElement.getAttribute("data-md-color-scheme");
    if (scheme === "slate") return true;
    if (scheme === "default") return false;
    return SYSTEM.matches;
  }

  function palette() {
    return dark()
      ? {
          ink: "#c9ced8", faint: "#98a0af", rule: "#454b57", zero: "#7e8797",
          panel: "#22262e", edge: "#3b414c",
        }
      : {
          ink: "#3d4451", faint: "#78808f", rule: "#d5dae2", zero: "#8b93a3",
          panel: "#ffffff", edge: "#c9ced8",
        };
  }

  const fmt = (v) => (v >= 0 ? "+" : "") + Number(v).toFixed(4);

  function option(data, metric) {
    const c = palette();
    const rows = data.rows.filter((r) => r[metric]);
    const axis = ORDER.filter((b) => rows.some((r) => r.backbone === b));
    const arms = Object.keys(ARMS).filter((a) => rows.some((r) => r.arm === a));
    const at = (b) => axis.indexOf(b);

    const series = [];
    arms.forEach((arm, armIndex) => {
      const mine = rows.filter((r) => r.arm === arm);
      const nudge = (armIndex - (arms.length - 1) / 2) * 0.16;

      // Min-max whisker: with three seeds the spread usually dwarfs the mean.
      series.push({
        name: arm,
        type: "custom",
        silent: true,
        legendHoverLink: false,
        renderItem: function (params, api) {
          const row = mine[params.dataIndex];
          if (!row || row[metric].seeds.length < 2) return null;
          const values = row[metric].seeds;
          const x = api.coord([at(row.backbone) + nudge, 0])[0];
          const top = api.coord([0, Math.max.apply(null, values)])[1];
          const bottom = api.coord([0, Math.min.apply(null, values)])[1];
          return {
            type: "line",
            shape: { x1: x, y1: top, x2: x, y2: bottom },
            style: { stroke: ARMS[arm], lineWidth: 1.5, opacity: 0.35 },
          };
        },
        data: mine.map((r) => [at(r.backbone) + nudge, r[metric].mean]),
        z: 2,
      });

      series.push({
        name: arm,
        type: "scatter",
        symbolSize: 8,
        itemStyle: { color: ARMS[arm], opacity: 0.5, borderColor: c.panel, borderWidth: 1 },
        emphasis: { scale: 1.5, itemStyle: { opacity: 0.95 } },
        data: mine.flatMap((r) =>
          r[metric].seeds.map((v, i) => ({
            value: [at(r.backbone) + nudge + (i - (r[metric].seeds.length - 1) / 2) * 0.05, v],
            seed: i,
            arm: arm,
            backbone: r.backbone,
          })),
        ),
        z: 3,
      });

      series.push({
        name: arm,
        type: "line",
        symbol: "rect",
        symbolSize: [26, 4],
        lineStyle: { width: 1.5, type: "dotted", opacity: 0.5, color: ARMS[arm] },
        itemStyle: { color: ARMS[arm] },
        emphasis: { focus: "series", scale: 1.2 },
        // No mean below three seeds, so an unfinished cell cannot read as settled.
        data: axis.map((b) => {
          const row = mine.find((r) => r.backbone === b);
          return row && row[metric].seeds.length >= SEEDS ? row[metric].mean : null;
        }),
        z: 4,
      });
    });

    return {
      animationDuration: 500,
      animationEasing: "cubicOut",
      textStyle: { color: c.ink },
      color: arms.map((a) => ARMS[a]),
      grid: { left: 78, right: 26, top: 52, bottom: 46 },
      legend: {
        data: arms,
        formatter: (name) => T[name] || name,
        top: 8,
        itemGap: 24,
        icon: "roundRect",
        textStyle: { color: c.ink },
        inactiveColor: c.faint,
      },
      toolbox: {
        right: 8,
        top: 4,
        iconStyle: { borderColor: c.faint },
        emphasis: { iconStyle: { borderColor: c.ink } },
        feature: { saveAsImage: { title: T.png, pixelRatio: 2, backgroundColor: c.panel } },
      },
      tooltip: {
        trigger: "item",
        backgroundColor: c.panel,
        borderColor: c.edge,
        borderWidth: 1,
        padding: [8, 11],
        extraCssText: dark()
          ? "box-shadow:0 4px 18px rgba(0,0,0,.55)"
          : "box-shadow:0 4px 18px rgba(15,20,30,.14)",
        textStyle: { color: c.ink, fontSize: 12 },
        formatter: function (p) {
          const dot = '<span style="color:' + ARMS[p.seriesName] + '">●</span> ';
          const grey = '<span style="color:' + c.faint + '">';
          if (p.seriesType === "scatter") {
            const d = p.data;
            return (
              dot + "<b>" + d.backbone + "</b> " + (T[d.arm] || d.arm) +
              "<br>" + grey + T.seed + d.seed + "</span> <b>" + fmt(d.value[1]) + "</b>"
            );
          }
          const row = rows.find((r) => r.backbone === axis[p.dataIndex] && r.arm === p.seriesName);
          if (!row) return "";
          const cell = row[metric];
          return (
            dot + "<b>" + row.backbone + "</b> " + (T[row.arm] || row.arm) +
            "<br>" + grey + T.mean + cell.seeds.length + "</span> <b>" + fmt(cell.mean) +
            "</b><br>" + grey + T.wins + "</span> " + cell.wins + "/" + cell.seeds.length +
            "<br>" + grey + T.seeds + "</span> " + cell.seeds.map(fmt).join(", ")
          );
        },
      },
      xAxis: {
        type: "category",
        data: axis,
        boundaryGap: true,
        axisTick: { alignWithLabel: true, lineStyle: { color: c.rule } },
        axisLine: { lineStyle: { color: c.rule } },
        axisLabel: { color: c.ink, fontSize: 12 },
      },
      yAxis: {
        type: "value",
        name: ZH ? metric + " " + T.axis : "paired " + metric + " delta",
        nameLocation: "end",
        nameTextStyle: { color: c.faint, align: "left", padding: [0, 0, 6, -66] },
        axisLine: { show: false },
        axisLabel: { color: c.faint, formatter: fmt },
        splitLine: { lineStyle: { color: c.rule, type: [3, 4] } },
      },
      series: series.concat([
        {
          // Above zero the graft beat its same-seed baseline.
          type: "line",
          data: axis.map(() => 0),
          symbol: "none",
          lineStyle: { color: c.zero, width: 1 },
          silent: true,
          legendHoverLink: false,
          tooltip: { show: false },
          z: 1,
        },
      ]),
    };
  }

  function toggle(host, data, onPick) {
    const bar = document.createElement("div");
    bar.className = "esmoe-metric";
    data.metrics.forEach((metric, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = metric;
      button.className = index === 0 ? "is-on" : "";
      button.addEventListener("click", function () {
        bar.querySelectorAll("button").forEach((b) => b.classList.remove("is-on"));
        button.classList.add("is-on");
        onPick(metric);
      });
      bar.appendChild(button);
    });
    host.parentNode.insertBefore(bar, host);
  }

  function draw() {
    const host = document.getElementById("esmoe-effect");
    const data = window.ESMOE_EFFECT;
    if (!host || !window.echarts || !data || !data.rows || !data.rows.length) return;

    // SVG stays crisp at any zoom and avoids canvas sizing on a cold load.
    const existing = window.echarts.getInstanceByDom(host);
    const chart =
      existing && !existing.isDisposed()
        ? existing
        : window.echarts.init(host, null, { renderer: "svg" });

    let metric = data.metrics[0];
    const paint = () => chart.setOption(option(data, metric), true);
    paint();

    if (!host.dataset.wired) {
      host.dataset.wired = "1";
      toggle(host, data, function (picked) {
        metric = picked;
        paint();
      });
      // Init before the stylesheet lands measures zero; re-measure once layout settles.
      requestAnimationFrame(() => chart.resize());
      setTimeout(() => chart.resize(), 120);
      window.addEventListener("resize", () => chart.resize());
      new MutationObserver(paint).observe(document.body, {
        attributes: true,
        attributeFilter: ["data-md-color-scheme"],
      });
      // Still on the system setting: an OS theme change must repaint too.
      SYSTEM.addEventListener("change", paint);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", draw);
  } else {
    draw();
  }
  if (window.document$) window.document$.subscribe(draw); // Material's instant navigation
})();
