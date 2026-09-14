// Figures for the experiments pages. Every number comes from data.js, which scripts/charts.py
// regenerates from results/, so this file only decides how the numbers are drawn.
(function () {
  const ZH = (document.documentElement.lang || "").toLowerCase().startsWith("zh");
  const SYSTEM = window.matchMedia("(prefers-color-scheme: dark)");

  // Hue tells the framework apart, depth tells whether the block is in: upstream fork is amber,
  // official ultralytics + esmoe is teal, and the arm without the block is the lighter shade.
  const ARM = { A: "#B26A00", A0: "#E8A33D", B: "#007F69", C: "#2BB598" };
  const SPLIT = { train: "#1f5f8b", val: "#e8a33d", test: "#7aa6c2" };
  const PRECISION = { mixed: "#3e7c8c", fp32: "#c2662d" };
  const BALANCE = { switch: "#007F69", gshard: "#B26A00", gshard_probs: "#E8A33D", master: "#7a5cc2", none: "#8b93a3" };

  const L = ZH
    ? {
        png: "存为 PNG", images: "图片", boxes: "标注框", perImage: "每张框数", split: { train: "训练集", val: "验证集", test: "测试集" },
        share: "占比", side: "缩放到 800 像素后的框边长（像素）", median: "中位边长", runs: "运行次数", hours: "卡时",
        mixed: "混合精度", fp32: "FP32", seed: "seed ", gap: "重复运行的 mAP50 差", mean: "均值", max: "最大",
        top1: "主导专家的 top-1 份额", delta: "配对 mAP50 差", checkpoints: "检查点", deadAny: "有死专家", deadNone: "无死专家",
        usage: "进入 top-2 的图片比例", block: "块", expert: "专家", spread: "路由概率的离散程度",
        grad: "每单位权重的梯度", positive: "为正", interval: "95% 置信区间",
        balance: { switch: "Switch（读概率）", gshard: "GShard（读门控）", gshard_probs: "GShard（读概率）", master: "论文式 13", none: "无平衡项" },
        names: { "A - B": "A − B 两个实现", "A - A0": "A − A0 上游加块", "B - C": "B − C 本包加块", "(A - A0) - (B - C)": "差之差" },
        arms: { A: "A 上游分支 + ES_MOE", A0: "A0 上游分支去块", B: "B 官方 + esmoe", C: "C 官方去块" },
        band: "混合精度噪声底 ±0.0045",
      }
    : {
        png: "PNG", images: "images", boxes: "boxes", perImage: "boxes per image", split: { train: "train", val: "val", test: "test" },
        share: "share", side: "box side after letterboxing to 800 px (px)", median: "median side", runs: "runs", hours: "card-hours",
        mixed: "mixed precision", fp32: "FP32", seed: "seed ", gap: "mAP50 gap between repeats", mean: "mean", max: "largest",
        top1: "leading expert's top-1 share", delta: "paired mAP50 delta", checkpoints: "checkpoints", deadAny: "with dead experts", deadNone: "no dead expert",
        usage: "share of images in the top-2", block: "block", expert: "expert", spread: "spread of the routing probabilities",
        grad: "gradient per unit weight", positive: "positive", interval: "95% CI",
        balance: { switch: "Switch (reads probs)", gshard: "GShard (reads gate)", gshard_probs: "GShard (reads probs)", master: "paper eq. 13", none: "no balance term" },
        names: { "A - B": "A − B implementations", "A - A0": "A − A0 block on the fork", "B - C": "B − C block with esmoe", "(A - A0) - (B - C)": "difference of differences" },
        arms: { A: "A fork + ES_MOE", A0: "A0 fork, no block", B: "B official + esmoe", C: "C official, no block" },
        band: "mixed-precision noise floor ±0.0045",
      };

  function dark() {
    const scheme = document.body.getAttribute("data-md-color-scheme");
    if (scheme === "slate") return true;
    if (scheme === "default") return false;
    return SYSTEM.matches;
  }

  function theme() {
    return dark()
      ? { ink: "#d4d9e2", faint: "#98a0af", rule: "#3b4250", panel: "#1b2231", edge: "#3b414c", zero: "#7e8797" }
      : { ink: "#27303f", faint: "#6f7787", rule: "#dde2ea", panel: "#ffffff", edge: "#c9ced8", zero: "#8b93a3" };
  }

  const signed = (v) => (v >= 0 ? "+" : "") + Number(v).toFixed(4);
  const thousands = (v) => Number(v).toLocaleString(ZH ? "zh-CN" : "en-US");

  function frame(c, extra) {
    return Object.assign(
      {
        animationDuration: 600,
        animationEasing: "cubicOut",
        textStyle: { color: c.ink, fontFamily: "inherit" },
        grid: { left: 64, right: 28, top: 56, bottom: 48, containLabel: true },
        legend: { top: 6, icon: "roundRect", itemGap: 18, textStyle: { color: c.ink }, inactiveColor: c.faint },
        toolbox: {
          right: 4, top: 2, iconStyle: { borderColor: c.faint },
          feature: { saveAsImage: { title: L.png, pixelRatio: 2, backgroundColor: c.panel } },
        },
        tooltip: {
          backgroundColor: c.panel, borderColor: c.edge, borderWidth: 1, padding: [8, 11],
          textStyle: { color: c.ink, fontSize: 12 },
          extraCssText: "box-shadow:0 6px 22px rgba(10,16,28,.18);border-radius:8px",
        },
      },
      extra,
    );
  }

  function axis(c, extra) {
    return Object.assign(
      {
        axisLine: { lineStyle: { color: c.rule } },
        axisTick: { lineStyle: { color: c.rule } },
        axisLabel: { color: c.faint },
        nameTextStyle: { color: c.faint },
        splitLine: { lineStyle: { color: c.rule, type: [3, 4] } },
      },
      extra,
    );
  }

  const SPLITS = ["train", "val", "test"];

  const FIGURES = {
    "dataset-splits": function (d, c) {
      const s = d.dataset.splits;
      return frame(c, {
        tooltip: Object.assign(frame(c).tooltip, {
          trigger: "axis",
          formatter: (items) => {
            const f = s[SPLITS[items[0].dataIndex]];
            return "<b>" + L.split[SPLITS[items[0].dataIndex]] + "</b><br>" + L.images + " " + thousands(f.images) +
              "<br>" + L.boxes + " " + thousands(f.boxes) + "<br>" + L.perImage + " " + (f.boxes / f.images).toFixed(1) +
              "<br>" + (f.bytes / 1e9).toFixed(2) + " GB";
          },
        }),
        legend: Object.assign(frame(c).legend, { data: [L.images, L.boxes] }),
        xAxis: axis(c, { type: "category", data: SPLITS.map((k) => L.split[k]), splitLine: { show: false }, axisLabel: { color: c.ink } }),
        yAxis: [
          axis(c, { type: "value", name: L.images }),
          axis(c, { type: "value", name: L.boxes, splitLine: { show: false } }),
        ],
        series: [
          { name: L.images, type: "bar", barWidth: 26, itemStyle: { color: "#1f5f8b", borderRadius: [5, 5, 0, 0] },
            label: { show: true, position: "top", color: c.ink, formatter: (p) => thousands(p.value) },
            data: SPLITS.map((k) => s[k].images) },
          { name: L.boxes, type: "bar", yAxisIndex: 1, barWidth: 26, itemStyle: { color: "#e8a33d", borderRadius: [5, 5, 0, 0] },
            label: { show: true, position: "top", color: c.ink, formatter: (p) => thousands(p.value) },
            data: SPLITS.map((k) => s[k].boxes) },
        ],
      });
    },

    "dataset-classes": function (d, c) {
      const s = d.dataset.splits;
      const order = d.dataset.names.map((n, i) => i).sort((a, b) => s.train.classes[a] - s.train.classes[b]);
      return frame(c, {
        grid: { left: 118, right: 36, top: 44, bottom: 30, containLabel: false },
        tooltip: Object.assign(frame(c).tooltip, { trigger: "axis", axisPointer: { type: "shadow" } }),
        legend: Object.assign(frame(c).legend, { data: SPLITS.map((k) => L.split[k]) }),
        xAxis: axis(c, { type: "value" }),
        yAxis: axis(c, { type: "category", data: order.map((i) => d.dataset.names[i]), axisLabel: { color: c.ink }, splitLine: { show: false } }),
        series: SPLITS.map((k) => ({
          name: L.split[k], type: "bar", stack: "all", barWidth: 16,
          itemStyle: { color: SPLIT[k] }, emphasis: { focus: "series" },
          data: order.map((i) => s[k].classes[i] || 0),
        })),
      });
    },

    "dataset-resolutions": function (d, c) {
      const s = d.dataset.splits;
      const totals = {};
      SPLITS.forEach((k) => s[k].resolutions.forEach(([w, h, n]) => { totals[w + "×" + h] = (totals[w + "×" + h] || 0) + n; }));
      const keys = Object.keys(totals).sort((a, b) => totals[b] - totals[a]);
      const count = (k, key) => (s[k].resolutions.find(([w, h]) => w + "×" + h === key) || [0, 0, 0])[2];
      return frame(c, {
        tooltip: Object.assign(frame(c).tooltip, { trigger: "axis", axisPointer: { type: "shadow" } }),
        legend: Object.assign(frame(c).legend, { data: SPLITS.map((k) => L.split[k]) }),
        grid: { left: 48, right: 24, top: 52, bottom: 64, containLabel: true },
        xAxis: axis(c, { type: "category", data: keys, axisLabel: { color: c.ink, rotate: 35 }, splitLine: { show: false } }),
        yAxis: axis(c, { type: "value", name: L.images }),
        series: SPLITS.map((k) => ({
          name: L.split[k], type: "bar", stack: "all", barMaxWidth: 34,
          itemStyle: { color: SPLIT[k] }, emphasis: { focus: "series" },
          data: keys.map((key) => count(k, key)),
        })),
      });
    },

    "dataset-sides": function (d, c) {
      const s = d.dataset.splits;
      const edges = s.train.side_at_800.edges;
      const labels = edges.slice(0, -1).map((e, i) => (i === edges.length - 2 ? "≥" + e : e + "–" + edges[i + 1]));
      return frame(c, {
        tooltip: Object.assign(frame(c).tooltip, {
          trigger: "axis", axisPointer: { type: "shadow" },
          valueFormatter: (v) => (v * 100).toFixed(1) + "%",
        }),
        legend: Object.assign(frame(c).legend, { data: SPLITS.map((k) => L.split[k] + "  " + L.median + " " + s[k].median_side.at_800 + " px") }),
        xAxis: axis(c, { type: "category", data: labels, name: L.side, nameLocation: "middle", nameGap: 32, axisLabel: { color: c.ink }, splitLine: { show: false } }),
        yAxis: axis(c, { type: "value", name: L.share, axisLabel: { color: c.faint, formatter: (v) => Math.round(v * 100) + "%" } }),
        series: SPLITS.map((k) => {
          const counts = s[k].side_at_800.counts;
          const total = counts.reduce((a, b) => a + b, 0);
          return {
            name: L.split[k] + "  " + L.median + " " + s[k].median_side.at_800 + " px", type: "bar", barGap: "8%",
            itemStyle: { color: SPLIT[k], borderRadius: [4, 4, 0, 0] },
            data: counts.map((n) => n / total),
          };
        }),
      });
    },

    "protocol-backbones": function (d, c) {
      const b = d.protocol.backbones;
      const keys = Object.keys(b).sort((x, y) => b[y].runs - b[x].runs);
      return frame(c, {
        tooltip: Object.assign(frame(c).tooltip, { trigger: "axis" }),
        legend: Object.assign(frame(c).legend, { data: [L.runs, L.hours] }),
        xAxis: axis(c, { type: "category", data: keys, axisLabel: { color: c.ink, rotate: 20 }, splitLine: { show: false } }),
        yAxis: [axis(c, { type: "value", name: L.runs }), axis(c, { type: "value", name: L.hours, splitLine: { show: false } })],
        series: [
          { name: L.runs, type: "bar", barMaxWidth: 30, itemStyle: { color: "#1f5f8b", borderRadius: [5, 5, 0, 0] },
            label: { show: true, position: "top", color: c.ink }, data: keys.map((k) => b[k].runs) },
          { name: L.hours, type: "line", yAxisIndex: 1, symbolSize: 7, itemStyle: { color: "#e8a33d" },
            lineStyle: { width: 2 }, data: keys.map((k) => Math.round(b[k].hours)) },
        ],
      });
    },

    "same-config-arms": function (d, c, host) {
      const metric = host.dataset.metric || "mAP50";
      const rows = d.same_config[metric].rows.slice().sort((a, b) =>
        a.precision === b.precision ? a.seed - b.seed : a.precision === "mixed" ? -1 : 1);
      const cats = rows.map((r) => (r.precision === "fp32" ? L.fp32 : L.mixed) + " · " + L.seed + r.seed);
      const values = rows.flatMap((r) => ["A", "A0", "B", "C"].map((a) => r[a]));
      const lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
      return frame(c, {
        tooltip: Object.assign(frame(c).tooltip, { trigger: "axis", valueFormatter: (v) => Number(v).toFixed(4) }),
        legend: Object.assign(frame(c).legend, { data: ["A", "A0", "B", "C"].map((a) => L.arms[a]) }),
        xAxis: axis(c, { type: "category", data: cats, axisLabel: { color: c.ink, interval: 0, rotate: 20 }, splitLine: { show: false } }),
        yAxis: axis(c, { type: "value", name: metric, min: +(lo - 0.004).toFixed(3), max: +(hi + 0.002).toFixed(3),
          axisLabel: { color: c.faint, formatter: (v) => v.toFixed(3) } }),
        series: ["A", "A0", "B", "C"].map((a) => ({
          name: L.arms[a], type: "line", symbol: "circle", symbolSize: 11, showSymbol: true,
          lineStyle: { width: 0 }, itemStyle: { color: ARM[a], borderColor: c.panel, borderWidth: 1.5 },
          emphasis: { scale: 1.4 }, data: rows.map((r) => r[a]),
        })),
      });
    },

    "same-config-differences": function (d, c, host) {
      const metric = host.dataset.metric || "mAP50";
      const diffs = d.same_config[metric].differences;
      const names = ["A - B", "A - A0", "B - C", "(A - A0) - (B - C)"];
      const families = ["mixed", "fp32"];
      const series = [];
      families.forEach((fam, fi) => {
        const nudge = (fi - 0.5) * 0.28;
        const mine = names.map((n) => diffs.find((x) => x.precision === fam && x.name === n));
        series.push({
          name: L[fam], type: "custom", renderItem: (params, api) => {
            const row = mine[params.dataIndex];
            if (!row || row.lo === null) return null;
            const y = api.coord([0, params.dataIndex + nudge])[1];
            const x1 = api.coord([row.lo, 0])[0], x2 = api.coord([row.hi, 0])[0];
            return { type: "group", children: [
              { type: "line", shape: { x1: x1, y1: y, x2: x2, y2: y }, style: { stroke: PRECISION[fam], lineWidth: 2.2, opacity: 0.75 } },
              { type: "line", shape: { x1: x1, y1: y - 5, x2: x1, y2: y + 5 }, style: { stroke: PRECISION[fam], lineWidth: 2 } },
              { type: "line", shape: { x1: x2, y1: y - 5, x2: x2, y2: y + 5 }, style: { stroke: PRECISION[fam], lineWidth: 2 } },
            ] };
          },
          data: mine.map((row, i) => [row ? row.mean : 0, i]), z: 2, tooltip: { show: false },
        });
        series.push({
          name: L[fam], type: "scatter", symbolSize: 13, itemStyle: { color: PRECISION[fam], borderColor: c.panel, borderWidth: 1.5 },
          data: mine.map((row, i) => ({ value: [row.mean, i + nudge], row: row })), z: 4,
        });
        series.push({
          name: L[fam], type: "scatter", symbolSize: 6, itemStyle: { color: PRECISION[fam], opacity: 0.45 },
          data: mine.flatMap((row, i) => row.values.map((v) => ({ value: [v, i + nudge], row: row }))), z: 3,
        });
      });
      series.push({
        type: "line", data: [], markArea: { silent: true, itemStyle: { color: c.zero, opacity: 0.12 },
          label: { show: true, position: "insideTop", color: c.faint, formatter: L.band }, data: [[{ xAxis: -0.0045 }, { xAxis: 0.0045 }]] },
        markLine: { silent: true, symbol: "none", lineStyle: { color: c.zero, type: "solid" }, label: { show: false }, data: [{ xAxis: 0 }] },
      });
      return frame(c, {
        color: [PRECISION.mixed, PRECISION.fp32],
        grid: { left: 170, right: 30, top: 50, bottom: 52, containLabel: false },
        legend: Object.assign(frame(c).legend, { data: [L.mixed, L.fp32] }),
        tooltip: Object.assign(frame(c).tooltip, {
          trigger: "item",
          formatter: (p) => {
            const r = p.data.row;
            return "<b>" + L.names[r.name] + "</b> · " + L[r.precision] + "<br>" + L.mean + " <b>" + signed(r.mean) + "</b><br>" +
              L.interval + " [" + signed(r.lo) + ", " + signed(r.hi) + "]<br>" + L.positive + " " + r.positive + "/" + r.values.length +
              "<br>" + r.values.map(signed).join(", ");
          },
        }),
        xAxis: axis(c, { type: "value", name: metric, nameLocation: "middle", nameGap: 28, axisLabel: { color: c.faint, formatter: signed } }),
        yAxis: axis(c, { type: "value", min: -0.6, max: 3.6, inverse: true, splitLine: { show: false },
          axisTick: { customValues: [0, 1, 2, 3], lineStyle: { color: c.rule } },
          axisLabel: { color: c.ink, customValues: [0, 1, 2, 3], formatter: (v) => L.names[names[Math.round(v)]] } }),
        series: series,
      });
    },

    "noise-floor": function (d, c) {
      const pairs = d.floor.slice().sort((a, b) => (a.precision > b.precision ? -1 : a.precision < b.precision ? 1 : b.gap - a.gap));
      const short = (p) => {
        const backbone = p.label.split(/-e\d|-baseline|@/)[0].replace("yolo-master-n", "master-n");
        const fork = p.label.indexOf("/yolo-master]") >= 0;
        const role = p.label.indexOf("baseline") >= 0 ? (fork ? "A0" : "C") : p.label.indexOf("gshard-norm-dense") >= 0 ? (fork ? "A" : "B") : "";
        return (role || backbone) + " " + L.seed + p.seed;
      };
      const stats = (fam) => {
        const g = pairs.filter((p) => p.precision === fam).map((p) => p.gap);
        return { mean: g.reduce((a, b) => a + b, 0) / g.length, max: Math.max.apply(null, g) };
      };
      return frame(c, {
        tooltip: Object.assign(frame(c).tooltip, {
          trigger: "item",
          formatter: (p) => { const q = pairs[p.dataIndex]; return "<b>" + short(q) + "</b> · " + L[q.precision] + "<br>" + q.first.toFixed(4) + " → " + q.repeat.toFixed(4) + "<br>" + L.gap + " <b>" + q.gap.toFixed(4) + "</b>"; },
        }),
        legend: Object.assign(frame(c).legend, {
          data: ["mixed", "fp32"].map((f) => L[f] + "  " + L.mean + " " + stats(f).mean.toFixed(4) + " · " + L.max + " " + stats(f).max.toFixed(4)),
        }),
        grid: { left: 56, right: 24, top: 56, bottom: 56, containLabel: true },
        xAxis: axis(c, { type: "category", data: pairs.map(short), axisLabel: { color: c.ink, rotate: 30 }, splitLine: { show: false } }),
        yAxis: axis(c, { type: "value", name: L.gap, axisLabel: { color: c.faint, formatter: (v) => v.toFixed(3) } }),
        series: ["mixed", "fp32"].map((fam) => ({
          name: L[fam] + "  " + L.mean + " " + stats(fam).mean.toFixed(4) + " · " + L.max + " " + stats(fam).max.toFixed(4),
          type: "bar", barMaxWidth: 26, stack: "gap", itemStyle: { color: PRECISION[fam], borderRadius: [4, 4, 0, 0] },
          data: pairs.map((p) => (p.precision === fam ? p.gap : null)),
          markLine: { symbol: "none", silent: true, lineStyle: { color: PRECISION[fam], type: "dashed" },
            label: { show: false }, data: [{ yAxis: +stats(fam).mean.toFixed(4) }] },
        })),
      });
    },

    "routing-scatter": function (d, c) {
      const pts = d.routing.points;
      const xs = pts.map((p) => p.top1), ys = pts.map((p) => p.delta);
      const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
      const mx = mean(xs), my = mean(ys);
      const cov = mean(xs.map((x, i) => (x - mx) * (ys[i] - my)));
      const sd = (a, m) => Math.sqrt(mean(a.map((v) => (v - m) * (v - m))));
      const r = cov / (sd(xs, mx) * sd(ys, my));
      const kinds = Object.keys(BALANCE).filter((k) => pts.some((p) => p.balance === k));
      return frame(c, {
        title: { text: "r = " + signed(r).slice(0, 6) + " · n = " + pts.length, right: 44, top: 4, textStyle: { color: c.faint, fontSize: 12, fontWeight: "normal" } },
        legend: Object.assign(frame(c).legend, { data: kinds.map((k) => L.balance[k]), left: 8 }),
        tooltip: Object.assign(frame(c).tooltip, {
          trigger: "item",
          formatter: (p) => { const q = p.data.q; return "<b>" + q.variant + "</b><br>" + L.top1 + " " + q.top1 + "<br>" + L.delta + " <b>" + signed(q.delta) + "</b><br>" + q.dead + " dead · " + q.blocks + " " + L.block; },
        }),
        xAxis: axis(c, { type: "value", name: L.top1, nameLocation: "middle", nameGap: 30, min: 0.4, max: 1 }),
        yAxis: axis(c, { type: "value", name: L.delta, axisLabel: { color: c.faint, formatter: signed } }),
        series: kinds.map((k) => ({
          name: L.balance[k], type: "scatter",
          symbol: "circle", itemStyle: { color: BALANCE[k], opacity: 0.78, borderColor: c.panel, borderWidth: 1 },
          emphasis: { scale: 1.5 },
          data: pts.filter((p) => p.balance === k).map((q) => ({ value: [q.top1, q.delta], symbolSize: q.blocks > 1 ? 15 : 9, q: q })),
          markLine: k === kinds[0] ? { silent: true, symbol: "none", lineStyle: { color: c.zero }, label: { show: false }, data: [{ yAxis: 0 }] } : undefined,
        })),
      });
    },

    "dead-experts": function (d, c) {
      const pts = d.routing.points;
      const kinds = Object.keys(BALANCE).filter((k) => pts.some((p) => p.balance === k));
      return frame(c, {
        tooltip: Object.assign(frame(c).tooltip, { trigger: "axis", axisPointer: { type: "shadow" } }),
        legend: Object.assign(frame(c).legend, { data: [L.deadNone, L.deadAny] }),
        xAxis: axis(c, { type: "category", data: kinds.map((k) => L.balance[k]), axisLabel: { color: c.ink, interval: 0 }, splitLine: { show: false } }),
        yAxis: axis(c, { type: "value", name: L.checkpoints }),
        series: [
          { name: L.deadNone, type: "bar", stack: "n", barMaxWidth: 46, itemStyle: { color: "#2BB598" },
            label: { show: true, color: "#fff" }, data: kinds.map((k) => pts.filter((p) => p.balance === k && !p.dead).length || null) },
          { name: L.deadAny, type: "bar", stack: "n", barMaxWidth: 46, itemStyle: { color: "#b5453b", borderRadius: [5, 5, 0, 0] },
            label: { show: true, color: "#fff" }, data: kinds.map((k) => pts.filter((p) => p.balance === k && p.dead).length || null) },
        ],
      });
    },

    "same-config-dead": function (d, c) {
      const arms = d.routing.same_config.slice().sort((a, b) => (a.precision === b.precision ? a.seed - b.seed : a.precision === "mixed" ? -1 : 1));
      const rows = arms.map((a) => L[a.precision] + " · " + L.seed + a.seed);
      const cols = [];
      arms[0].blocks.forEach((b, bi) => b.usage.forEach((u, ei) => cols.push(L.block + " " + bi + " · " + L.expert + " " + ei)));
      const cells = [];
      arms.forEach((a, ri) => {
        let ci = 0;
        a.blocks.forEach((b) => b.usage.forEach((u) => { cells.push([ci, ri, +u.toFixed(3)]); ci += 1; }));
      });
      return frame(c, {
        grid: { left: 120, right: 30, top: 30, bottom: 110, containLabel: false },
        legend: { show: false },
        tooltip: Object.assign(frame(c).tooltip, { trigger: "item", formatter: (p) => rows[p.value[1]] + "<br>" + cols[p.value[0]] + "<br>" + L.usage + " <b>" + p.value[2] + "</b>" }),
        xAxis: { type: "category", data: cols, axisLabel: { color: c.faint, rotate: 60, fontSize: 10 }, axisLine: { lineStyle: { color: c.rule } }, splitArea: { show: false } },
        yAxis: { type: "category", data: rows, axisLabel: { color: c.ink }, axisLine: { lineStyle: { color: c.rule } } },
        visualMap: { min: 0, max: 1, calculable: false, orient: "horizontal", left: "center", bottom: 4, itemWidth: 12, itemHeight: 160,
          text: ["1", "0"], textStyle: { color: c.faint }, inRange: { color: ["#b5453b", "#f3e3c4", "#007F69"] } },
        series: [{ type: "heatmap", data: cells, itemStyle: { borderColor: c.panel, borderWidth: 2 }, emphasis: { itemStyle: { borderColor: c.ink } } }],
      });
    },

    pressure: function (d, c) {
      const spreads = [0.15, 0.5, 1.5];
      const terms = [["switch", "switch"], ["gshard_probs", "gshard_probs"], ["gshard", "gshard"], ["master", "master"]];
      const median = (a) => { const s = a.slice().sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
      return frame(c, {
        tooltip: Object.assign(frame(c).tooltip, { trigger: "axis", valueFormatter: (v) => Number(v).toExponential(2) }),
        legend: Object.assign(frame(c).legend, { data: terms.map(([k]) => L.balance[k]) }),
        xAxis: axis(c, { type: "category", boundaryGap: false, data: spreads.map(String), name: L.spread, nameLocation: "middle", nameGap: 30, axisLabel: { color: c.ink }, splitLine: { show: false } }),
        yAxis: axis(c, { type: "log", name: L.grad, axisLabel: { color: c.faint, formatter: (v) => Number(v).toExponential(0) } }),
        series: terms.map(([k, field]) => ({
          name: L.balance[k], type: "line", symbol: "circle", symbolSize: 10, lineStyle: { width: 2.4, color: BALANCE[k] },
          itemStyle: { color: BALANCE[k], borderColor: c.panel, borderWidth: 1.5 },
          data: spreads.map((s) => median(d.pressure.filter((r) => r.spread === s).map((r) => r[field]))),
        })),
      });
    },

    "same-config-hours": function (d, c) {
      const rows = d.same_config.mAP50.rows;
      const fams = ["mixed", "fp32"];
      const mean = (fam, a) => { const v = rows.filter((r) => r.precision === fam).map((r) => r.hours[a]); return +(v.reduce((s, x) => s + x, 0) / v.length).toFixed(2); };
      return frame(c, {
        tooltip: Object.assign(frame(c).tooltip, { trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v) => v + " h" }),
        legend: Object.assign(frame(c).legend, { data: ["A", "A0", "B", "C"].map((a) => L.arms[a]) }),
        xAxis: axis(c, { type: "category", data: fams.map((f) => L[f]), axisLabel: { color: c.ink }, splitLine: { show: false } }),
        yAxis: axis(c, { type: "value", name: L.hours }),
        series: ["A", "A0", "B", "C"].map((a) => ({
          name: L.arms[a], type: "bar", barMaxWidth: 30, itemStyle: { color: ARM[a], borderRadius: [4, 4, 0, 0] },
          label: { show: true, position: "top", color: c.faint, fontSize: 11 }, data: fams.map((f) => mean(f, a)),
        })),
      });
    },
  };

  // Charts that read either metric get the same two-button switch the effect chart has.
  function metricSwitch(host, repaint) {
    if (host.dataset.metrics !== "both" || host.previousElementSibling?.classList.contains("esmoe-metric")) return;
    const bar = document.createElement("div");
    bar.className = "esmoe-metric";
    ["mAP50", "mAP50-95"].forEach((metric, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = metric;
      button.className = index === 0 ? "is-on" : "";
      button.addEventListener("click", () => {
        bar.querySelectorAll("button").forEach((b) => b.classList.remove("is-on"));
        button.classList.add("is-on");
        host.dataset.metric = metric;
        repaint();
      });
      bar.appendChild(button);
    });
    host.parentNode.insertBefore(bar, host);
  }

  function draw() {
    const data = window.ESMOE_DATA;
    if (!window.echarts || !data) return;
    document.querySelectorAll("[data-figure]").forEach((host) => {
      const build = FIGURES[host.dataset.figure];
      if (!build || (host.dataset.figure.startsWith("dataset") && !data.dataset)) return;
      const existing = window.echarts.getInstanceByDom(host);
      const chart = existing && !existing.isDisposed() ? existing : window.echarts.init(host, null, { renderer: "canvas" });
      const paint = () => chart.setOption(build(data, theme(), host), true);
      paint();
      if (host.dataset.wired) return;
      host.dataset.wired = "1";
      metricSwitch(host, paint);
      requestAnimationFrame(() => chart.resize());
      window.addEventListener("resize", () => chart.resize());
      new MutationObserver(paint).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
      SYSTEM.addEventListener("change", paint);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", draw);
  else draw();
  if (window.document$) window.document$.subscribe(draw);
})();
