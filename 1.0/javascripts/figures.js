// Figures for the experiment pages. Every number comes from data.js, which scripts/charts.py
// regenerates from results/, so this file only decides how the numbers are drawn.
(function () {
  const ZH = (document.documentElement.lang || "").toLowerCase().startsWith("zh");
  const SYSTEM = window.matchMedia("(prefers-color-scheme: dark)");
  const FONT = '"Source Sans 3", "Noto Sans SC", system-ui, sans-serif';

  // Hue tells the framework apart, depth tells whether the block is in: the fork is amber, official
  // ultralytics + esmoe is teal, and the arm without the block takes the lighter shade.
  const ARM = { A: "#B26A00", A0: "#E8A33D", B: "#007F69", C: "#2BB598" };
  const GRAFT = { default: "#2f6f9f", rewire: "#c2662d" };
  const SPLIT = { train: "#3e7c8c", val: "#e8a33d", test: "#9fc4cc" };
  const PRECISION = { mixed: "#3e7c8c", fp32: "#c2662d" };
  const AREA = { APs: "#3e7c8c", APm: "#e8a33d", APl: "#7a5cc2" };
  const BALANCE = { switch: "#007F69", gshard: "#B26A00", gshard_probs: "#E8A33D", master: "#7a5cc2", none: "#9aa3b2" };
  const WARN = "#b5453b";
  // Mid-tones that read on white sink into the dark page; the dark scheme lifts each one a step.
  const LIFT = {
    "#B26A00": "#d9901f", "#E8A33D": "#f0b75a", "#007F69": "#23b393", "#2BB598": "#52d1b3",
    "#2f6f9f": "#5d9bcc", "#c2662d": "#e08a4f", "#3e7c8c": "#62a8b8", "#e8a33d": "#f0b75a",
    "#9fc4cc": "#b7d8df", "#7a5cc2": "#a08be6", "#b5453b": "#e06a5e", "#9aa3b2": "#aeb6c4",
  };
  const BACKBONES = ["yolov5n", "yolov8n", "yolov9t", "yolov10n", "yolo11n", "yolo12n", "yolo26n"];
  const SPLITS = ["train", "val", "test"];

  const L = ZH
    ? {
        png: "存为 PNG", images: "图片", boxes: "标注框", perImage: "每张框数", split: { train: "训练集", val: "验证集", test: "测试集" },
        side: "框边长（缩放到 800 像素，像素）", median: "中位", runs: "运行次数", hours: "卡时",
        mixed: "混合精度", fp32: "FP32", seed: "seed ", gap: "mAP50 差", mean: "均值", max: "最大", dead: "个死专家",
        top1: "主导专家的 top-1 份额", checkpoints: "检查点", deadAny: "有死专家", deadNone: "无死专家",
        usage: "进入 top-2 的比例", block: "块", expert: "专家", spread: "路由概率的离散程度", grad: "每单位权重的梯度",
        positive: "为正", interval: "95% 区间", ratio: "卡时 / 同 seed 基线", count: "块数", kernel: "主导专家的核",
        corr: "路由概率与目标尺寸的相关系数", epoch: "epoch", relative: "最大相对差", fork: "YOLO-Master 分支", esmoe: "官方 + esmoe",
        default: "默认接法", rewire: "改接", seeds: "逐 seed", band: "噪声底 ±",
        balance: { switch: "Switch（读概率）", gshard: "GShard（读门控）", gshard_probs: "GShard（读概率）", master: "论文式 13", none: "无平衡项" },
        names: { "A - B": "A − B 两个实现", "A - A0": "A − A0 分支上加块", "B - C": "B − C 官方上加块", "(A - A0) - (B - C)": "差之差" },
        arms: { A: "A 分支 + ES_MOE", A0: "A0 分支去块", B: "B 官方 + esmoe", C: "C 官方去块" },
        pairs: {
          "A against B": "A 对 B：两个训练器",
          "A0 against C": "A0 对 C：两个基线",
          "A against A with one router layer nudged (relative 1e-06)": "A 对 A：一层路由扰动 1e-6",
          "B against B with one router layer nudged (relative 1e-06)": "B 对 B：一层路由扰动 1e-6",
        },
        tokens: { rewire: "改接", norm: "输出归一化", dense: "跑满专家", gshard: "GShard", master: "式 13", x4: "四块", noaux: "无平衡项" },
        candidates: { e2k1: "2 专家 top-1", e4k1: "4 专家 top-1", e4k2: "4 专家 top-2", e4k2w0: "4 专家 top-2，无辅助损失", e8k2: "8 专家 top-2" },
        budgets: { e20: "20 epoch · 640", e50: "50 epoch · 640", e100: "100 epoch · 640", e120: "120 epoch · 800", v11: "YOLO11n · 20 epoch" },
      }
    : {
        png: "PNG", images: "images", boxes: "boxes", perImage: "boxes per image", split: { train: "train", val: "val", test: "test" },
        side: "box side letterboxed to 800 px", median: "median", runs: "runs", hours: "card-hours",
        mixed: "mixed", fp32: "FP32", seed: "seed ", gap: "mAP50 gap", mean: "mean", max: "largest", dead: "dead experts",
        top1: "leading expert's top-1 share", checkpoints: "checkpoints", deadAny: "with dead experts", deadNone: "no dead expert",
        usage: "share of images in the top-2", block: "block", expert: "expert", spread: "spread of routing probabilities", grad: "gradient per unit weight",
        positive: "positive", interval: "95% CI", ratio: "card-hours over the same-seed baseline", count: "blocks", kernel: "leading expert's kernel",
        corr: "routing probability vs object size", epoch: "epoch", relative: "largest relative gap", fork: "YOLO-Master fork", esmoe: "official + esmoe",
        default: "default", rewire: "rewired", seeds: "per seed", band: "noise floor ±",
        balance: { switch: "Switch (probs)", gshard: "GShard (gate)", gshard_probs: "GShard (probs)", master: "paper eq. 13", none: "no balance term" },
        names: { "A - B": "A − B implementations", "A - A0": "A − A0 block on the fork", "B - C": "B − C block with esmoe", "(A - A0) - (B - C)": "difference of differences" },
        arms: { A: "A fork + ES_MOE", A0: "A0 fork, no block", B: "B official + esmoe", C: "C official, no block" },
        pairs: {
          "A against B": "A vs B: the two trainers",
          "A0 against C": "A0 vs C: the two baselines",
          "A against A with one router layer nudged (relative 1e-06)": "A vs A: one router layer nudged 1e-6",
          "B against B with one router layer nudged (relative 1e-06)": "B vs B: one router layer nudged 1e-6",
        },
        tokens: { rewire: "rewired", norm: "out norm", dense: "dense", gshard: "GShard", master: "eq. 13", x4: "four blocks", noaux: "no balance term" },
        candidates: { e2k1: "2 experts top-1", e4k1: "4 experts top-1", e4k2: "4 experts top-2", e4k2w0: "4 experts top-2, no aux loss", e8k2: "8 experts top-2" },
        budgets: { e20: "20 epochs · 640", e50: "50 epochs · 640", e100: "100 epochs · 640", e120: "120 epochs · 800", v11: "YOLO11n · 20 epochs" },
      };

  function dark() {
    const scheme = document.body.getAttribute("data-md-color-scheme");
    if (scheme === "slate") return true;
    if (scheme === "default") return false;
    return SYSTEM.matches;
  }

  function theme(host) {
    const palette = dark()
      ? { ink: "#e2e6ee", faint: "#98a1b3", rule: "#2a3244", zero: "#6f7a8c", panel: "#141b29", shade: "rgba(255,255,255,.04)" }
      : { ink: "#1d2433", faint: "#6b7385", rule: "#e4e8ef", zero: "#a3abb9", panel: "#ffffff", shade: "rgba(29,36,51,.04)" };
    // A phone gets smaller type, a scrolling legend and no axis titles; the figure caption names the axes.
    return Object.assign(palette, { narrow: host.clientWidth < 560 });
  }

  const tone = (hex) => (dark() && LIFT[hex]) || hex;

  // Every literal colour in a finished option goes through the lift, so no builder has to remember it.
  function lifted(value) {
    if (typeof value === "string") return tone(value);
    if (Array.isArray(value)) return value.map(lifted);
    if (value && typeof value === "object" && !(value instanceof Date)) {
      const out = {};
      for (const key of Object.keys(value)) out[key] = lifted(value[key]);
      return out;
    }
    return value;
  }

  const signed = (v) => (v >= 0 ? "+" : "−") + Math.abs(Number(v)).toFixed(4);
  const thousands = (v) => Number(v).toLocaleString(ZH ? "zh-CN" : "en-US");
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const size = (c, wide) => (c.narrow ? wide - 1 : wide);
  const named = (c, name) => (c.narrow ? "" : name);

  function tip(c, extra) {
    return Object.assign(
      {
        confine: true, backgroundColor: c.panel, borderColor: c.rule, borderWidth: 1, padding: [8, 12],
        textStyle: { color: c.ink, fontSize: 12, fontFamily: FONT },
        extraCssText: "border-radius:10px;box-shadow:0 10px 28px rgba(12,18,32,.16)",
      },
      extra,
    );
  }

  function legend(c, data) {
    return {
      type: c.narrow ? "scroll" : "plain", data: data, top: 0, left: 0, right: c.narrow ? 0 : 28,
      icon: "circle", itemWidth: 8, itemHeight: 8, itemGap: c.narrow ? 10 : 16,
      textStyle: { color: c.faint, fontSize: size(c, 12), fontFamily: FONT }, inactiveColor: c.rule,
      pageIconColor: c.faint, pageTextStyle: { color: c.faint },
    };
  }

  function frame(c, extra) {
    return Object.assign(
      {
        backgroundColor: "transparent",
        animationDuration: 450,
        animationEasing: "cubicOut",
        textStyle: { color: c.ink, fontFamily: FONT, fontSize: size(c, 12) },
        // Room above the plot for the legend row and, below it, the y-axis title.
        grid: { left: 4, right: c.narrow ? 6 : 14, top: 54, bottom: 4, containLabel: true },
        legend: { show: false },
        toolbox: {
          show: !c.narrow, right: 0, top: -2, itemSize: 13, iconStyle: { borderColor: c.faint },
          feature: { saveAsImage: { title: L.png, pixelRatio: 2, backgroundColor: c.panel } },
        },
        tooltip: tip(c),
      },
      extra,
    );
  }

  function axis(c, extra) {
    return Object.assign(
      {
        axisLine: { show: false, lineStyle: { color: c.rule } },
        axisTick: { show: false },
        axisLabel: { color: c.faint, fontSize: size(c, 11), fontFamily: FONT, hideOverlap: true },
        nameTextStyle: { color: c.faint, fontSize: size(c, 11), fontFamily: FONT },
        splitLine: { lineStyle: { color: c.rule, type: [4, 4] } },
      },
      extra,
    );
  }

  function category(c, data, extra) {
    return axis(c, Object.assign(
      {
        type: "category", data: data, axisLine: { show: true, lineStyle: { color: c.rule } }, splitLine: { show: false },
        axisLabel: { color: c.ink, fontSize: size(c, 11), fontFamily: FONT, hideOverlap: true, rotate: c.narrow ? 35 : 0 },
      },
      extra,
    ));
  }

  const shadow = (c) => ({ type: "shadow", shadowStyle: { color: c.shade } });
  const zeroLine = (c, key, at) => ({
    silent: true, symbol: "none", label: { show: false }, lineStyle: { color: c.zero, type: "solid", width: 1 },
    data: [{ [key]: at || 0 }],
  });
  const empty = (c) => frame(c, { title: { text: "—", left: "center", top: "middle", textStyle: { color: c.faint } } });

  // Short labels for arms: the variant string is exact but reads like a file name.
  function short(variant) {
    const [head, rest = ""] = variant.split("@");
    const stack = (rest.split("[")[1] || "").replace("]", "");
    const fp32 = rest.indexOf("fp32") >= 0;
    const match = head.match(/^(yolo-master-n|yolov?\d+[a-z])-(.*)$/);
    if (!match) return variant;
    const backbone = match[1];
    if (backbone === "yolo-master-n") return (stack.endsWith("/yolo-master") ? "A − A0" : "B − C") + " · " + (fp32 ? L.fp32 : L.mixed);
    const tokens = match[2].split("-");
    const weight = tokens[0].match(/w([\d.]+)$/);
    const words = tokens.slice(1).filter((t) => !/^t[\d.]+$/.test(t)).map((t) => L.tokens[t] || t);
    if (weight && Number(weight[1]) === 0) words.push(L.tokens.noaux);
    else if (weight && Number(weight[1]) !== 0.01) words.push("w" + weight[1]);
    const name = backbone.replace("yolov", "v").replace("yolo", "");
    const where = stack.startsWith("4090") ? "4090" : stack.indexOf("metax3.7") >= 0 ? "C500·3.7" : "C500";
    return name + (words.length ? " " + words.join(" ") : "") + " · " + where;
  }

  // Dots for seeds and a short bar for their mean, which every per-seed figure shares.
  function seedSeries(name, colour, points, horizontal) {
    return [
      {
        name: name, type: "scatter", symbolSize: 7, z: 3, itemStyle: { color: colour, opacity: 0.45 },
        data: points.flatMap((p) => p.seeds.map((v, i) => ({
          value: horizontal ? [v, p.at] : [p.at + (i - (p.seeds.length - 1) / 2) * 0.06, v], row: p,
        }))),
      },
      {
        name: name, type: "scatter", symbol: "rect", symbolSize: horizontal ? [3, 16] : [16, 3], z: 4, itemStyle: { color: colour },
        data: points.filter((p) => p.seeds.length >= 3).map((p) => ({ value: horizontal ? [mean(p.seeds), p.at] : [p.at, mean(p.seeds)], row: p })),
      },
    ];
  }

  function seedTip(p) {
    const r = p.data.row;
    return "<b>" + r.label + "</b><br>" + L.mean + " <b>" + signed(mean(r.seeds)) + "</b> · " + L.positive + " " +
      r.seeds.filter((v) => v > 0).length + "/" + r.seeds.length + "<br>" + L.seeds + " " + r.seeds.map(signed).join(", ");
  }

  // The mixed-precision noise floor, read from the repeated runs rather than written down here.
  function floorBand(d) {
    const gaps = d.floor.filter((p) => p.precision === "mixed").map((p) => p.gap);
    return gaps.length ? +mean(gaps).toFixed(4) : null;
  }

  function positionAxis(c, labels, name) {
    return axis(c, {
      type: "value", min: -0.5, max: labels.length - 0.5, splitLine: { show: false },
      axisLine: { show: true, lineStyle: { color: c.rule } }, name: named(c, name),
      axisLabel: { color: c.ink, fontSize: size(c, 11), fontFamily: FONT, rotate: c.narrow ? 35 : 0,
        customValues: labels.map((_, i) => i), formatter: (v) => labels[Math.round(v)] || "" },
    });
  }

  function areaBars(c, labels, cells) {
    const areas = ["APs", "APm", "APl"];
    const present = cells.map((cell, i) => [cell, labels[i]]).filter(([cell]) => cell && cell.seeds.length);
    if (!present.length) return empty(c);
    return frame(c, {
      legend: legend(c, areas),
      tooltip: tip(c, {
        trigger: "axis", axisPointer: shadow(c),
        formatter: (items) => {
          const [cell, label] = present[items[0].dataIndex];
          return "<b>" + label + "</b><br>" + areas.map((a) =>
            a + " " + L.mean + " <b>" + signed(mean(cell.seeds.map((s) => s[a]))) + "</b> · " + cell.seeds.map((s) => signed(s[a])).join(", ")).join("<br>");
        },
      }),
      xAxis: category(c, present.map(([, label]) => label), { axisLabel: { color: c.ink, fontSize: size(c, 11), fontFamily: FONT, interval: 0, rotate: c.narrow ? 35 : 0 } }),
      yAxis: axis(c, { type: "value", name: named(c, "Δ AP"), axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: signed } }),
      series: areas.map((area) => ({
        name: area, type: "bar", barGap: "12%", barMaxWidth: 14, itemStyle: { color: AREA[area], borderRadius: 2 },
        data: present.map(([cell]) => mean(cell.seeds.map((s) => s[area]))),
      })).concat([{ type: "line", data: [], markLine: zeroLine(c, "yAxis") }]),
    });
  }

  const FIGURES = {
    "dataset-splits": function (d, c) {
      const s = d.dataset.splits;
      const label = { show: !c.narrow, position: "top", color: c.faint, fontSize: 11, formatter: (p) => thousands(p.value) };
      return frame(c, {
        legend: legend(c, [L.images, L.boxes]),
        tooltip: tip(c, {
          trigger: "axis", axisPointer: shadow(c),
          formatter: (items) => {
            const f = s[SPLITS[items[0].dataIndex]];
            return "<b>" + L.split[SPLITS[items[0].dataIndex]] + "</b><br>" + L.images + " " + thousands(f.images) +
              "<br>" + L.boxes + " " + thousands(f.boxes) + "<br>" + L.perImage + " " + (f.boxes / f.images).toFixed(1) + "<br>" + (f.bytes / 1e9).toFixed(2) + " GB";
          },
        }),
        xAxis: category(c, SPLITS.map((k) => L.split[k]), { axisLabel: { color: c.ink, fontSize: size(c, 11), fontFamily: FONT } }),
        yAxis: [
          axis(c, { type: "value", name: named(c, L.images), axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: (v) => (v >= 1000 ? v / 1000 + "k" : v) } }),
          axis(c, { type: "value", name: named(c, L.boxes), splitLine: { show: false }, axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: (v) => (v >= 1000 ? v / 1000 + "k" : v) } }),
        ],
        series: [
          { name: L.images, type: "bar", barWidth: c.narrow ? 16 : 22, itemStyle: { color: SPLIT.train, borderRadius: [4, 4, 0, 0] }, label: label, data: SPLITS.map((k) => s[k].images) },
          { name: L.boxes, type: "bar", yAxisIndex: 1, barWidth: c.narrow ? 16 : 22, itemStyle: { color: SPLIT.val, borderRadius: [4, 4, 0, 0] }, label: label, data: SPLITS.map((k) => s[k].boxes) },
        ],
      });
    },

    "dataset-classes": function (d, c) {
      const s = d.dataset.splits;
      const order = d.dataset.names.map((n, i) => i).sort((a, b) => s.train.classes[a] - s.train.classes[b]);
      return frame(c, {
        legend: legend(c, SPLITS.map((k) => L.split[k])),
        tooltip: tip(c, { trigger: "axis", axisPointer: shadow(c) }),
        xAxis: axis(c, { type: "value", axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: (v) => (v >= 1000 ? v / 1000 + "k" : v) } }),
        yAxis: category(c, order.map((i) => d.dataset.names[i]), { axisLabel: { color: c.ink, fontSize: size(c, 11), fontFamily: FONT, rotate: 0 } }),
        series: SPLITS.map((k, index) => ({
          name: L.split[k], type: "bar", stack: "all", barWidth: c.narrow ? 10 : 13,
          itemStyle: { color: SPLIT[k], borderRadius: index === 2 ? [0, 3, 3, 0] : 0 }, emphasis: { focus: "series" },
          data: order.map((i) => s[k].classes[i] || 0),
        })),
      });
    },

    "dataset-resolutions": function (d, c) {
      const s = d.dataset.splits;
      const totals = {};
      SPLITS.forEach((k) => s[k].resolutions.forEach(([w, h, n]) => { totals[w + "×" + h] = (totals[w + "×" + h] || 0) + n; }));
      const keys = Object.keys(totals).sort((a, b) => totals[b] - totals[a]).slice(0, 8);
      const count = (k, key) => (s[k].resolutions.find(([w, h]) => w + "×" + h === key) || [0, 0, 0])[2];
      return frame(c, {
        legend: legend(c, SPLITS.map((k) => L.split[k])),
        tooltip: tip(c, { trigger: "axis", axisPointer: shadow(c) }),
        xAxis: category(c, keys, { axisLabel: { color: c.ink, fontSize: size(c, 10), fontFamily: FONT, rotate: 35, interval: 0 } }),
        yAxis: axis(c, { type: "value", name: named(c, L.images) }),
        series: SPLITS.map((k, index) => ({
          name: L.split[k], type: "bar", stack: "all", barMaxWidth: 24,
          itemStyle: { color: SPLIT[k], borderRadius: index === 2 ? [3, 3, 0, 0] : 0 }, emphasis: { focus: "series" },
          data: keys.map((key) => count(k, key)),
        })),
      });
    },

    "dataset-sides": function (d, c) {
      const s = d.dataset.splits;
      const edges = s.train.side_at_800.edges;
      const labels = edges.slice(0, -1).map((e, i) => (i === edges.length - 2 ? "≥" + e : e + "–" + edges[i + 1]));
      const label = (k) => L.split[k] + " · " + L.median + " " + s[k].median_side.at_800 + " px";
      return frame(c, {
        legend: legend(c, SPLITS.map(label)),
        tooltip: tip(c, { trigger: "axis", axisPointer: shadow(c), valueFormatter: (v) => (v * 100).toFixed(1) + "%" }),
        grid: { left: 4, right: 14, top: 54, bottom: c.narrow ? 4 : 22, containLabel: true },
        xAxis: category(c, labels, { name: named(c, L.side), nameLocation: "middle", nameGap: 28 }),
        yAxis: axis(c, { type: "value", axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: (v) => Math.round(v * 100) + "%" } }),
        series: SPLITS.map((k) => {
          const counts = s[k].side_at_800.counts;
          const total = counts.reduce((a, b) => a + b, 0) || 1;
          return { name: label(k), type: "bar", barGap: "10%", barMaxWidth: 12, itemStyle: { color: SPLIT[k], borderRadius: [3, 3, 0, 0] }, data: counts.map((n) => n / total) };
        }),
      });
    },

    "protocol-backbones": function (d, c) {
      const b = d.protocol.backbones;
      const keys = Object.keys(b).sort((x, y) => b[y].runs - b[x].runs);
      return frame(c, {
        legend: legend(c, [L.runs, L.hours]),
        tooltip: tip(c, { trigger: "axis" }),
        xAxis: category(c, keys),
        yAxis: [axis(c, { type: "value", name: named(c, L.runs) }), axis(c, { type: "value", name: named(c, L.hours), splitLine: { show: false } })],
        series: [
          { name: L.runs, type: "bar", barMaxWidth: 24, itemStyle: { color: SPLIT.train, borderRadius: [4, 4, 0, 0] },
            label: { show: !c.narrow, position: "top", color: c.faint, fontSize: 11 }, data: keys.map((k) => b[k].runs) },
          { name: L.hours, type: "line", yAxisIndex: 1, symbol: "circle", symbolSize: 6, itemStyle: { color: SPLIT.val }, lineStyle: { width: 2 }, data: keys.map((k) => Math.round(b[k].hours)) },
        ],
      });
    },

    generations: function (d, c, host) {
      const metric = host.dataset.metric || "mAP50";
      const rows = ((window.ESMOE_EFFECT || {}).rows || []).filter((r) => r[metric]);
      if (!rows.length) return empty(c);
      const labels = BACKBONES.filter((b) => rows.some((r) => r.backbone === b));
      const band = floorBand(d);
      const series = [["default graft", "default"], ["rewire", "rewire"]].flatMap(([key, name], index) => seedSeries(
        L[name], GRAFT[name],
        rows.filter((r) => r.arm === key).map((r) => ({ at: labels.indexOf(r.backbone) + (index - 0.5) * 0.28, seeds: r[metric].seeds, label: r.backbone + " · " + L[name] })),
      ));
      series.push({
        type: "line", data: [], markLine: zeroLine(c, "yAxis"),
        markArea: band === null ? undefined : { silent: true, itemStyle: { color: c.zero, opacity: 0.1 }, data: [[{ yAxis: -band }, { yAxis: band }]] },
      });
      return frame(c, {
        legend: legend(c, [L.default, L.rewire]),
        tooltip: tip(c, { trigger: "item", formatter: seedTip }),
        xAxis: positionAxis(c, labels.map((b) => b.replace("yolov", "v").replace("yolo", "")), ""),
        yAxis: axis(c, { type: "value", name: named(c, "Δ " + metric), axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: signed } }),
        series: series,
      });
    },

    alignment: function (d, c) {
      const rows = ((window.ESMOE_EFFECT || {}).alignment || []).slice().sort((a, b) => a.mean - b.mean);
      if (!rows.length) return empty(c);
      const labels = rows.map((r) => short(r.backbone + "-" + r.arm + "@e120f1i800[metaxc500/metax3.3]").replace(" · C500", ""));
      const points = rows.map((r, i) => ({ at: i, seeds: r.seeds, label: labels[i] }));
      const [dots, bars] = seedSeries("seeds", SPLIT.train, points, true);
      const tint = (p) => tone(mean(p.data.row.seeds) >= 0 ? BALANCE.switch : WARN);
      dots.itemStyle = { color: tint, opacity: 0.45 };
      bars.itemStyle = { color: tint };
      return frame(c, {
        grid: { left: 4, right: 18, top: 10, bottom: 4, containLabel: true },
        tooltip: tip(c, { trigger: "item", formatter: seedTip }),
        xAxis: axis(c, { type: "value", axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: signed } }),
        yAxis: category(c, labels, { axisLabel: { color: c.ink, fontSize: size(c, 11), fontFamily: FONT, rotate: 0 } }),
        series: [dots, bars, { type: "line", data: [], markLine: zeroLine(c, "xAxis") }],
      });
    },

    "buckets-generations": function (d, c, host) {
      const graft = host.dataset.arm || "default";
      const pattern = new RegExp("^(" + BACKBONES.join("|") + ")-e4k2w0\\.01" + (graft === "rewire" ? "-rewire" : "") + "@e120f1i800\\[(4090/torch2\\.6|metaxc500/metax3\\.3)\\]$");
      const cells = BACKBONES.map((b) => d.buckets.find((cell) => pattern.test(cell.variant) && cell.variant.startsWith(b + "-")));
      return areaBars(c, BACKBONES.map((b) => b.replace("yolov", "v").replace("yolo", "")), cells);
    },

    "buckets-same-config": function (d, c) {
      const order = [["mixed", true], ["mixed", false], ["fp32", true], ["fp32", false]];
      const cells = order.map(([precision, fork]) => d.buckets.find((b) =>
        b.variant.startsWith("yolo-master-n-") && (b.variant.indexOf("fp32") >= 0) === (precision === "fp32") && b.variant.endsWith("/yolo-master]") === fork));
      return areaBars(c, cells.map((cell) => (cell ? short(cell.variant) : "")), cells);
    },

    "selection-candidates": function (d, c) {
      const keys = [["e2k1", "yolov8n-e2k1w0.01"], ["e4k1", "yolov8n-e4k1w0.01"], ["e4k2", "yolov8n-e4k2w0.01"], ["e4k2w0", "yolov8n-e4k2w0.0"], ["e8k2", "yolov8n-e8k2w0.01"]];
      const cells = keys.map(([key, head]) => [key, (d.selection.find((s) => s.variant === head + "@e20f0.25i640[4090d/torch2.6]") || {}).mAP50]).filter(([, seeds]) => seeds);
      if (!cells.length) return empty(c);
      const labels = cells.map(([key]) => L.candidates[key]);
      const points = cells.map(([, seeds], i) => ({ at: i, seeds: seeds, label: labels[i] }));
      return frame(c, {
        grid: { left: 4, right: 18, top: 10, bottom: 4, containLabel: true },
        tooltip: tip(c, { trigger: "item", formatter: seedTip }),
        xAxis: axis(c, { type: "value", axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: signed } }),
        yAxis: category(c, labels, { inverse: true, axisLabel: { color: c.ink, fontSize: size(c, 11), fontFamily: FONT, rotate: 0 } }),
        series: [{
          type: "bar", barWidth: 12, z: 2, silent: true,
          itemStyle: { borderRadius: 3, opacity: 0.28, color: (p) => tone(p.value >= 0 ? GRAFT.default : WARN) },
          data: points.map((p) => p.seeds[0]),
        }].concat(seedSeries("seeds", GRAFT.default, points, true), [{ type: "line", data: [], markLine: zeroLine(c, "xAxis") }]),
      });
    },

    "selection-budget": function (d, c) {
      const effect = ((window.ESMOE_EFFECT || {}).rows || []).find((r) => r.backbone === "yolov8n" && r.arm === "default graft");
      const find = (variant) => (d.selection.find((s) => s.variant === variant) || {}).mAP50;
      const cells = [
        ["e20", find("yolov8n-e4k2w0.01@e20f1i640[4090d/torch2.6]")],
        ["e50", find("yolov8n-e4k2w0.01@e50f1i640[4090d/torch2.6]")],
        ["e100", find("yolov8n-e4k2w0.01@e100f1i640[4090d/torch2.6]")],
        ["e120", effect && effect.mAP50.seeds],
        ["v11", find("yolo11n-e4k2w0.01@e20f1i640[4090d/torch2.6]")],
      ].filter(([, seeds]) => seeds);
      if (!cells.length) return empty(c);
      const labels = cells.map(([key]) => L.budgets[key]);
      const points = cells.map(([key, seeds], i) => ({ at: i, seeds: seeds, label: labels[i], other: key === "v11" }));
      return frame(c, {
        legend: legend(c, ["YOLOv8n", "YOLO11n"]),
        tooltip: tip(c, { trigger: "item", formatter: seedTip }),
        xAxis: positionAxis(c, labels, ""),
        yAxis: axis(c, { type: "value", name: named(c, "Δ mAP50"), axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: signed } }),
        series: seedSeries("YOLOv8n", GRAFT.default, points.filter((p) => !p.other))
          .concat(seedSeries("YOLO11n", GRAFT.rewire, points.filter((p) => p.other)), [{ type: "line", data: [], markLine: zeroLine(c, "yAxis") }]),
      });
    },

    cost: function (d, c) {
      const cells = d.cost.filter((cell) => !cell.variant.startsWith("yolo-master-n") && cell.ratios.length)
        .map((cell) => ({ label: short(cell.variant), value: mean(cell.ratios), cell: cell })).sort((a, b) => a.value - b.value);
      if (!cells.length) return empty(c);
      return frame(c, {
        grid: { left: 4, right: 44, top: 10, bottom: 4, containLabel: true },
        tooltip: tip(c, { trigger: "item", formatter: (p) => { const x = cells[p.dataIndex]; return "<b>" + x.label + "</b><br>" + L.mean + " ×" + x.value.toFixed(3) + "<br>" + L.seeds + " " + x.cell.ratios.map((v) => "×" + v.toFixed(3)).join(", "); } }),
        xAxis: axis(c, { type: "value", min: 0.9, axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: (v) => "×" + v.toFixed(1) } }),
        yAxis: category(c, cells.map((x) => x.label), { axisLabel: { color: c.ink, fontSize: size(c, 10), fontFamily: FONT, rotate: 0, interval: 0 } }),
        series: [
          { type: "bar", barWidth: 8, itemStyle: { borderRadius: [0, 3, 3, 0], color: (p) => tone(cells[p.dataIndex].label.indexOf(L.tokens.x4) >= 0 ? SPLIT.val : SPLIT.train) },
            label: { show: true, position: "right", color: c.faint, fontSize: 10, formatter: (p) => "×" + p.value.toFixed(2) }, data: cells.map((x) => x.value) },
          { type: "line", data: [], markLine: zeroLine(c, "xAxis", 1) },
        ],
      });
    },

    "same-config-arms": function (d, c, host) {
      const metric = host.dataset.metric || "mAP50";
      const rows = d.same_config[metric].rows.slice().sort((a, b) => (a.precision === b.precision ? a.seed - b.seed : a.precision === "mixed" ? -1 : 1));
      if (!rows.length) return empty(c);
      const cats = rows.map((r) => (r.precision === "fp32" ? L.fp32 : L.mixed) + " · " + r.seed);
      const values = rows.flatMap((r) => ["A", "A0", "B", "C"].map((a) => r[a]));
      return frame(c, {
        legend: legend(c, ["A", "A0", "B", "C"].map((a) => L.arms[a])),
        tooltip: tip(c, { trigger: "axis", valueFormatter: (v) => Number(v).toFixed(4) }),
        xAxis: category(c, cats, { axisLabel: { color: c.ink, fontSize: size(c, 11), fontFamily: FONT, interval: 0, rotate: c.narrow ? 35 : 0 } }),
        yAxis: axis(c, { type: "value", name: named(c, metric), min: +(Math.min.apply(null, values) - 0.004).toFixed(3), max: +(Math.max.apply(null, values) + 0.002).toFixed(3),
          axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: (v) => v.toFixed(3) } }),
        series: ["A", "A0", "B", "C"].map((a) => ({
          name: L.arms[a], type: "line", symbol: "circle", symbolSize: c.narrow ? 8 : 10, lineStyle: { width: 0 },
          itemStyle: { color: ARM[a], borderColor: c.panel, borderWidth: 1.5 }, emphasis: { scale: 1.4 }, data: rows.map((r) => r[a]),
        })),
      });
    },

    "same-config-differences": function (d, c, host) {
      const metric = host.dataset.metric || "mAP50";
      const diffs = d.same_config[metric].differences;
      const names = ["A - B", "A - A0", "B - C", "(A - A0) - (B - C)"];
      const families = ["mixed", "fp32"].filter((fam) => diffs.some((x) => x.precision === fam));
      if (!families.length) return empty(c);
      const band = floorBand(d);
      const series = [];
      families.forEach((fam, fi) => {
        const nudge = families.length > 1 ? (fi - 0.5) * 0.3 : 0;
        const mine = names.map((n, i) => [diffs.find((x) => x.precision === fam && x.name === n), i]).filter(([row]) => row);
        series.push({
          name: L[fam], type: "custom", z: 2, tooltip: { show: false },
          renderItem: (params, api) => {
            const [row, i] = mine[params.dataIndex];
            if (row.lo === null) return null;
            const y = api.coord([0, i + nudge])[1];
            const x1 = api.coord([row.lo, 0])[0], x2 = api.coord([row.hi, 0])[0];
            const stroke = { stroke: PRECISION[fam], lineWidth: 2 };
            return { type: "group", children: [
              { type: "line", shape: { x1: x1, y1: y, x2: x2, y2: y }, style: Object.assign({ opacity: 0.7 }, stroke) },
              { type: "line", shape: { x1: x1, y1: y - 4, x2: x1, y2: y + 4 }, style: stroke },
              { type: "line", shape: { x1: x2, y1: y - 4, x2: x2, y2: y + 4 }, style: stroke },
            ] };
          },
          data: mine.map(([row, i]) => [row.mean, i]),
        });
        series.push({ name: L[fam], type: "scatter", symbolSize: 11, z: 4, itemStyle: { color: PRECISION[fam], borderColor: c.panel, borderWidth: 1.5 },
          data: mine.map(([row, i]) => ({ value: [row.mean, i + nudge], row: row })) });
        series.push({ name: L[fam], type: "scatter", symbolSize: 5, z: 3, itemStyle: { color: PRECISION[fam], opacity: 0.45 },
          data: mine.flatMap(([row, i]) => row.values.map((v) => ({ value: [v, i + nudge], row: row }))) });
      });
      series.push({
        type: "line", data: [], markLine: zeroLine(c, "xAxis"),
        markArea: band === null ? undefined : { silent: true, itemStyle: { color: c.zero, opacity: 0.1 },
          label: { show: !c.narrow, position: "insideTop", color: c.faint, fontSize: 10, formatter: L.band + band }, data: [[{ xAxis: -band }, { xAxis: band }]] },
      });
      return frame(c, {
        color: families.map((fam) => PRECISION[fam]),
        legend: legend(c, families.map((fam) => L[fam])),
        tooltip: tip(c, {
          trigger: "item",
          formatter: (p) => {
            const r = p.data.row;
            const ci = r.lo === null ? "–" : "[" + signed(r.lo) + ", " + signed(r.hi) + "]";
            return "<b>" + L.names[r.name] + "</b> · " + L[r.precision] + "<br>" + L.mean + " <b>" + signed(r.mean) + "</b><br>" +
              L.interval + " " + ci + "<br>" + L.positive + " " + r.positive + "/" + r.values.length + "<br>" + L.seeds + " " + r.values.map(signed).join(", ");
          },
        }),
        xAxis: axis(c, {
          type: "value", name: named(c, "Δ " + metric), nameLocation: "middle", nameGap: 26, splitNumber: c.narrow ? 3 : 5,
          // The axis follows the intervals, which the renderer draws past the points it would otherwise fit.
          min: (v) => +(Math.min(v.min, ...diffs.map((x) => (x.lo === null ? x.mean : x.lo))) - 0.002).toFixed(3),
          max: (v) => +(Math.max(v.max, ...diffs.map((x) => (x.hi === null ? x.mean : x.hi))) + 0.002).toFixed(3),
          axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: (v) => (c.narrow ? (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3) : signed(v)) },
        }),
        yAxis: axis(c, { type: "value", min: -0.6, max: 3.6, inverse: true, splitLine: { show: false },
          axisLabel: { color: c.ink, fontSize: size(c, 11), fontFamily: FONT, customValues: [0, 1, 2, 3], formatter: (v) => L.names[names[Math.round(v)]] } }),
        series: series,
      });
    },

    "same-config-hours": function (d, c) {
      const rows = d.same_config.mAP50.rows;
      const fams = ["mixed", "fp32"].filter((fam) => rows.some((r) => r.precision === fam));
      if (!fams.length) return empty(c);
      const avg = (fam, a) => +mean(rows.filter((r) => r.precision === fam).map((r) => r.hours[a])).toFixed(2);
      return frame(c, {
        legend: legend(c, ["A", "A0", "B", "C"].map((a) => L.arms[a])),
        tooltip: tip(c, { trigger: "axis", axisPointer: shadow(c), valueFormatter: (v) => v + " h" }),
        xAxis: category(c, fams.map((f) => L[f]), { axisLabel: { color: c.ink, fontSize: size(c, 11), fontFamily: FONT } }),
        yAxis: axis(c, { type: "value", name: named(c, L.hours) }),
        series: ["A", "A0", "B", "C"].map((a) => ({
          name: L.arms[a], type: "bar", barMaxWidth: 22, itemStyle: { color: ARM[a], borderRadius: [4, 4, 0, 0] },
          label: { show: !c.narrow, position: "top", color: c.faint, fontSize: 10 }, data: fams.map((f) => avg(f, a)),
        })),
      });
    },

    "recipe-parity": function (d, c) {
      const pairs = d.recipe_parity;
      if (!pairs.length) return empty(c);
      const colours = [ARM.B, ARM.C, ARM.A, "#7a5cc2"];
      return frame(c, {
        legend: legend(c, pairs.map((p) => L.pairs[p.pair] || p.pair)),
        tooltip: tip(c, { trigger: "axis", valueFormatter: (v) => Number(v).toExponential(2) }),
        grid: { left: 4, right: 14, top: c.narrow ? 40 : 58, bottom: c.narrow ? 4 : 20, containLabel: true },
        xAxis: category(c, pairs[0].epochs.map((e) => String(e.epoch)), { name: named(c, L.epoch), nameLocation: "middle", nameGap: 24, boundaryGap: false }),
        yAxis: axis(c, { type: "log", name: named(c, L.relative), axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: (v) => Number(v).toExponential(0) } }),
        series: pairs.map((p, i) => ({
          name: L.pairs[p.pair] || p.pair, type: "line", symbol: "circle", symbolSize: 6,
          lineStyle: { width: 2, type: i >= 2 ? "dashed" : "solid", color: colours[i % colours.length] }, itemStyle: { color: colours[i % colours.length] },
          // A log axis has no zero; an epoch where the two traces agree exactly is left as a gap.
          data: p.epochs.map((e) => (e.relative > 0 ? e.relative : null)),
        })),
      });
    },

    "release-check": function (d, c) {
      const rows = d.release_check;
      if (!rows.length) return empty(c);
      return frame(c, {
        legend: legend(c, [L.fork, L.esmoe]),
        tooltip: tip(c, { trigger: "axis", axisPointer: shadow(c), valueFormatter: (v) => Number(v).toFixed(5) }),
        xAxis: category(c, rows.map((r) => r.metric.replace("metrics/", "").replace("(B)", "")), { axisLabel: { color: c.ink, fontSize: size(c, 11), fontFamily: FONT, interval: 0 } }),
        yAxis: axis(c, { type: "value", min: 0, max: 0.8 }),
        series: [["fork", ARM.A], ["esmoe", ARM.B]].map(([key, colour]) => ({
          name: L[key], type: "bar", barWidth: c.narrow ? 12 : 20, itemStyle: { color: colour, borderRadius: [4, 4, 0, 0] },
          label: { show: !c.narrow, position: "top", color: c.faint, fontSize: 10, formatter: (p) => p.value.toFixed(5) }, data: rows.map((r) => r[key]),
        })),
      });
    },

    "noise-floor": function (d, c) {
      const pairs = d.floor.slice().sort((a, b) => (a.precision > b.precision ? -1 : a.precision < b.precision ? 1 : b.gap - a.gap));
      const families = ["mixed", "fp32"].filter((fam) => pairs.some((p) => p.precision === fam));
      if (!families.length) return empty(c);
      const label = (p) => {
        const fork = p.label.indexOf("/yolo-master]") >= 0;
        const role = p.label.indexOf("baseline") >= 0 ? (fork ? "A0" : "C") : p.label.indexOf("gshard-norm-dense") >= 0 ? (fork ? "A" : "B") : null;
        return (role || short(p.label).split(" · ")[0]) + " · " + p.seed;
      };
      const stats = (fam) => {
        const g = pairs.filter((p) => p.precision === fam).map((p) => p.gap);
        return L[fam] + " · " + L.mean + " " + mean(g).toFixed(4) + " · " + L.max + " " + Math.max.apply(null, g).toFixed(4);
      };
      return frame(c, {
        legend: legend(c, families.map(stats)),
        tooltip: tip(c, { trigger: "item", formatter: (p) => { const q = pairs[p.dataIndex]; return "<b>" + label(q) + "</b> · " + L[q.precision] + "<br>" + q.first.toFixed(4) + " → " + q.repeat.toFixed(4) + "<br>" + L.gap + " <b>" + q.gap.toFixed(4) + "</b>"; } }),
        xAxis: category(c, pairs.map(label), { axisLabel: { color: c.ink, fontSize: size(c, 10), fontFamily: FONT, rotate: 35, interval: 0 } }),
        yAxis: axis(c, { type: "value", name: named(c, L.gap), axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: (v) => v.toFixed(3) } }),
        series: families.map((fam) => ({
          name: stats(fam), type: "bar", barMaxWidth: 20, stack: "gap", itemStyle: { color: PRECISION[fam], borderRadius: [4, 4, 0, 0] },
          data: pairs.map((p) => (p.precision === fam ? p.gap : null)),
        })),
      });
    },

    "routing-scatter": function (d, c) {
      const pts = d.routing.points;
      if (pts.length < 3) return empty(c);
      const xs = pts.map((p) => p.top1), ys = pts.map((p) => p.delta);
      const mx = mean(xs), my = mean(ys);
      const sd = (a, m) => Math.sqrt(mean(a.map((v) => (v - m) * (v - m))));
      const r = mean(xs.map((x, i) => (x - mx) * (ys[i] - my))) / (sd(xs, mx) * sd(ys, my));
      const kinds = Object.keys(BALANCE).filter((k) => pts.some((p) => p.balance === k));
      return frame(c, {
        title: { text: "r = " + signed(r).slice(0, 6) + " · n = " + pts.length, right: c.narrow ? 0 : 26, bottom: c.narrow ? 0 : 26, textStyle: { color: c.faint, fontSize: 11, fontWeight: "normal", fontFamily: FONT } },
        legend: legend(c, kinds.map((k) => L.balance[k])),
        grid: { left: 4, right: 14, top: c.narrow ? 60 : 54, bottom: c.narrow ? 18 : 22, containLabel: true },
        tooltip: tip(c, { trigger: "item", formatter: (p) => { const q = p.data.q; return "<b>" + short(q.variant + "@e120f1i800[x]").split(" · ")[0] + "</b> · seed " + q.run.split("-s")[1].split("-")[0] + "<br>" + L.top1 + " " + q.top1 + "<br>Δ mAP50 <b>" + signed(q.delta) + "</b><br>" + q.dead + " " + L.dead; } }),
        xAxis: axis(c, { type: "value", name: named(c, L.top1), nameLocation: "middle", nameGap: 24, min: (v) => Math.floor(v.min * 10) / 10, max: 1.02,
          axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: (v) => (v > 1 ? "" : v.toFixed(1)) } }),
        yAxis: axis(c, { type: "value", axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: signed } }),
        series: kinds.map((k, i) => ({
          name: L.balance[k], type: "scatter", itemStyle: { color: BALANCE[k], opacity: 0.8, borderColor: c.panel, borderWidth: 1 }, emphasis: { scale: 1.5 },
          data: pts.filter((p) => p.balance === k).map((q) => ({ value: [q.top1, q.delta], symbolSize: q.blocks > 1 ? 13 : 8, q: q })),
          markLine: i === 0 ? zeroLine(c, "yAxis") : undefined,
        })),
      });
    },

    "dead-experts": function (d, c) {
      const pts = d.routing.points;
      const kinds = Object.keys(BALANCE).filter((k) => pts.some((p) => p.balance === k));
      if (!kinds.length) return empty(c);
      return frame(c, {
        legend: legend(c, [L.deadNone, L.deadAny]),
        tooltip: tip(c, { trigger: "axis", axisPointer: shadow(c) }),
        xAxis: category(c, kinds.map((k) => L.balance[k]), { axisLabel: { color: c.ink, fontSize: size(c, 10), fontFamily: FONT, interval: 0, rotate: c.narrow ? 35 : 0 } }),
        yAxis: axis(c, { type: "value", name: named(c, L.checkpoints) }),
        series: [
          { name: L.deadNone, type: "bar", stack: "n", barMaxWidth: 30, itemStyle: { color: ARM.C }, data: kinds.map((k) => pts.filter((p) => p.balance === k && !p.dead).length || null) },
          { name: L.deadAny, type: "bar", stack: "n", barMaxWidth: 30, itemStyle: { color: WARN, borderRadius: [4, 4, 0, 0] },
            label: { show: true, position: "top", color: c.faint, fontSize: 10 }, data: kinds.map((k) => pts.filter((p) => p.balance === k && p.dead).length || null) },
        ],
      });
    },

    pressure: function (d, c) {
      const spreads = [...new Set(d.pressure.map((r) => r.spread))].sort((a, b) => a - b);
      if (!spreads.length) return empty(c);
      const terms = ["switch", "gshard_probs", "gshard", "master"];
      const median = (a) => { const s = a.slice().sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
      return frame(c, {
        legend: legend(c, terms.map((k) => L.balance[k])),
        tooltip: tip(c, { trigger: "axis", valueFormatter: (v) => Number(v).toExponential(2) }),
        grid: { left: 4, right: 14, top: c.narrow ? 60 : 54, bottom: c.narrow ? 4 : 22, containLabel: true },
        xAxis: category(c, spreads.map(String), { name: named(c, L.spread), nameLocation: "middle", nameGap: 24, boundaryGap: false }),
        yAxis: axis(c, { type: "log", name: named(c, L.grad), axisLabel: { color: c.faint, fontSize: size(c, 11), formatter: (v) => Number(v).toExponential(0) } }),
        series: terms.map((k) => ({
          name: L.balance[k], type: "line", symbol: "circle", symbolSize: 8, lineStyle: { width: 2, color: BALANCE[k] },
          itemStyle: { color: BALANCE[k], borderColor: c.panel, borderWidth: 1.5 },
          data: spreads.map((s) => median(d.pressure.filter((r) => r.spread === s).map((r) => r[k]))),
        })),
      });
    },

    "scale-correlations": function (d, c) {
      const values = d.scale.correlations;
      if (!values.length) return empty(c);
      const edges = [];
      for (let e = -0.5; e < 0.6001; e += 0.1) edges.push(+e.toFixed(1));
      const counts = edges.slice(0, -1).map((lo, i) => values.filter((v) => v >= lo && v < edges[i + 1]).length);
      return frame(c, {
        grid: { left: 4, right: 14, top: 10, bottom: c.narrow ? 4 : 22, containLabel: true },
        tooltip: tip(c, { trigger: "axis", axisPointer: shadow(c) }),
        xAxis: category(c, edges.slice(0, -1).map((lo) => (lo >= 0 ? "+" : "") + lo.toFixed(1)), { name: named(c, L.corr), nameLocation: "middle", nameGap: 24, boundaryGap: true }),
        yAxis: axis(c, { type: "value" }),
        series: [{ type: "bar", barCategoryGap: "14%", itemStyle: { borderRadius: [3, 3, 0, 0], color: (p) => tone(edges[p.dataIndex] < 0 ? ARM.A0 : SPLIT.train) }, data: counts }],
      });
    },

    "scale-leaders": function (d, c) {
      const kernels = Object.keys(d.scale.leaders).sort((a, b) => a - b);
      if (!kernels.length) return empty(c);
      return frame(c, {
        grid: { left: 4, right: 14, top: 10, bottom: c.narrow ? 4 : 22, containLabel: true },
        tooltip: tip(c, { trigger: "axis" }),
        xAxis: category(c, kernels.map((k) => k + "×" + k), { name: named(c, L.kernel), nameLocation: "middle", nameGap: 24 }),
        yAxis: axis(c, { type: "value", name: named(c, L.count) }),
        series: [{ type: "bar", barWidth: c.narrow ? 20 : 28, itemStyle: { color: SPLIT.train, borderRadius: [4, 4, 0, 0] }, label: { show: true, position: "top", color: c.faint, fontSize: 11 }, data: kernels.map((k) => d.scale.leaders[k]) }],
      });
    },

    "same-config-dead": function (d, c) {
      const arms = d.routing.same_config.slice().sort((a, b) => (a.precision === b.precision ? a.seed - b.seed : a.precision === "mixed" ? -1 : 1));
      if (!arms.length) return empty(c);
      const rows = arms.map((a) => L[a.precision] + " · " + a.seed);
      const cols = [];
      arms[0].blocks.forEach((b, bi) => b.usage.forEach((u, ei) => cols.push(L.block + " " + bi + " · " + L.expert + " " + ei)));
      const cells = [];
      arms.forEach((a, ri) => {
        let ci = 0;
        a.blocks.forEach((b) => b.usage.forEach((u) => { cells.push([ci, ri, +u.toFixed(3)]); ci += 1; }));
      });
      return frame(c, {
        grid: { left: 4, right: 10, top: 6, bottom: 40, containLabel: true },
        tooltip: tip(c, { trigger: "item", formatter: (p) => rows[p.value[1]] + "<br>" + cols[p.value[0]] + "<br>" + L.usage + " <b>" + p.value[2] + "</b>" }),
        xAxis: { type: "category", data: cols, axisTick: { show: false }, axisLine: { show: false },
          axisLabel: { show: !c.narrow, color: c.faint, fontSize: 9, rotate: 60, fontFamily: FONT, interval: 0, formatter: (v) => v.replace(/^.* · /, "").replace(L.expert + " ", "") } },
        yAxis: { type: "category", data: rows, axisTick: { show: false }, axisLine: { show: false }, axisLabel: { color: c.ink, fontSize: size(c, 11), fontFamily: FONT } },
        visualMap: { min: 0, max: 1, orient: "horizontal", left: "center", bottom: 0, itemWidth: 9, itemHeight: 120, text: ["1", "0"],
          textStyle: { color: c.faint, fontSize: 10 }, inRange: { color: [WARN, dark() ? "#3a3526" : "#f1e2c2", ARM.B] } },
        series: [{ type: "heatmap", data: cells, itemStyle: { borderColor: c.panel, borderWidth: c.narrow ? 1 : 2, borderRadius: 2 }, emphasis: { itemStyle: { borderColor: c.ink } } }],
      });
    },
  };

  // A pill switch above a figure; the figure reads the chosen value from its own dataset.
  function switcher(host, key, values, labels, repaint) {
    const bar = document.createElement("div");
    bar.className = "esmoe-metric";
    values.forEach((value, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = labels[index];
      button.className = index === 0 ? "is-on" : "";
      button.addEventListener("click", () => {
        bar.querySelectorAll("button").forEach((b) => b.classList.remove("is-on"));
        button.classList.add("is-on");
        host.dataset[key] = value;
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
      if (!build || host.dataset.wired || (host.dataset.figure.startsWith("dataset") && !data.dataset)) return;
      host.dataset.wired = "1";
      const chart = window.echarts.init(host, null, { renderer: "canvas" });
      // One figure whose data is missing must not take the rest of the page down with it.
      const paint = () => {
        try {
          chart.setOption(lifted(build(data, theme(host), host)), true);
        } catch (error) {
          console.error("figure " + host.dataset.figure, error);
        }
      };
      paint();
      if (host.dataset.metrics === "both") switcher(host, "metric", ["mAP50", "mAP50-95"], ["mAP50", "mAP50-95"], paint);
      if (host.dataset.arms === "both") switcher(host, "arm", ["default", "rewire"], [L.default, L.rewire], paint);
      let narrow = host.clientWidth < 560;
      new ResizeObserver(() => {
        chart.resize();
        if ((host.clientWidth < 560) !== narrow) {
          narrow = !narrow;
          paint();
        }
      }).observe(host);
      new MutationObserver(paint).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
      SYSTEM.addEventListener("change", paint);
    });
  }

  (window.ESMOE = window.ESMOE || {}).figures = draw;
})();
