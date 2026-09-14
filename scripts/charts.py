"""Draw every figure the docs site, README and wiki show, straight from results/.

    uv run python scripts/charts.py

Outputs:

- `docs/assets/effect.svg`, `effect-zh.svg`, `alignment.svg`, `alignment-zh.svg`: static figures for
  README and the wiki, where GitHub strips scripts and only a picture renders.
- `docs/javascripts/data.js`: `window.ESMOE_EFFECT`, the seven-generation paired deltas and the
  alignment arms, and `window.ESMOE_DATA`, everything else the ECharts figures draw: dataset
  statistics (`results/dataset.json`), the protocol matrix, the same-configuration arms, repeated runs,
  routing records, balancing pressure, area buckets, the selection stage, the release check,
  step-for-step traces and card-hours.

The numbers come through report.py and same_config.py, or from the tables their sibling scripts write,
so a figure cannot drift away from the tables it illustrates.
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report import KEYS, arm, bounds, dedupe, interval, load, paired, published, schedule, variant  # noqa: E402
from same_config import ARMS as FOUR  # noqa: E402
from same_config import DELTAS  # noqa: E402
from same_config import collect as four_arms  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SVG_OUT = {"en": ROOT / "docs" / "assets" / "effect.svg", "zh": ROOT / "docs" / "assets" / "effect-zh.svg"}
ALIGN_OUT = {"en": ROOT / "docs" / "assets" / "alignment.svg", "zh": ROOT / "docs" / "assets" / "alignment-zh.svg"}
DATA_OUT = ROOT / "docs" / "javascripts" / "data.js"
DATASET = ROOT / "results" / "dataset.json"
ROUTING = ROOT / "results" / "routing"
PRESSURE = ROOT / "results" / "pressure.md"
BUCKETS = ROOT / "results" / "buckets"
RELEASE_CHECK = ROOT / "results" / "release_check.md"
RECIPE_PARITY = ROOT / "results" / "recipe_parity.md"
AREAS = ("AP", "APs", "APm", "APl")

# Oldest to newest: the axis order is the claim, so it is fixed rather than sorted.
ORDER = ("yolov5n", "yolov8n", "yolov9t", "yolov10n", "yolo11n", "yolo12n", "yolo26n")
PROTOCOL = "@e120f1i800"
METRICS = (("mAP50", KEYS[0]), ("mAP50-95", KEYS[1]))

W, H = 760, 380
PAD = {"l": 74, "r": 166, "t": 40, "b": 58}
ARMS = (("e4k2w0.01", "default graft", "#2f6f9f"), ("e4k2w0.01-rewire", "rewire", "#c2662d"))

# The figure ships in both languages: an English report should not carry Chinese axis labels,
# and a Chinese one should not carry English.
TEXT = {
    "en": {
        "title": "Paired mAP50 delta against the same-seed baseline",
        "arm": "arm",
        "arms": ("ES-MoE default", "ES-MoE rewired"),
        "notes": (
            "dot = one seed",
            "bar = mean of three",
            "no bar = under 3 seeds",
            "800px, 120 epochs,",
            "full VisDrone",
        ),
        "font": "system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif",
    },
    "zh": {
        "title": "同 seed 配对的 mAP50 差值",
        "arm": "臂",
        "arms": ("ES-MoE 默认", "ES-MoE 改接"),
        "notes": (
            "散点 = 单个 seed",
            "横杠 = 三 seed 均值",
            "无横杠 = 不足三个 seed",
            "800px、120 epoch、",
            "VisDrone 全量",
        ),
        "font": "Noto Sans SC,Source Sans 3,Microsoft YaHei,system-ui,sans-serif",
    },
}


def collect(key):
    """Paired deltas per backbone and arm, protocol runs only."""
    runs, _ = dedupe(load())
    _, deltas = paired(runs, key)
    table = {}
    for label, values in deltas.items():
        head, _, rest = label.partition("-")
        block = rest.split("@")[0]
        if PROTOCOL not in label or head not in ORDER:
            continue
        table[(head, block)] = values
    return table


def scale(lo, hi):
    span = max(abs(lo), abs(hi)) * 1.15 or 0.01
    plot_h = H - PAD["t"] - PAD["b"]

    def y(value):
        return PAD["t"] + plot_h / 2 - value / span * plot_h / 2

    return y, span


def svg(table, lang="en"):
    present = [b for b in ORDER if any((b, a) in table for a, _, _ in ARMS)]
    values = [v for key in table for v in table[key]]
    y, span = scale(min(values), max(values))
    plot_w = W - PAD["l"] - PAD["r"]
    step = plot_w / max(len(present), 1)

    words = TEXT[lang]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="{words["font"]}" font-size="12">',
        "<style>",
        "  .ink{fill:#3d4451}.rule{stroke:#c9ced8}.zero{stroke:#8b93a3}.faint{fill:#78808f}",
        "  @media (prefers-color-scheme:dark){",
        "    .ink{fill:#c9ced8}.rule{stroke:#454b57}.zero{stroke:#7e8797}.faint{fill:#98a0af}}",
        "</style>",
        f'<text x="{PAD["l"]}" y="22" class="ink" font-size="13" font-weight="600">{words["title"]}</text>',
    ]

    for tick in (-1, -0.5, 0, 0.5, 1):
        value = span * tick
        yy = y(value)
        cls = "zero" if tick == 0 else "rule"
        dash = "" if tick == 0 else ' stroke-dasharray="3 4"'
        parts.append(
            f'<line x1="{PAD["l"]}" y1="{yy:.1f}" x2="{W - PAD["r"]}" y2="{yy:.1f}" '
            f'class="{cls}"{dash} stroke-width="1"/>'
        )
        parts.append(f'<text x="{PAD["l"] - 10}" y="{yy + 4:.1f}" class="faint" text-anchor="end">{value:+.4f}</text>')

    for index, backbone in enumerate(present):
        cx = PAD["l"] + step * (index + 0.5)
        parts.append(f'<text x="{cx:.1f}" y="{H - PAD["b"] + 24}" class="ink" text-anchor="middle">{backbone}</text>')
        for offset, (block, _, colour) in zip((-14, 14), ARMS, strict=True):
            series = table.get((backbone, block))
            if not series:
                continue
            x = cx + offset
            # No mean below three seeds, so an unfinished arm cannot read as settled.
            if len(series) >= 3:
                mean = statistics.mean(series)
                parts.append(
                    f'<line x1="{x - 11}" y1="{y(mean):.1f}" x2="{x + 11}" '
                    f'y2="{y(mean):.1f}" stroke="{colour}" stroke-width="2.5" '
                    f'stroke-linecap="round"/>'
                )
            for seed, value in enumerate(series):
                jitter = (seed - (len(series) - 1) / 2) * 5
                parts.append(
                    f'<circle cx="{x + jitter:.1f}" cy="{y(value):.1f}" r="2.6" fill="{colour}" fill-opacity="0.5"/>'
                )

    legend_x = W - PAD["r"] + 16
    parts.append(f'<text x="{legend_x}" y="{PAD["t"] + 6}" class="ink" font-weight="600">{words["arm"]}</text>')
    for row, ((_, _, colour), name) in enumerate(zip(ARMS, words["arms"], strict=True)):
        yy = PAD["t"] + 28 + row * 20
        parts.append(
            f'<line x1="{legend_x}" y1="{yy}" x2="{legend_x + 22}" y2="{yy}" '
            f'stroke="{colour}" stroke-width="2.5" stroke-linecap="round"/>'
        )
        parts.append(f'<text x="{legend_x + 28}" y="{yy + 4}" class="ink">{name}</text>')
    note_y = PAD["t"] + 84
    for row, line in enumerate(words["notes"]):
        parts.append(f'<text x="{legend_x}" y="{note_y + row * 16}" class="faint">{line}</text>')

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


ALIGN_W, ALIGN_ROW = 760, 30
ALIGN_PAD = {"l": 250, "r": 96, "t": 46, "b": 40}
ALIGN_TEXT = {
    "en": {
        "title": "Upstream's settings, measured against the same-seed baseline",
        "note": "dot = one seed, bar = mean of three",
        "empty": "no alignment arm has results yet",
    },
    "zh": {
        "title": "上游的几处设置，对同 seed 基线量出来的差值",
        "note": "散点 = 单个 seed，横杠 = 三 seed 均值",
        "empty": "对照臂尚未产出结果",
    },
}


def align_svg(table, lang="en"):
    """One row per alignment arm: the paired deltas, and the mean once three seeds are in.

    A separate figure because these compare block configurations on one backbone, which the
    seven-generation axis cannot express. Drawn from the same numbers as the tables.
    """
    words, style = ALIGN_TEXT[lang], TEXT[lang]
    headline = {block for block, _, _ in ARMS}
    rows = [
        (backbone, block, series)
        for (backbone, block), series in sorted(table.items())
        if block not in headline and series
    ]
    height = ALIGN_PAD["t"] + ALIGN_PAD["b"] + ALIGN_ROW * max(len(rows), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ALIGN_W} {height}" width="{ALIGN_W}" '
        f'height="{height}" font-family="{style["font"]}" font-size="12">',
        "<style>",
        "  .ink{fill:#3d4451}.rule{stroke:#c9ced8}.zero{stroke:#8b93a3}.faint{fill:#78808f}",
        "  @media (prefers-color-scheme:dark){",
        "    .ink{fill:#c9ced8}.rule{stroke:#454b57}.zero{stroke:#7e8797}.faint{fill:#98a0af}}",
        "</style>",
        f'<text x="16" y="22" class="ink" font-size="13" font-weight="600">{words["title"]}</text>',
    ]
    if not rows:
        parts.append(f'<text x="16" y="{ALIGN_PAD["t"] + 10}" class="faint">{words["empty"]}</text></svg>')
        return "\n".join(parts)

    values = [v for _, _, series in rows for v in series]
    span = max(abs(min(values)), abs(max(values))) * 1.2 or 0.01
    plot_w = ALIGN_W - ALIGN_PAD["l"] - ALIGN_PAD["r"]

    def x(value):
        return ALIGN_PAD["l"] + plot_w / 2 + value / span * plot_w / 2

    for tick in (-1, -0.5, 0, 0.5, 1):
        xx = x(span * tick)
        cls = "zero" if tick == 0 else "rule"
        dash = "" if tick == 0 else ' stroke-dasharray="3 4"'
        parts.append(
            f'<line x1="{xx:.1f}" y1="{ALIGN_PAD["t"] - 12}" x2="{xx:.1f}" '
            f'y2="{height - ALIGN_PAD["b"] + 4}" class="{cls}"{dash} stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.1f}" y="{height - ALIGN_PAD["b"] + 20}" class="faint" '
            f'text-anchor="middle">{span * tick:+.4f}</text>'
        )

    for index, (backbone, block, series) in enumerate(rows):
        yy = ALIGN_PAD["t"] + ALIGN_ROW * index + 8
        colour = "#2f6f9f" if statistics.mean(series) >= 0 else "#b5453b"
        parts.append(
            f'<text x="{ALIGN_PAD["l"] - 12}" y="{yy + 4}" class="ink" text-anchor="end">'
            f"{backbone} {block.replace('e4k2w0.01', '').lstrip('-') or 'default'}</text>"
        )
        if len(series) >= 3:
            mean = statistics.mean(series)
            parts.append(
                f'<line x1="{x(mean):.1f}" y1="{yy - 7}" x2="{x(mean):.1f}" y2="{yy + 7}" '
                f'stroke="{colour}" stroke-width="2.5" stroke-linecap="round"/>'
            )
        for seed, value in enumerate(series):
            jitter = (seed - (len(series) - 1) / 2) * 4
            parts.append(
                f'<circle cx="{x(value):.1f}" cy="{yy + jitter:.1f}" r="2.6" fill="{colour}" fill-opacity="0.55"/>'
            )
        wins = sum(1 for v in series if v > 0)
        parts.append(
            f'<text x="{ALIGN_W - ALIGN_PAD["r"] + 14}" y="{yy + 4}" class="faint">'
            f"{statistics.mean(series):+.4f}  {wins}/{len(series)}</text>"
        )

    parts.append(f'<text x="16" y="{height - 10}" class="faint">{words["note"]}</text></svg>')
    return "\n".join(parts)


def alignment(tables):
    """Every arm outside the two headline ones, as rows the docs page can tabulate.

    The upstream-alignment arms compare block configurations on one backbone, a different question
    from the seven-generation figure and not something its axes can carry. Empty until those runs
    land, and the page shows nothing while it is.
    """
    rows, headline = [], {block for block, _, _ in ARMS}
    for (backbone, block), series in sorted(tables["mAP50"].items()):
        if block in headline or not series:
            continue
        seeds = ", ".join(f"{value:.4f}" for value in series)
        rows.append(
            f'    {{backbone: "{backbone}", arm: "{block}", mean: {statistics.mean(series):.4f}, '
            f'ci: "{interval(series)}", seeds: [{seeds}], wins: {sum(1 for v in series if v > 0)}}}'
        )
    return rows


def data_js(tables):
    """The same numbers as a tiny module the docs site reads without a fetch.

    Shipping the data as a script keeps every page path correct under the i18n plugin, which a
    relative fetch would not survive. Both metrics travel together so switching between them
    costs no request.
    """
    rows = []
    for backbone in ORDER:
        for block, name, _ in ARMS:
            cells = []
            for metric, _key in METRICS:
                series = tables[metric].get((backbone, block))
                if not series:
                    continue
                seeds = ", ".join(f"{value:.4f}" for value in series)
                wins = sum(1 for value in series if value > 0)
                cells.append(f'"{metric}": {{mean: {statistics.mean(series):.4f}, seeds: [{seeds}], wins: {wins}}}')
            if cells:
                joined = ", ".join(cells)
                rows.append(f'    {{backbone: "{backbone}", arm: "{name}", {joined}}}')
    names = ", ".join(f'"{metric}"' for metric, _ in METRICS)
    body = ",\n".join(rows)
    extra = ",\n".join(alignment(tables))
    return (
        "// Generated by scripts/charts.py from results/*.json. Do not edit by hand.\n"
        "window.ESMOE_EFFECT = {\n"
        f"  metrics: [{names}],\n"
        f"  rows: [\n{body}\n  ],\n"
        f"  alignment: [\n{extra}\n  ]\n"
        "};\n"
    )


def summary(values):
    """Mean and the two ends of its 95% interval, as numbers a chart can place."""
    mean, lo, hi = bounds(values)
    return {
        "mean": round(mean, 4),
        "lo": None if lo is None else round(lo, 4),
        "hi": None if hi is None else round(hi, 4),
    }


def protocol(records):
    """How much the protocol matrix is: runs and card-hours, by backbone and by card."""
    runs = [r for r in records if PROTOCOL in variant(r)]
    by_backbone, by_card = {}, {}
    for r in runs:
        for table, name in ((by_backbone, arm(r)[0]), (by_card, r["hardware"]["gpu"])):
            cell = table.setdefault(name, {"runs": 0, "hours": 0.0})
            cell["runs"] += 1
            cell["hours"] = round(cell["hours"] + r["budget"].get("gpu_hours", 0.0), 2)
    return {
        "records": len(records),
        "runs": len(runs),
        "hours": round(sum(r["budget"].get("gpu_hours", 0.0) for r in runs), 1),
        "epochs": 120,
        "imgsz": 800,
        "batch": 32,
        "backbones": by_backbone,
        "cards": by_card,
    }


def same_config(records):
    """The four arms per precision and seed, and each difference with its interval."""
    seeds = four_arms(records)
    found = {}
    for metric, key in METRICS:
        rows, groups = [], {}
        for (family, seed), arms in seeds.items():
            value = {name: arms[name]["metrics"][key] for name in FOUR}
            rows.append(
                {
                    "precision": family,
                    "seed": seed,
                    **{name: round(value[name], 4) for name in FOUR},
                    "hours": {name: arms[name]["budget"]["gpu_hours"] for name in FOUR},
                }
            )
            for name, fn in DELTAS.items():
                groups.setdefault((family, name), []).append(fn(value))
        found[metric] = {
            "rows": rows,
            "differences": [
                {
                    "precision": family,
                    "name": name,
                    "values": [round(v, 4) for v in values],
                    "positive": sum(1 for v in values if v > 0),
                    **summary(values),
                }
                for (family, name), values in groups.items()
            ],
        }
    return found


def floor(repeats):
    """Every protocol-budget run that was trained twice, and how far apart the two landed."""
    pairs = []
    for (label, seed), first, again in repeats:
        budget = schedule(first)
        if not budget.startswith("e120f1i800"):
            continue
        a, b = first["metrics"][KEYS[0]], again["metrics"][KEYS[0]]
        pairs.append(
            {
                "label": label,
                "seed": seed,
                "precision": "fp32" if budget.endswith("fp32") else "mixed",
                "first": round(a, 4),
                "repeat": round(b, 4),
                "gap": round(abs(a - b), 4),
            }
        )
    return pairs


def blocks_of(path):
    """The blocks of a routing record, old single-block records included.

    Mirrors routing.normalise, which cannot be imported here without pulling in torch.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    return record.get("blocks", [record])


def routing(runs, records):
    """Per checkpoint: how concentrated the dispatch is, how many experts died, and the paired delta."""
    proto = [r for r in runs if PROTOCOL in variant(r)]
    base = {arm(r): r for r in proto if r["config"]["arch"] == "baseline"}
    points = []
    for r in proto:
        path = ROUTING / f"{published(r)}-best.json"
        if r["config"]["arch"] == "baseline" or arm(r) not in base or not path.is_file():
            continue
        cfg, blocks = r["config"], blocks_of(path)
        points.append(
            {
                "run": published(r),
                "backbone": arm(r)[0],
                "variant": variant(r).split("@")[0],
                "balance": "none" if cfg.get("aux_weight", 0.01) == 0 else cfg.get("balance") or "switch",
                "blocks": len(blocks),
                "top1": round(statistics.mean(max(b["top1_share"]) for b in blocks), 3),
                "dead": sum(len(b["dead_experts"]) for b in blocks),
                "delta": round(r["metrics"][KEYS[0]] - base[arm(r)]["metrics"][KEYS[0]], 4),
            }
        )
    # The B arm each comparison was registered on, so a repeat of the same seed cannot add a second row.
    chosen = sorted(
        (ROUTING / f"{published(arms['B'])}.json", family, seed) for (family, seed), arms in four_arms(records).items()
    )
    arms = [
        {
            "precision": family,
            "seed": seed,
            "blocks": [
                {"dead": b["dead_experts"], "top1": b["top1_share"], "usage": b["usage"]} for b in blocks_of(path)
            ],
        }
        for path, family, seed in chosen
        if path.is_file()
    ]
    return {"points": points, "same_config": arms}


def table_rows(path, width):
    """Cells of every markdown table row with `width` columns; nothing if the table was never written."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines if line.startswith("|")]
    return [cells for cells in rows if len(cells) == width]


def parsed(path, rows):
    """A table that exists but yields no rows changed shape; drawing nothing would hide that."""
    if path.is_file() and not rows:
        raise SystemExit(f"{path.relative_to(ROOT)}: no rows parsed; its columns changed")
    return rows


def pressure():
    """The balancing terms' gradient per unit weight, read back from the table pressure.py wrote."""
    rows = []
    for cells in table_rows(PRESSURE, 7):
        if cells[1].replace(".", "", 1).isdigit():
            rows.append(
                {
                    "checkpoint": cells[0],
                    "spread": float(cells[1]),
                    "switch": float(cells[2]),
                    "gshard_probs": float(cells[3]),
                    "gshard": float(cells[4]),
                    "master": float(cells[5]),
                }
            )
    return parsed(PRESSURE, rows)


def buckets(runs):
    """Paired COCO-area deltas per arm, paired the way scripts/buckets.py pairs them."""
    records = {}
    for path in BUCKETS.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "coco" in record:
            records[record["weights"].removesuffix(".pt").removesuffix("-best")] = record
    pairs = [(run, records[published(run)]) for run in runs if published(run) in records]
    base = {arm(run): bucket for run, bucket in pairs if run["config"]["arch"] == "baseline"}
    cells = {}
    for run, bucket in sorted(pairs, key=lambda pair: (variant(pair[0]), pair[0]["seed"])):
        reference = base.get(arm(run))
        if run["config"]["arch"] == "baseline" or not reference:
            continue
        cells.setdefault(variant(run), []).append(
            {"seed": run["seed"], **{k: round(bucket["coco"][k] - reference["coco"][k], 4) for k in AREAS}}
        )
    return [{"variant": name, "seeds": seeds} for name, seeds in cells.items()]


def selection(runs):
    """Every paired cell outside the protocol budget and the one-epoch smoke runs: the selection stage."""
    found = {}
    for metric, key in METRICS:
        _, deltas = paired(runs, key)
        for label, values in deltas.items():
            budget = label.split("@")[1].split("[")[0]
            if PROTOCOL in label or budget.startswith("e1f"):
                continue
            found.setdefault(label, {"variant": label, "budget": budget})[metric] = [round(v, 4) for v in values]
    return sorted(found.values(), key=lambda cell: cell["variant"])


def by_size():
    """Router probability against object size, over every block analysed, and which kernel leads."""
    correlations, leaders = [], {}
    for path in sorted(ROUTING.glob("*.json")):
        for block in blocks_of(path):
            correlations += [round(c, 3) for c in block["prob_vs_scale_corr"]]
            lead = max(range(len(block["top1_share"])), key=block["top1_share"].__getitem__)
            kernel = str(block["kernels"][lead])
            leaders[kernel] = leaders.get(kernel, 0) + 1
    blocks = sum(leaders.values())
    return {"blocks": blocks, "correlations": correlations, "leaders": leaders}


def release_check():
    """The released checkpoint measured on the fork and rebuilt with this package, metric by metric."""
    rows = [
        {"metric": cells[0], "fork": float(cells[1]), "esmoe": float(cells[2])}
        for cells in table_rows(RELEASE_CHECK, 5)
        if cells[0].startswith("metrics/")
    ]
    return parsed(RELEASE_CHECK, rows)


def recipe_parity():
    """Step-for-step traces: the gap between two trainers, next to the gap one nudged router layer makes."""
    sections, current = [], None
    lines = RECIPE_PARITY.read_text(encoding="utf-8").splitlines() if RECIPE_PARITY.is_file() else []
    for line in lines:
        if line.startswith("## "):
            current = {"pair": line[3:].strip(), "epochs": []}
            sections.append(current)
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if current is not None and len(cells) == 8 and cells[0].isdigit():
            current["epochs"].append({"epoch": int(cells[0]), "loss": float(cells[2]), "relative": float(cells[3])})
    return parsed(RECIPE_PARITY, [section for section in sections if section["epochs"]])


def cost(runs):
    """Card-hours of each grafted protocol run over its same-seed, same-card baseline."""
    proto = [r for r in runs if PROTOCOL in variant(r) and r["budget"].get("gpu_hours")]
    base = {arm(r): r["budget"]["gpu_hours"] for r in proto if r["config"]["arch"] == "baseline"}
    ratios = {}
    for r in proto:
        if r["config"]["arch"] != "baseline" and arm(r) in base:
            ratios.setdefault(variant(r), []).append(round(r["budget"]["gpu_hours"] / base[arm(r)], 3))
    return [{"variant": name, "ratios": values} for name, values in sorted(ratios.items())]


def extras():
    """Everything the experiments page draws beyond the seven-generation figure, in one object."""
    records = load()
    runs, repeats = dedupe(records)
    return {
        "dataset": json.loads(DATASET.read_text(encoding="utf-8")) if DATASET.is_file() else None,
        "protocol": protocol(records),
        "same_config": same_config(records),
        "floor": floor(repeats),
        "routing": routing(runs, records),
        "pressure": pressure(),
        "buckets": buckets(runs),
        "selection": selection(runs),
        "scale": by_size(),
        "release_check": release_check(),
        "recipe_parity": recipe_parity(),
        "cost": cost(runs),
    }


def data_module(tables=None):
    """The whole of docs/javascripts/data.js, so a test can tell whether the committed file is current."""
    tables = tables or {metric: collect(key) for metric, key in METRICS}
    more = json.dumps(extras(), ensure_ascii=False, separators=(",", ":"))
    return data_js(tables) + f"window.ESMOE_DATA = {more};\n"


def main():
    tables = {metric: collect(key) for metric, key in METRICS}
    table = tables["mAP50"]
    if not table:
        raise SystemExit("no protocol runs found in results/")
    for lang, path in SVG_OUT.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg(table, lang), encoding="utf-8")
    for lang, path in ALIGN_OUT.items():
        path.write_text(align_svg(table, lang), encoding="utf-8")
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    DATA_OUT.write_text(data_module(tables), encoding="utf-8")
    figures = len(SVG_OUT) + len(ALIGN_OUT)
    print(f"wrote {figures} figures and {DATA_OUT.relative_to(ROOT)} covering {len(table)} arms")
    for (backbone, block), values in sorted(table.items()):
        print(f"  {backbone:<9} {block:<20} {statistics.mean(values):+.4f}  n={len(values)}")


if __name__ == "__main__":
    main()
