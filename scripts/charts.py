"""Draw the paired-effect figures straight from results/*.json.

Two outputs, one source of numbers. The docs site gets `javascripts/data.js`, which the ECharts
view reads to draw an interactive version; README and the wiki get `assets/effect.svg`, because
GitHub strips scripts there and a static figure is the only thing that renders. Both come from
report.py, so a figure cannot drift away from the tables it illustrates.
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report import KEYS, dedupe, interval, load, paired  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SVG_OUT = {"en": ROOT / "docs" / "assets" / "effect.svg", "zh": ROOT / "docs" / "assets" / "effect.zh.svg"}
ALIGN_OUT = {"en": ROOT / "docs" / "assets" / "alignment.svg", "zh": ROOT / "docs" / "assets" / "alignment.zh.svg"}
DATA_OUT = ROOT / "docs" / "javascripts" / "data.js"

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
    DATA_OUT.write_text(data_js(tables), encoding="utf-8")
    figures = len(SVG_OUT) + len(ALIGN_OUT)
    print(f"wrote {figures} figures and {DATA_OUT.relative_to(ROOT)} covering {len(table)} arms")
    for (backbone, block), values in sorted(table.items()):
        print(f"  {backbone:<9} {block:<20} {statistics.mean(values):+.4f}  n={len(values)}")


if __name__ == "__main__":
    main()
