"""Aggregate results/*.json into a seed-aware comparison table."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("metrics/mAP50(B)", "metrics/mAP50-95(B)")


def load():
    runs = []
    for path in sorted((ROOT / "results").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        # results/ also holds verify.json, which is a different kind of evidence entirely.
        if "experiment_id" in record and record.get("status") == "success":
            runs.append(record)
    return runs


def variant(record):
    """Label an arm by everything that has to match for a comparison to be fair.

    Backbone, block config and budget all move the metric, so folding them into one label would
    silently average across different experiments.
    """
    cfg, data, budget = record["config"], record["dataset"], record["budget"]
    backbone = Path(cfg["model_yaml"]).stem.split("-esmoe")[0]
    block = "baseline" if cfg["arch"] == "baseline" else f"e{cfg['num_experts']}k{cfg['top_k']}w{cfg['aux_weight']}"
    return f"{backbone}-{block}@e{budget['epochs']}f{data['fraction']:g}"


def arm(record):
    """The part of the label a baseline shares with the ESMoE arms it is compared against."""
    cfg, data, budget = record["config"], record["dataset"], record["budget"]
    backbone = Path(cfg["model_yaml"]).stem.split("-esmoe")[0]
    return backbone, budget["epochs"], data["fraction"], record["seed"]


def spread(values):
    if not values:
        return "-"
    if len(values) < 2:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def paired(runs, key):
    """Per-seed deltas against the baseline of the same seed.

    Mean ± std of two arms hides that both arms move together across seeds; the paired delta is
    what the budget-fair comparison actually licenses with three seeds.
    """
    base = {arm(r): r["metrics"][key] for r in runs if r["config"]["arch"] == "baseline" and key in r["metrics"]}
    rows, deltas = [], defaultdict(list)
    for r in runs:
        name, key_arm = variant(r), arm(r)
        if r["config"]["arch"] == "baseline" or key_arm not in base or key not in r["metrics"]:
            continue
        delta = r["metrics"][key] - base[key_arm]
        rows.append((name, r["seed"], base[key_arm], r["metrics"][key], delta))
        deltas[name].append(delta)
    return rows, deltas


def dedupe(runs):
    """Keep one record per (variant, seed) and report repeats as a determinism check.

    Re-running an identical config is evidence about reproducibility, not an extra sample; folding
    it into the mean would silently weight that seed twice.
    """
    kept, repeats = {}, []
    for r in runs:
        key = (variant(r), r["seed"])
        if key in kept:
            repeats.append((key, kept[key], r))
        else:
            kept[key] = r
    return list(kept.values()), repeats


def main():
    runs, repeats = dedupe(load())
    by_variant = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for key in KEYS:
            if key in r["metrics"]:
                by_variant[variant(r)][key].append(r["metrics"][key])

    out = [
        "## Runs",
        "",
        "| run | variant | seed | mAP50 | mAP50-95 | params | wall_s |",
        "|:--:|:--:|:--:|:--:|:--:|:--:|:--:|",
    ]
    for r in runs:
        out.append(
            f"| {r['experiment_id']} | {variant(r)} | {r['seed']} "
            f"| {r['metrics'].get(KEYS[0], 0):.4f} | {r['metrics'].get(KEYS[1], 0):.4f} "
            f"| {r['params']} | {r['budget']['wall_seconds']} |"
        )

    out += ["", "## Across seeds", "", "| variant | seeds | mAP50 | mAP50-95 |", "|:--:|:--:|:--:|:--:|"]
    for name, metrics in by_variant.items():
        out.append(f"| {name} | {len(metrics[KEYS[0]])} | {spread(metrics[KEYS[0]])} | {spread(metrics[KEYS[1]])} |")

    for key in KEYS:
        rows, deltas = paired(runs, key)
        if not rows:
            continue
        out += [
            "",
            f"## Paired against baseline - {key}",
            "",
            "| variant | seed | baseline | variant | delta |",
            "|:--:|:--:|:--:|:--:|:--:|",
        ]
        for name, seed, b, v, d in rows:
            out.append(f"| {name} | {seed} | {b:.4f} | {v:.4f} | {d:+.4f} |")
        out += ["", "| variant | seeds | mean delta | wins |", "|:--:|:--:|:--:|:--:|"]
        for name, values in deltas.items():
            wins = sum(1 for v in values if v > 0)
            out.append(f"| {name} | {len(values)} | {statistics.mean(values):+.4f} | {wins}/{len(values)} |")

    if repeats:
        out += [
            "",
            "## Determinism (repeated runs of an identical config)",
            "",
            "| variant | seed | mAP50 first | mAP50 repeat | identical |",
            "|:--:|:--:|:--:|:--:|:--:|",
        ]
        for (name, seed), first, again in repeats:
            a, b = first["metrics"].get(KEYS[0], 0), again["metrics"].get(KEYS[0], 0)
            out.append(f"| {name} | {seed} | {a:.4f} | {b:.4f} | {'yes' if a == b else 'no'} |")

    table = "\n".join(out)
    (ROOT / "results" / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
