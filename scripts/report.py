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
        if record["status"] == "success":
            runs.append(record)
    return runs


def spread(values):
    if len(values) < 2:
        return f"{values[0]:.4f}" if values else "-"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def main():
    runs = load()
    by_arch = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for key in KEYS:
            if key in r["metrics"]:
                by_arch[r["config"]["arch"]][key].append(r["metrics"][key])

    lines = ["| run | arch | seed | mAP50 | mAP50-95 | params | wall_s |", "|:--:|:--:|:--:|:--:|:--:|:--:|:--:|"]
    for r in runs:
        lines.append(
            f"| {r['experiment_id']} | {r['config']['arch']} | {r['seed']} "
            f"| {r['metrics'].get(KEYS[0], 0):.4f} | {r['metrics'].get(KEYS[1], 0):.4f} "
            f"| {r['params']} | {r['budget']['wall_seconds']} |"
        )
    lines += ["", "| arch | seeds | mAP50 | mAP50-95 |", "|:--:|:--:|:--:|:--:|"]
    for arch, metrics in by_arch.items():
        n = len(metrics[KEYS[0]])
        lines.append(f"| {arch} | {n} | {spread(metrics[KEYS[0]])} | {spread(metrics[KEYS[1]])} |")

    table = "\n".join(lines)
    (ROOT / "results" / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
