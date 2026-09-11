"""Aggregate results/*.json into a seed-aware comparison table."""

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("metrics/mAP50(B)", "metrics/mAP50-95(B)")


def published(record) -> str:
    """The name a run's checkpoint, curve and analyses are published under.

    Normally the run directory. Two hosts once trained under one directory name, so the record of
    the earlier run names what its files were published as instead.
    """
    artifact = record["artifact"]
    return artifact.get("published_as") or PurePosixPath(artifact["path"]).parts[-3]


def load():
    runs = []
    for path in sorted((ROOT / "results").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        # results/ also holds verify.json, which is a different kind of evidence entirely.
        if "experiment_id" in record and record.get("status") == "success":
            runs.append(record)
    return runs


def stack(record):
    """The accelerator and runtime build a run was measured on.

    Each cell pairs its seeds on one card. Keeping the stack in the key means a rerun on another
    host forms its own group rather than a silent extra sample in someone else's mean.
    """
    hw = record.get("hardware", {})
    gpu = hw.get("gpu", "?").replace("NVIDIA GeForce ", "").replace("RTX ", "").replace(" ", "").lower()
    torch_version = hw.get("torch", "?")
    build = re.match(r"([A-Za-z]+)(\d+\.\d+)", torch_version.partition("+")[2])
    runtime = f"{build[1].lower()}{build[2]}" if build else "torch" + ".".join(torch_version.split(".")[:2])
    # YOLO-Master's fork reports the same ultralytics version as the release it forked, but trains
    # differently; its runs pair with its own baseline.
    ref = record.get("git_ref")
    trained_by = (ref.get("framework", "ultralytics") if isinstance(ref, dict) else "ultralytics").partition("@")[0]
    return f"{gpu}/{runtime}" + ("" if trained_by == "ultralytics" else f"/{trained_by}")


def variant(record):
    """Label an arm by everything that has to match for a comparison to be fair.

    Backbone, block config, budget and hardware all move the metric, so folding them into one
    label would silently average across different experiments. Image size is part of the budget:
    800 and 640 runs of the same schedule are different experiments, not repeats of one.
    """
    cfg, data, budget = record["config"], record["dataset"], record["budget"]
    backbone = Path(cfg["model_yaml"]).stem.split("-esmoe")[0]
    block = "baseline" if cfg["arch"] == "baseline" else f"e{cfg['num_experts']}k{cfg['top_k']}w{cfg['aux_weight']}"
    if cfg.get("rewire"):
        block += "-rewire"
    if cfg.get("balance", "switch") not in ("switch", "none"):
        block += f"-{cfg['balance']}"
    # The upstream-alignment switches change the model, so they belong in the key: a run with the
    # output norm is not a repeat of one without it.
    if cfg.get("out_norm"):
        block += "-norm"
    if cfg.get("dense_training"):
        block += "-dense"
    if cfg.get("sparse_inference") is False:
        block += "-denseval"
    if cfg.get("dynamic_threshold"):
        block += f"-t{cfg['dynamic_threshold']:g}"
    if cfg.get("blocks", 1) not in (0, 1):
        block += f"-x{cfg['blocks']}"
    return f"{backbone}-{block}@e{budget['epochs']}f{data['fraction']:g}i{budget.get('imgsz', '?')}[{stack(record)}]"


def arm(record):
    """The part of the label a baseline shares with the ESMoE arms it is compared against."""
    cfg, data, budget = record["config"], record["dataset"], record["budget"]
    backbone = Path(cfg["model_yaml"]).stem.split("-esmoe")[0]
    return backbone, budget["epochs"], data["fraction"], budget.get("imgsz"), stack(record), record["seed"]


def spread(values):
    if not values:
        return "-"
    if len(values) < 2:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


# Two-sided 95% critical values of Student's t by degrees of freedom. Three seeds leaves two, and
# the interval is correspondingly wide: that is the width a three-sample estimate actually has.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def interval(values):
    """95% confidence interval for a mean paired delta."""
    if len(values) < 2:
        return "-"
    half = T95.get(len(values) - 1, 1.96) * statistics.stdev(values) / len(values) ** 0.5
    mean = statistics.mean(values)
    return f"[{mean - half:+.4f}, {mean + half:+.4f}]"


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
        out += ["", "| variant | seeds | mean delta | 95% CI | wins |", "|:--:|:--:|:--:|:--:|:--:|"]
        for name, values in deltas.items():
            wins = sum(1 for v in values if v > 0)
            out.append(
                f"| {name} | {len(values)} | {statistics.mean(values):+.4f} "
                f"| {interval(values)} | {wins}/{len(values)} |"
            )

    if repeats:
        out += [
            "",
            "## Determinism (repeated runs of an identical config)",
            "",
            "| variant | seed | mAP50 first | mAP50 repeat | gap |",
            "|:--:|:--:|:--:|:--:|:--:|",
        ]
        by_stack = defaultdict(list)
        for (name, seed), first, again in repeats:
            a, b = first["metrics"].get(KEYS[0], 0), again["metrics"].get(KEYS[0], 0)
            out.append(f"| {name} | {seed} | {a:.4f} | {b:.4f} | {abs(a - b):.4f} |")
            by_stack[(stack(first), first["budget"]["epochs"])].append(abs(a - b))
        # This is the denominator for every effect above: where two runs of one configuration are
        # this far apart, an arm's mean delta of the same size says nothing about the arm. Budget
        # is part of the key because a one-epoch probe and a full run do not measure the same thing.
        out += ["", "| stack | epochs | repeats | mean gap | largest gap |", "|:--:|:--:|:--:|:--:|:--:|"]
        for (name, epochs), gaps in sorted(by_stack.items()):
            out.append(f"| {name} | {epochs} | {len(gaps)} | {statistics.mean(gaps):.4f} | {max(gaps):.4f} |")

    table = "\n".join(out)
    (ROOT / "results" / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
