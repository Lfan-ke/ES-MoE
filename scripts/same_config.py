"""YOLO-Master's yolo-master-n trained two ways under one protocol.

Each seed trains four arms on one card, so every difference below is taken inside a card:

    A   YOLO-Master's fork, yolo-master-n with its four ES_MOE blocks
    A0  YOLO-Master's fork, the same model without the blocks
    B   official ultralytics + esmoe, the same model with four ESMoE blocks and recipe="upstream"
    C   official ultralytics, the same model without the blocks

A - B compares the two implementations as each is run. (A - A0) - (B - C) compares what the block
adds inside each framework, where the frameworks' own differences cancel, provided both arms of a
framework trained at the same precision; the precision table says whether they did.

Metrics come from `results/measured/`, where `scripts/measure.py` puts what one piece of code
measures from every arm's final weights, and fall back to the training curve's best epoch for a
seed that has not been measured. The two are not interchangeable: each trainer validates with its
own framework, and the B arm validated through a pruning bug fixed in d6f4dcc.

    uv run python scripts/same_config.py        # results/same_config.md
"""

import statistics
from collections import defaultdict

from report import KEYS, ROOT, interval, load, stack

ARMS = ("A", "A0", "B", "C")
DELTAS = {
    "A - B": lambda r: r["A"] - r["B"],
    "A - A0": lambda r: r["A"] - r["A0"],
    "B - C": lambda r: r["B"] - r["C"],
    "(A - A0) - (B - C)": lambda r: (r["A"] - r["A0"]) - (r["B"] - r["C"]),
}


def arm_of(record) -> str | None:
    cfg, budget = record["config"], record["budget"]
    if "yolo-master-n" not in cfg["model_yaml"] or (budget["epochs"], budget.get("imgsz")) != (120, 800):
        return None
    if record["dataset"]["fraction"] != 1.0:
        return None
    fork = stack(record).endswith("/yolo-master")
    if cfg["arch"] == "baseline":
        return "A0" if fork else "C"
    if fork and cfg["arch"] == "upstream":
        return "A"
    if not fork and cfg.get("recipe") == "upstream":
        return "B"
    return None


def precision(record) -> str:
    """The precision a run asked for, which decides which family of seeds it belongs to.

    The fork switches mixed precision off the first time a gradient is not finite, so in the mixed
    family A and A0 end up in FP32 while B and C stay mixed, and A - B carries that difference as
    well as the implementation's. The FP32 family asks every arm for FP32 up front, and the same
    seed therefore appears once in each family: they are different experiments, not repeats.
    """
    return "fp32" if record["budget"].get("amp", True) is False else "mixed"


def collect(records) -> dict[tuple[str, int], dict[str, dict]]:
    """(precision, seed) -> arm -> record; a later record of the same key replaces an earlier one."""
    seeds: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for record in records:
        if arm := arm_of(record):
            seeds[(precision(record), record["seed"])][arm] = record
    return {key: arms for key, arms in sorted(seeds.items()) if set(arms) == set(ARMS)}


def card(arms: dict[str, dict]) -> str:
    """The card a seed trained on, which all four arms must share once the framework is set aside."""
    cards = {stack(record).removesuffix("/yolo-master") for record in arms.values()}
    if len(cards) != 1:
        raise ValueError(f"one seed's arms trained on different stacks: {sorted(cards)}")
    return cards.pop()


def main() -> None:
    seeds = collect(load())
    # `load` reports a re-measured run through that measurement; a seed whose four arms were all
    # measured that way is comparable arm to arm, one that mixes the two is not.
    sources = {
        key: "measured" if all("metrics_by_trainer" in arms[arm] for arm in ARMS) else "curve"
        for key, arms in seeds.items()
    }
    out = ["# Same configuration: YOLO-Master's fork against official ultralytics + esmoe", ""]
    if not seeds:
        out.append("No seed has all four arms yet.")
    for key in KEYS:
        rows, deltas = [], defaultdict(lambda: defaultdict(list))
        for (family, seed), arms in seeds.items():
            value = {arm: arms[arm]["metrics"][key] for arm in ARMS}
            found = {name: fn(value) for name, fn in DELTAS.items()}
            for name, delta in found.items():
                deltas[family][name].append(delta)
            cells = " | ".join(f"{value[arm]:.4f}" for arm in ARMS)
            differences = " | ".join(f"{d:+.4f}" for d in found.values())
            rows.append(f"| {family} | {seed} | {card(arms)} | {sources[(family, seed)]} | {cells} | {differences} |")
        if not rows:
            continue
        out += [
            f"## {key}",
            "",
            "| precision | seed | card | metrics | A | A0 | B | C | " + " | ".join(DELTAS) + " |",
            "|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|" + ":--:|" * len(DELTAS),
            *rows,
            "",
            "| precision | difference | seeds | mean | 95% CI | positive |",
            "|:--:|:--:|:--:|:--:|:--:|:--:|",
        ]
        for family, found in deltas.items():
            for name, values in found.items():
                positive = sum(1 for v in values if v > 0)
                out.append(
                    f"| {family} | {name} | {len(values)} | {statistics.mean(values):+.4f} "
                    f"| {interval(values)} | {positive}/{len(values)} |"
                )
        out.append("")
    if seeds:
        out += [
            "## What each run actually trained with",
            "",
            "| precision | seed | arm | run | amp asked | amp at end | batch at end | epochs replayed | hours |",
            "|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|",
        ]
        for (family, seed), arms in seeds.items():
            for arm in ARMS:
                budget = arms[arm]["budget"]
                out.append(
                    f"| {family} | {seed} | {arm} | {arms[arm]['experiment_id']} | {budget.get('amp', True)} "
                    f"| {budget.get('amp_at_end', '?')} | {budget.get('batch_at_end', '?')} "
                    f"| {budget.get('epochs_replayed', '?')} | {budget['gpu_hours']:.2f} |"
                )
    table = "\n".join(out)
    (ROOT / "results" / "same_config.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
