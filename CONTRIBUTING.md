# Contributing

## Setup

    uv sync --group dev
    uv run pytest -q

## Before a pull request

- `uv run ruff check .` and `uv run ruff format .` must pass; CI also runs the test matrix on two ultralytics pins.
- A change that touches training behaviour needs evidence, not assertion: extend `scripts/verify.py` or add a test that fails without the change.
- Experiment claims follow the repository protocol (VisDrone, imgsz 800, 120 epochs, three paired seeds) and the pre-registered judgment lines in `docs/JUDGMENT.md`. Negative results are welcome and get recorded, not buried.

## What gets rejected

A config key without a wired effect, a benchmark at mismatched budgets, or a number without a run record in `results/`.
