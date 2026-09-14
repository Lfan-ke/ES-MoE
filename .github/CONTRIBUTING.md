# Contributing

    uv sync --group dev
    uv run ruff format . && uv run ruff check .
    uv run pytest -q
    uv run python scripts/check_notebook.py     # only if you touched notebooks/

All four must pass before a pull request. CI runs the same commands on Python 3.10 and 3.12, against
both the latest ultralytics and 8.4.101, plus the notebook on CPU.

## Conventions

- Comments explain **why**, never what; obvious things carry no comment.
- Type hints use builtin generics and `X | None`; `typing` imports only when unavoidable.
- The public API is what `esmoe.__all__` lists; everything else may change without notice.
- One commit message line, `type(scope): summary.`, in English.

## Evidence

- A change that touches training behaviour needs a test that fails without it, or a new check in
  `scripts/verify.py`.
- Any statement about accuracy needs a run record in `results/` produced by `scripts/train.py`, with
  the same data, budget and seeds as the baseline it is compared against, read against the
  pre-registered lines in `docs/JUDGMENT.md`. Paired per-seed deltas, not two means.
- A negative or inconclusive result still goes in `results/`, and the conclusion in the docs gets
  requalified rather than dropped.

A config key without a wired effect, a benchmark at mismatched budgets, or a number without a run
record is not merged.
