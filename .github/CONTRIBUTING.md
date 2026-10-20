# Contributing

    uv sync --group dev
    uv run ruff format . && uv run ruff check .
    uv run pytest -q
    uv run python scripts/check_notebook.py     # only if you touched notebooks/

All four must pass before a pull request. CI runs the same commands on Python 3.12 and 3.13, against
both the latest ultralytics and 8.4.101, plus a leg with ultralytics absent, the Keras layer on
tensorflow, jax and torch, and the notebook on CPU.

## Layout

A **backend** is the tensor library the block is written in, an **integration** is a training stack
built on one. Between them sits a backend-free core, `esmoe/core/`, which reads the arguments and settles
them into one plan. A new backend adds only
the tensor work and exports the names `esmoe/torch/` exports; a new integration exports `graft` and
`equip`. Both are one row in `BACKENDS` or `INTEGRATIONS`.

## Conventions

- Comments explain **why**, never what; obvious things carry no comment.
- Type hints use builtin generics and `X | None`; `typing` imports only when unavoidable.
- Private names take a double underscore, and each module ends with an explicit `__all__`.
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
