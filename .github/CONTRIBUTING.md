# Contributing

    uv sync --group dev
    uv run ruff format . && uv run ruff check .
    uv run pytest -q
    uv run python scripts/check_notebook.py     # only if you touched notebooks/

All four must pass before a pull request. CI runs the same commands on Python 3.10 and 3.12, against
both the locked ultralytics and 8.4.101, plus the notebook on CPU.

## Conventions

- Comments explain **why**, never what; obvious things carry no comment.
- Type hints use builtin generics and `X | None`; `typing` imports only when unavoidable.
- Public API stays small: `ESMoE`, `inject_esmoe`, `graft`, `attach_aux_loss`, `equip`,
  `collect_aux_loss`, `blocks`.
- One commit message line, `type(scope): summary.`, in English.

## Claims about numbers

Any statement about accuracy needs a run record in `results/` produced by `scripts/train.py`, with
the same data, budget and seeds as the baseline it is compared against. Paired per-seed deltas, not
two means. If a result is negative or inconclusive, it still goes in `results/`, and the conclusion
in the docs gets requalified rather than dropped.
