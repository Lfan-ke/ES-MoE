<!-- Keep all four sections. One line is enough where a section does not apply ("not applicable: docs only"). -->

### Change summary

<!-- What changed and why. -->

### Test evidence

<!-- Commands and results, and the ultralytics version they ran against. -->

### Ablation data

<!-- Required when the change can move a metric: the run records in `results/`, same data, budget and seeds as the baseline, paired per seed. -->

### Known limitations

<!-- What this change does not cover, and what would falsify it. -->

### Checklist

- [ ] `ruff format`, `ruff check` and `pytest` pass; so does `scripts/check_notebook.py` if `notebooks/` changed.
- [ ] A change to training behaviour comes with a test that fails without it, or a check in `scripts/verify.py`.
- [ ] No credentials, local paths or weights are committed.
