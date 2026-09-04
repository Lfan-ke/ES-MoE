# Results

Every row is one run record in [`results/`](https://github.com/Lfan-ke/ES-MoE/tree/main/results), and
this page is generated from them by `scripts/report.py`. Arms are grouped by backbone, block
configuration and budget, so two different budgets never land in the same average. Read the paired
tables rather than the two means, and read [limitations](limitations.md) before quoting a number;
cell-by-cell verdicts are on the [judgment lines](JUDGMENT.md) page.

The 36 protocol-matrix `best.pt` checkpoints and their full training arguments live on the
[`checkpoints` branch](https://github.com/Lfan-ke/ES-MoE/tree/checkpoints) (Git LFS, isolated from
`main`): every bucket and routing number recomputes from them.

--8<-- "results/summary.md"

## Area buckets

--8<-- "results/buckets.md"

## Router behaviour

--8<-- "results/routing.md"
