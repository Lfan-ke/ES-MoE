# Baseline, increment boundary and work package

Fixed per the 2026-08-23 supplementary rules on historical results and increment acceptance, which
take precedence over the task book wherever the two differ on baselines and increments.

## Locked references

| item | value |
|:--:|:--:|
| public BASE_REF | `acce839c7e895d6b179de7f7093fa879e237cc7b` (Tencent/YOLO-Master main at 2026-08-21 23:59:59 +0800) |
| release the library version comes from | `YOLO-Master-v26.08` -> `43d40117c30811204fb9347efeabddce15f11a62` |
| runtime dependency | stock `ultralytics==8.4.101`, the version both references carry |
| deliverable | `Lfan-ke/ES-MoE` |
| FINAL_REF | set at closing; current head is recorded in every run record under `git_ref.toolkit` |

## Why the usual diff command does not apply here

The deliverable is a new standalone distribution, not a fork of YOLO-Master, so it shares no commit
history with the baseline and `git merge-base` is empty by construction. The increment is therefore
the whole repository: its first commit `acbcdfe` is dated 2026-08-25, after the lock, and every
later commit is signed and pushed to a public repository.

The ES-MoE block here is a reimplementation against the published design, not a file copy of
`ultralytics/nn/modules/moe/modules.py`. What is new relative to the baseline:

- channel-preserving block with lazy channel inference, so a stock `parse_model` can size it and one
  config line works across scales, where upstream configs hardcode the channel count;
- config grafting with head renumbering;
- an auxiliary-loss path for stock ultralytics, which has none of upstream's registry, mixture-loss
  collector or trainer plumbing;
- a budget-fair harness producing per-run machine-readable records, a paired multi-seed comparison
  and a determinism check.

## Historical work, stated plainly

The SoC-phase work on issue #52 and the merged upstream PRs #192 and #194 predate the lock. They are
cited as capability evidence and as the origin of this topic only. They are not claimed as this
round's P1/P2 increment, and no historical-result registration was filed before the 2026-08-25
deadline, so nothing before the lock is claimed as new work here.

## Work package

Single owner (`Lfan-ke`): module and aux-loss plumbing, config grafting, compatibility matrix across
YOLOv8/YOLO11/YOLO12, budget-fair multi-seed harness, candidate selection, documentation and release.
Evidence: commits in this repository, `tests/`, `results/`, `docs/SELECTION.md`, `docs/MIDTERM.md`.
