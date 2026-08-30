# E1 mid-term report - ES-MoE plug-and-play toolkit

## 1. Question and hypothesis

Does the community ES-MoE block survive a budget-fair, multi-seed comparison, and can it be
delivered as an installable plug-in for stock Ultralytics rather than a fork-only module? The
hypothesis under test: at equal data, schedule and augmentation, adding one ES-MoE block with a
correctly wired load-balancing loss improves detection quality by a small but consistent margin.

## 2. Commit, data, budget, control, commands

- Toolkit: `Lfan-ke/ES-MoE` @ `bf2ebe2`, version 0.1.0. Baselines locked per task book:
  public BASE_REF `acce839c` (main at 2026-08-21 23:59:59) with release reference
  `YOLO-Master-v26.08` -> `43d40117c`; the runtime dependency is stock `ultralytics` 8.4.101/8.4.132.
- Data: VisDrone2019-DET, YOLO layout, `fraction=0.25`, imgsz 640, from scratch.
- Budget: 20 epochs, batch 32, AMP on, one RTX 4090 D; every arm identical.
- Control: same-seed baseline without the block; ablation arms differ only in `num_experts`,
  `top_k` or the aux weight.
- Reproduce:

      python scripts/capture_env.py
      EPOCHS=20 FRACTION=0.25 SEEDS="0 1 2" bash scripts/sweep.sh
      SEED=0 bash scripts/ablate.sh
      python scripts/report.py

## 3. Evidence completed

- Three-line integration works on stock Ultralytics: `inject_esmoe()`, `graft()`, `attach_aux_loss()`.
  `tests/test_ultralytics.py` builds and forwards YOLOv8n / YOLO11n / YOLO12n with the block, checks
  head renumbering after insertion, and asserts the aux term both changes the optimised loss and
  produces router gradients. 11 tests pass on the training machine.
- The auxiliary loss is visible in training output as its own `esmoe_aux` column in `results.csv`,
  non-zero in train and val, which is the task book's "not merely a config key" requirement.
- Selection under one budget (`docs/SELECTION.md`): 4 experts, top-2, aux weight 0.01 wins; top-1
  routing loses to the baseline; 8 experts buys nothing at 13% more parameters; turning the aux loss
  off costs 0.0012 mAP50 at identical parameter count.
- Confirmation on three seeds: paired win 3/3, +0.0021 mAP50 (0.0909 -> 0.0930) and +0.0005
  mAP50-95, at +10.4% parameters.
- Reproduction package present: `configs/ scripts/ results/ env/ README.md limitations.md`, one JSON
  record per run with the task book's minimum experiment fields.

## 4. Conclusions and what is still open

Established: the plug-in path works without patching Ultralytics; the aux loss reaches the
optimiser; the ranking of the candidate configurations at this budget; run-to-run determinism (a
repeated identical run reproduced mAP50 exactly).

Also established after the mid-term run: the gain survives a fourfold increase in training data. On
the full VisDrone training set at the same 20-epoch budget the paired result is again 3/3 seeds and
+0.0021 mAP50, with mAP50-95 improving from +0.0005 on the subset to +0.0019.

And established by the 50-epoch run: **the gain decays as the schedule lengthens**. At 50 epochs the
paired mAP50 mean falls to +0.0013 with one seed slightly negative (2/3), and mAP50-95 to +0.0008
(3/3). The honest reading is that the block buys earlier convergence rather than a higher ceiling.

Established as a negative result: the gain does not transfer to YOLO11n. Same data, budget and
seeds, paired mean -0.0009 mAP50 with one seed of three in favour. The compatibility claim therefore
covers mechanics on four generations; the accuracy claim covers YOLOv8n.

Not established: whether any gain remains at convergence, on COCO, or at other model scales. The
50-epoch trend points the other way and the experiment was not run to convergence, so no ceiling
claim is made. The
per-arm standard deviations overlap, so only the paired comparison carries the claim, and no
significance test is claimed on three seeds. One of the three full-data seeds is effectively a tie
(+0.0001), so the effect is not reliable within a single run. DDP behaviour of the aux term is
untested.

## 5. Next stage, risk triggers, collaboration

- Done since drafting: Colab quick start, full-dataset confirmation, PyPI releases 0.1.0 and 0.1.1,
  documentation site, CI including a notebook execution check.
- Next: English and Chinese tutorials; a longer schedule or a second backbone if GPU time allows.
- Risk trigger: if a longer budget erases the paired gain, the deliverable stays a correctness and
  tooling contribution (aux-loss plumbing, compatibility matrix, budget-fair harness) and the
  accuracy claim is withdrawn rather than defended.
- Risk trigger: single 24GB GPU. If multi-generation confirmation does not fit, fall back to two
  generations plus one backbone, as the task book's degradation plan allows.
- Collaboration: the multi-seed harness and per-run JSON records are the shared artefact E2 can
  reuse; the compatibility matrix is what E1 owes back to the other topics.
