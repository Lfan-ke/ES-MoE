# Known limitations

Scope of the evidence in `results/`, stated before anyone has to ask.

## Benchmark scope

- Numbers come from VisDrone2019-DET at imgsz 640, trained from scratch (no pretrained weights): a 25% subset for candidate selection, the full training set for confirmation. They are not COCO numbers and must not be read as a reproduction of the ES-MoE-N anchor (2.68M / 8.7 GFLOPs / 42.7 mAP), which is a COCO figure.
- Short budgets (tens of epochs from scratch) sit far from convergence. A gap measured here bounds the ranking of two blocks under equal budget; it does not predict the converged gap.
- The measured gain is **backbone-specific**: it is present on YOLOv8n (3/3 seeds at 20 epochs) and absent on YOLO11n (1/3 seeds, mean -0.0009 mAP50) under the same data, budget and seeds. Compatibility across four generations is a statement about mechanics, not about accuracy.
- The measured gain does not follow the schedule in one direction: paired mAP50 means are +0.0021 (3/3 seeds) at 20 epochs, +0.0013 (2/3) at 50 and +0.0046 (2/3) at 100, while the spread between seeds widens with the budget (+0.0104 to -0.0011 at 100 epochs). Treat the effect as small, positive on average and unreliable within a single run.
- Every number comes from one GPU. The DDP mechanics are verified (`scripts/verify.py`: the real worker file trains in a fresh interpreter; two gloo ranks each compute their own auxiliary term and their router gradients agree after all-reduce), but no multi-GPU accuracy figure exists.
- Sparse dispatch leaves the experts a batch did not route to out of the graph, so DDP needs `find_unused_parameters=True`. ultralytics builds it that way by default but switches it off under `compile=True`; the block cannot be used in that combination.

## Method scope

- `ESMoE` is channel preserving: it maps `c1 -> c1`. Upstream `ES_MOE` also supports `c1 -> c2`; that path is deliberately not reproduced, because stock `parse_model` assumes `c2 == ch[f]` for third-party modules.
- Channels are inferred on the first forward. A model that is scripted, exported, or `state_dict`-loaded before any forward has no expert weights to load yet.
- `attach_aux_loss` patches the task model class and keeps the weight at process scope, because the trainer rebuilds the model and takes the EMA copy before any callback runs. Consequently a process trains one aux-loss setting at a time, and a checkpoint reloaded in a process that never calls `attach_aux_loss` trains without the aux term.
- Export was verified against ONNX Runtime on inputs that route to different experts (`scripts/verify.py`); releases 0.1.0 to 0.1.2 baked the traced routing into the exported graph and should be re-exported.
- The load-balancing term is the Switch-Transformer formulation (`num_experts * sum(importance * load)`). No EMA normalisation of the aux magnitude is applied, unlike YOLO-Master's mixture controller.

## Protocol and evaluation

- The early records (imgsz 640, 20/50/100 epochs) do not sit under the repository's reproduction protocol and serve only same-budget comparison and parameter screening. A protocol-conformant set (imgsz 800, 120 epochs, `patience=0`, the full 548-image validation set) now exists: YOLOv8n paired mAP50 +0.0025 (2/3), mAP50-95 +0.0004. Conclusions are drawn from that set.
- Metrics in the run records come from the ultralytics evaluator with its default `max_det` of 300; `results/buckets.md` re-evaluates the six protocol-conformant checkpoints under COCO conventions with maxDets = 500. The two sets are not interchangeable, and neither is directly comparable to the official VisDrone leaderboard.
- The area buckets (small < 32², medium 32²–96², large ≥ 96², on ground-truth boxes in the original image) are the COCO definition adopted by this project, not a VisDrone one. Paired over three seeds: large objects get consistently worse (APl −0.0104, ARl −0.0129, 0/3 wins) and small-object recall consistently better (ARs +0.0026, 3/3 wins), by too little to lift AP.

## Reporting

- Seeds are reported individually and as mean ± sample standard deviation. With three seeds, the standard deviation is a coarse estimate; no significance test is claimed. On the full dataset one of the three seeds is effectively a tie (+0.0001 mAP50), so the effect is consistent in sign but not reliable within a single run.
- On top of the parameter increase the block costs about 5% wall-clock per epoch at imgsz 640 (829 s against 790 s, RTX 4090 D) and about 9% at imgsz 800 (85 against 78 minutes, RTX 4090).
