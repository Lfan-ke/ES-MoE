# Known limitations

Scope of the evidence in `results/`, stated before anyone has to ask.

## Benchmark scope

- Numbers come from VisDrone2019-DET at imgsz 640, trained from scratch (no pretrained weights): a 25% subset for candidate selection, the full training set for confirmation. They are not COCO numbers and must not be read as a reproduction of the ES-MoE-N anchor (2.68M / 8.7 GFLOPs / 42.7 mAP), which is a COCO figure.
- Short budgets (tens of epochs from scratch) sit far from convergence. A gap measured here bounds the ranking of two blocks under equal budget; it does not predict the converged gap.
- The measured gain **shrinks with the schedule**: paired mAP50 mean +0.0021 (3/3 seeds) at 20 epochs against +0.0013 (2/3 seeds) at 50 epochs, and mAP50-95 +0.0019 against +0.0008. Read the block as buying earlier convergence, not a higher ceiling.
- One machine, one RTX 4090 D. No multi-GPU or DDP run has been made, so DDP-specific aux-loss behaviour is untested.

## Method scope

- `ESMoE` is channel preserving: it maps `c1 -> c1`. Upstream `ES_MOE` also supports `c1 -> c2`; that path is deliberately not reproduced, because stock `parse_model` assumes `c2 == ch[f]` for third-party modules.
- Channels are inferred on the first forward. A model that is scripted, exported, or `state_dict`-loaded before any forward has no expert weights to load yet.
- `attach_aux_loss` patches the task model class and keeps the weight at process scope, because the trainer rebuilds the model and takes the EMA copy before any callback runs. Consequently a process trains one aux-loss setting at a time, and a checkpoint reloaded in a process that never calls `attach_aux_loss` trains without the aux term.
- Export was verified against ONNX Runtime on inputs that route to different experts (`scripts/verify.py`); releases 0.1.0 to 0.1.2 baked the traced routing into the exported graph and should be re-exported.
- The load-balancing term is the Switch-Transformer formulation (`num_experts * sum(importance * load)`). No EMA normalisation of the aux magnitude is applied, unlike YOLO-Master's mixture controller.

## Reporting

- Seeds are reported individually and as mean ± sample standard deviation. With three seeds, the standard deviation is a coarse estimate; no significance test is claimed. On the full dataset one of the three seeds is effectively a tie (+0.0001 mAP50), so the effect is consistent in sign but not reliable within a single run.
- The block costs about 5% wall-clock per epoch on top of the parameter increase (829 s against 790 s per full-data run on one RTX 4090 D).
