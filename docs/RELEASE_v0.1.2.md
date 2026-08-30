# ES-MoE 0.1.2

## Fixed

- The training progress header showed a single loss column on ultralytics releases that name losses
  from the returned loss dict (8.4.13x and newer). `attach_aux_loss` appended `esmoe_aux` to a
  `loss_names` tuple that is empty at `on_train_start` on those releases, so the header read
  `Epoch GPU_mem esmoe_aux Instances Size` while four numbers were printed under it. The logged and
  optimised values were always correct; only the header was wrong. Older releases, which fix the
  names before training, are unaffected and still get the name appended.

## Changed

- The package now lives at the repository root rather than under `src/`.
- The quick-start notebook ships the transcript of a real CPU run, so its output can be read without
  executing anything.
