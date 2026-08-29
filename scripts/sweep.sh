#!/usr/bin/env bash
# Budget-fair on/off sweep: identical data, epochs, batch and image size; only the block differs.
set -u
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-20}
FRACTION=${FRACTION:-0.25}
BATCH=${BATCH:-32}
SEEDS=${SEEDS:-"0 1 2"}
TAG=${TAG:-}

for seed in $SEEDS; do
  for arch in "" "--esmoe"; do
    timeout 14400 python3 scripts/train.py $arch --epochs "$EPOCHS" --fraction "$FRACTION" \
      --batch "$BATCH" --seed "$seed" --tag="$TAG" || echo "run failed: seed=$seed arch=${arch:-baseline}"
  done
done
pkill -9 -f 'scripts/train.py' 2>/dev/null
echo "sweep done"
