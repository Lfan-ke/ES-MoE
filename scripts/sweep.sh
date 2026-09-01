#!/usr/bin/env bash
# Budget-fair on/off sweep: identical data, epochs, batch and image size; only the block differs.
set -u
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-20}
FRACTION=${FRACTION:-0.25}
BATCH=${BATCH:-32}
SEEDS=${SEEDS:-"0 1 2"}
TAG=${TAG:-}
BASE=${BASE:-yolov8n.yaml}
IMGSZ=${IMGSZ:-640}
PATIENCE=${PATIENCE:-0}
LIMIT=${LIMIT:-43200}
ARMS=${ARMS:-"baseline esmoe"}

for seed in $SEEDS; do
  for arm in $ARMS; do
    case "$arm" in baseline) arch="";; esmoe) arch="--esmoe";; rewire) arch="--esmoe --rewire";; esac
    timeout "$LIMIT" python3 scripts/train.py $arch --base "$BASE" --epochs "$EPOCHS" --fraction "$FRACTION" \
      --batch "$BATCH" --imgsz "$IMGSZ" --patience "$PATIENCE" --seed "$seed" --tag="$TAG" \
      || echo "run failed: seed=$seed arm=$arm"
  done
done
pkill -9 -f 'scripts/train.py' 2>/dev/null
echo "sweep done"
