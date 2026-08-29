#!/usr/bin/env bash
# Candidate selection under one budget: same data, epochs, batch and seed; only the block config
# moves. Run after sweep.sh so the baseline it is compared against already exists.
set -u
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-20}
FRACTION=${FRACTION:-0.25}
BATCH=${BATCH:-32}
SEED=${SEED:-0}
# num_experts:top_k:aux_weight
CANDIDATES=${CANDIDATES:-"2:1:0.01 4:1:0.01 4:2:0.01 8:2:0.01 4:2:0.0"}

for c in $CANDIDATES; do
  IFS=":" read -r experts topk weight <<< "$c"
  timeout 3600 python3 scripts/train.py --esmoe --epochs "$EPOCHS" --fraction "$FRACTION" \
    --batch "$BATCH" --seed "$SEED" --num-experts "$experts" --top-k "$topk" \
    --aux-weight "$weight" --tag="-e${experts}k${topk}w${weight}" \
    || echo "run failed: $c"
done
pkill -9 -f 'scripts/train.py' 2>/dev/null
echo "ablation done"
