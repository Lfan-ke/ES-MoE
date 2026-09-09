#!/bin/bash
# Lanes pull jobs off a shared file; a run that already has all 120 epochs is skipped.
# Job line: <base.yaml> <arm> <seed> <tag> [aux_weight]
# LANES / LANE0 let a second worker join a card whose original lanes have already drained.
export MACA_HOME=/opt/maca MACA_PATH=/opt/maca
export PATH=/opt/conda/bin:$PATH
cd /data/esmoe-toolkit

JOBS=${JOBS:-/data/jobs.txt}
LOCK=/data/jobs.lock
LANES=${LANES:-2}
LANE0=${LANE0:-1}
touch "$LOCK"
echo $$ > "${PIDFILE:-/data/queue.pid}"

pop() { flock "$LOCK" -c "head -1 $JOBS; sed -i 1d $JOBS"; }

lane() {
  local id=$1 job base arm seed tag w arch flag name rows suffix
  while :; do
    job=$(pop); [ -z "$job" ] && break
    read -r base arm seed tag w <<< "$job"
    w=${w:-0.01}
    case "$arm" in
      baseline) flag=""; arch="baseline" ;;
      esmoe)    flag="--esmoe"; arch="esmoe" ;;
      rewire)   flag="--esmoe --rewire"; arch="esmoe-rewire" ;;
      gshard)   flag="--esmoe --balance gshard"; arch="esmoe-gshard" ;;
      master)   flag="--esmoe --balance master"; arch="esmoe-master" ;;
      stages)   flag="--esmoe --rewire --at backbone_stages"; arch="esmoe-rewire-stages" ;;
      norm)     flag="--esmoe --out-norm"; arch="esmoe-norm" ;;
      dense)    flag="--esmoe --dense-training"; arch="esmoe-dense" ;;
    esac
    [ "$arm" != "baseline" ] && flag="$flag --aux-weight $w"
    suffix=""
    [ "$arm" != "baseline" ] && [ "$w" != "0.01" ] && suffix="-w$w"
    name="${base%.yaml}-${arch}${suffix}-e120-s${seed}${tag}"
    rows=$(cat "runs/$name/results.csv" 2>/dev/null | wc -l)
    if [ "$rows" -ge 121 ]; then echo "[lane$id] skip $name"; continue; fi
    echo "[lane$id] $(date -Is) start $name"
    timeout -k 60 43200 python3 scripts/train.py $flag --base "$base" --epochs 120 \
      --fraction 1.0 --batch 32 --imgsz 800 --patience 0 --seed "$seed" --tag="$tag" \
      >> "/data/lane$id.log" 2>&1 || echo "[lane$id] FAILED $name"
    echo "[lane$id] $(date -Is) done $name"
  done
  echo "[lane$id] queue empty"
}

for i in $(seq "$LANE0" $((LANE0 + LANES - 1))); do lane "$i" & done
wait
echo "=== queue drained $(date -Is) ==="
