#!/bin/bash
# Lanes pull jobs off a shared file; a run that already has all 120 epochs is skipped.
# Job line: <base.yaml> <arm> <seed> <tag> [aux_weight]
#
# JOBS/LANE0/SLOTS let a second worker join a card whose original lanes have already drained:
# it waits for a trainer slot instead of adding a third run to a card sized for two.
# FORK is a checkout of YOLO-Master; the upstream and forkbase arms train on its ultralytics.
export MACA_HOME=/opt/maca MACA_PATH=/opt/maca
export PATH=/opt/conda/bin:$PATH
cd "${TOOLKIT:-/data/esmoe-toolkit}"

JOBS=${JOBS:-/data/jobs.txt}
LOCK=/data/jobs.lock
SLOTS=${SLOTS:-2}
LANES=${LANES:-2}
LANE0=${LANE0:-1}
FORK=${FORK:-/data/yolo-master}
# 1 trains in mixed precision, 0 in FP32. It reaches the run's name as well as its arguments, so an
# FP32 run never shares a directory with the mixed-precision run of the same configuration.
AMP=${AMP:-1}
# Seconds a run may take. A run the fork has switched to FP32 shares its card for longer than 12 h.
LIMIT=${LIMIT:-86400}
LAUNCH=/data/launch.lock
touch "$LOCK" "$LAUNCH"
echo $$ > "${PIDFILE:-/data/queue.pid}"

pop() { flock "$LOCK" -c "head -1 $JOBS; sed -i 1d $JOBS"; }

# One trainer per run wears the timeout wrapper; its dataloader workers do not, so this counts runs.
busy() { ps -eo args | grep -c "^timeout -k 60 $LIMIT python3 scripts/train.py"; }

lane() {
  local id=$1 job base arm seed tag w arch flag fork stem name rows suffix pid
  exec 9>"$LAUNCH"
  while :; do
    job=$(pop); [ -z "$job" ] && break
    read -r base arm seed tag w <<< "$job"
    # Normalised the way python's %g formats it, so a job line saying 0.0 or 1.50 still names the
    # run the trainer will name it, and a resumed run finds the directory it left behind.
    w=$(printf '%g' "${w:-0.01}")
    fork=""
    case "$arm" in
      baseline) flag=""; arch="baseline" ;;
      esmoe)    flag="--esmoe"; arch="esmoe" ;;
      rewire)   flag="--esmoe --rewire"; arch="esmoe-rewire" ;;
      gshard)   flag="--esmoe --balance gshard"; arch="esmoe-gshard" ;;
      master)   flag="--esmoe --balance master"; arch="esmoe-master" ;;
      stages)   flag="--esmoe --rewire --at backbone_stages"; arch="esmoe-rewire-stages" ;;
      norm)     flag="--esmoe --out-norm"; arch="esmoe-norm" ;;
      dense)    flag="--esmoe --dense-training"; arch="esmoe-dense" ;;
      recipe)   flag="--esmoe --grafted --recipe upstream"; arch="esmoe-upstream" ;;
      upstream) flag="--upstream"; arch="upstream"; fork=1 ;;
      forkbase) flag=""; arch="baseline"; fork=1 ;;
    esac
    suffix=""
    case "$flag" in
      *--esmoe*) flag="$flag --aux-weight $w"; [ "$w" != "0.01" ] && suffix="-w$w" ;;
    esac
    stem=$(basename "$base" .yaml)
    case "$flag" in *--grafted*) stem=${stem%-esmoe} ;; esac
    name="${stem}-${arch}${suffix}-e120-s${seed}${tag}$([ "$AMP" = 0 ] && echo -fp32)${fork:+-fork}"
    rows=$(cat "runs/$name/results.csv" 2>/dev/null | wc -l)
    if [ "$rows" -ge 121 ]; then echo "[lane$id] skip $name"; continue; fi

    # Claim a slot under the launch lock, and hold it until the new trainer shows up in ps, so two
    # lanes cannot both read the last free slot as theirs.
    flock 9
    while [ "$(busy)" -ge "$SLOTS" ]; do sleep 30; done
    echo "[lane$id] $(date -Is) start $name"
    # env execs timeout, so the process busy() counts still starts with it.
    env ${fork:+PYTHONPATH="$FORK"} timeout -k 60 "$LIMIT" python3 scripts/train.py $flag --base "$base" --epochs 120 \
      --fraction 1.0 --batch 32 --imgsz 800 --patience 0 --seed "$seed" --tag="$tag" --amp "$AMP" \
      >> "/data/lane$id.log" 2>&1 &
    pid=$!
    sleep 90
    flock -u 9
    wait "$pid" || echo "[lane$id] FAILED $name"
    echo "[lane$id] $(date -Is) done $name"
  done
  echo "[lane$id] queue empty"
}

for i in $(seq "$LANE0" $((LANE0 + LANES - 1))); do lane "$i" & done
wait
echo "=== queue drained $(date -Is) ==="
