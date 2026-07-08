#!/usr/bin/env bash
# Polls nvidia-smi until >=4 GPUs are simultaneously idle (<500MiB used), then
# launches the staircase ablation (scripts/run_ablation_staircase.py) pinned
# to those 4 GPUs via CUDA_VISIBLE_DEVICES. This is "multiple GPU-days" per
# the runner's own docstring, so it only starts once real capacity exists
# rather than contending with the in-flight het/hom bench_suite runs.
#
# If results/mechanistic/geo_profiles.json doesn't exist yet (prerequisite
# for rows 3-6, produced by the GPU7 chain watcher), runs only rows 0-2 for
# now; the caller should relaunch this script for the remaining rows once
# that artifact lands (idempotent: run_ablation_staircase.py isolates each
# row's own output dir, so partial completion is safe to resume/extend).
#
# GPU claims are serialized via flock across ALL gpu-watching scripts
# (staircase, oneflow, ...): the entire detect-and-launch decision happens
# inside one held lock per attempt, not just the nvidia-smi read, so two
# watchers can never both see the same idle GPU and double-book it.
set -u
cd "$(dirname "$0")/.."
LOG=logs/bench_suite/staircase_watcher.log
LOCK=/tmp/multilinguallatentmas_gpu_claim.lock
mkdir -p logs/bench_suite
export LOG
log() { echo "[$(date -u +%FT%TZ)] [staircase] $*" >> "$LOG"; }

try_claim_and_launch() (
  set -u
  N_NEEDED=4
  FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 500) print $1}')
  N_FREE=$(echo "$FREE_GPUS" | grep -c '[0-9]')
  [ "$N_FREE" -lt "$N_NEEDED" ] && return 1

  CLAIMED=$(echo "$FREE_GPUS" | head -"$N_NEEDED" | paste -sd, -)
  log "claimed gpus=$CLAIMED"

  ROWS="0,1,2"
  if [ -f results/mechanistic/geo_profiles.json ]; then
    ROWS=""  # default = all rows (0-6 + 7a)
    log "geo_profiles.json present -- running full staircase"
  else
    log "geo_profiles.json missing -- running rows 0,1,2 only for now"
  fi

  export CUDA_VISIBLE_DEVICES=$CLAIMED
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONPATH=src

  ROWS_ARG=()
  [ -n "$ROWS" ] && ROWS_ARG=(--rows "$ROWS")

  log "launching staircase on gpus=$CLAIMED rows=${ROWS:-all}"
  setsid nohup python scripts/run_ablation_staircase.py \
    --config configs/latent_coordination.yaml \
    "${ROWS_ARG[@]}" \
    >> logs/bench_suite/staircase_run.log 2>&1 < /dev/null &
  disown -a
  log "launched, driver pid=$!"
  return 0
)

log "watching for >=4 idle GPUs (<500MiB used)"
while true; do
  if flock "$LOCK" bash -c "$(declare -f try_claim_and_launch log); try_claim_and_launch"; then
    break
  fi
  sleep 300
done
