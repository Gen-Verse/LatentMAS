#!/usr/bin/env bash
# Polls nvidia-smi until >=3 GPUs are simultaneously idle (<500MiB used), then
# launches a targeted --comm-modes oneflow rerun of het_belebele_sg (the
# "Single-agent OneFlow (het)" row of tab:coord). single_agent_baseline stays
# cached and untouched; token_based_mas/latent_based_mas_ours are excluded
# from this invocation's --comm-modes, so it can run in parallel with the
# main het driver process without recomputing or interfering with those
# modes (they share the same on-disk result cache, keyed by mode name).
#
# 3 GPUs requested to match het_belebele_sg.yaml's existing per-role device
# assignment (translation/reasoning/safety each get their own cuda:N); only
# the "primary" role's model actually gets loaded under OneFlow (the other
# two role-agents share its weights in-process), but the config's device
# indices assume 3 GPUs are visible, so under-provisioning would index-error.
#
# GPU claims are serialized via flock across ALL gpu-watching scripts so two
# watchers can never double-book the same idle GPU.
set -u
cd "$(dirname "$0")/.."
LOG=logs/bench_suite/oneflow_watcher.log
LOCK=/tmp/multilinguallatentmas_gpu_claim.lock
mkdir -p logs/bench_suite
export LOG
log() { echo "[$(date -u +%FT%TZ)] [oneflow] $*" >> "$LOG"; }

try_claim_and_launch() (
  set -u
  N_NEEDED=3
  FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 500) print $1}')
  N_FREE=$(echo "$FREE_GPUS" | grep -c '[0-9]')
  [ "$N_FREE" -lt "$N_NEEDED" ] && return 1

  CLAIMED=$(echo "$FREE_GPUS" | head -"$N_NEEDED" | paste -sd, -)
  log "claimed gpus=$CLAIMED"

  export CUDA_VISIBLE_DEVICES=$CLAIMED
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONPATH=src

  log "launching het_belebele_sg --comm-modes oneflow on gpus=$CLAIMED"
  setsid nohup python scripts/run_coordination_pipeline.py \
    --config configs/bench_suite/het_belebele_sg.yaml --resume \
    --comm-modes oneflow \
    >> logs/bench_suite/het_belebele_sg.oneflow.log 2>&1 < /dev/null &
  disown -a
  log "launched, driver pid=$!"
  return 0
)

log "watching for >=3 idle GPUs (<500MiB used)"
while true; do
  if flock "$LOCK" bash -c "$(declare -f try_claim_and_launch log); try_claim_and_launch"; then
    break
  fi
  sleep 300
done
