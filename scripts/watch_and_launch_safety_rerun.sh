#!/usr/bin/env bash
# Polls nvidia-smi until >=3 GPUs are simultaneously idle (<500MiB used), then
# reruns het_belebele_sg's single_agent_baseline mode with the fixed
# SafetyAgent (specialized_agents.py: anti-disclaimer prompt instruction +
# max_new_tokens 256->384, landed 20260707). That mode's cached result
# (.cache/checkpoints/bench_suite/het_belebele_sg/coordination/_results/
# ..._mode__single_agent_baseline.pt) was computed pre-fix and carries 82/200
# unparseable safety verdicts (mostly disclaimer-only responses and verdicts
# truncated by the old 256-token budget) -- see
# results/bench_suite/het_belebele_sg/safety_reparse_summary.json.
#
# 3 GPUs requested to match het_belebele_sg.yaml's per-role device assignment
# (translation/reasoning/safety each get their own cuda:N) -- single_agent_
# baseline's per-task executor is picked by the router (_pick_single_agent),
# not fixed to one role, so any of the 3 loaded agents' devices may be used.
#
# The stale cache is moved aside (not deleted) as *.stale-safety-parser-v2,
# following the existing *.stale-safety-bug convention already in that
# directory from the previous safety-parser fix.
#
# GPU claims are serialized via flock across ALL gpu-watching scripts so two
# watchers can never double-book the same idle GPU.
set -u
cd "$(dirname "$0")/.."
LOG=logs/bench_suite/safety_rerun_watcher.log
LOCK=/tmp/multilinguallatentmas_gpu_claim.lock
CACHE_DIR=.cache/checkpoints/bench_suite/het_belebele_sg/coordination/_results
CACHE_FILE="$CACHE_DIR/coord__aisingapore_Llama-SEA-LION-v3-8B-IT_sail_Sailor2-8B-Chat_meta-llama_Llama-3.1-8B-Instruct_CohereLabs_aya-expanse-8b__06c47cdd__mode__single_agent_baseline.pt"
mkdir -p logs/bench_suite
export LOG
log() { echo "[$(date -u +%FT%TZ)] [safety_rerun] $*" >> "$LOG"; }

try_claim_and_launch() (
  set -u
  N_NEEDED=3
  FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 500) print $1}')
  N_FREE=$(echo "$FREE_GPUS" | grep -c '[0-9]')
  [ "$N_FREE" -lt "$N_NEEDED" ] && return 1

  CLAIMED=$(echo "$FREE_GPUS" | head -"$N_NEEDED" | paste -sd, -)
  log "claimed gpus=$CLAIMED"

  if [ -f "$CACHE_FILE" ]; then
    mv "$CACHE_FILE" "$CACHE_FILE.stale-safety-parser-v2"
    log "moved stale cache -> $(basename "$CACHE_FILE").stale-safety-parser-v2"
  else
    log "no cached single_agent_baseline result found at $CACHE_FILE; proceeding anyway"
  fi
  [ -f "$CACHE_FILE.reparsed.json" ] && mv "$CACHE_FILE.reparsed.json" "$CACHE_FILE.reparsed.json.stale-safety-parser-v2"

  export CUDA_VISIBLE_DEVICES=$CLAIMED
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONPATH=src

  log "launching het_belebele_sg --comm-modes single_agent_baseline (fixed parser) on gpus=$CLAIMED"
  setsid nohup python scripts/run_coordination_pipeline.py \
    --config configs/bench_suite/het_belebele_sg.yaml --resume \
    --comm-modes single_agent_baseline \
    >> logs/bench_suite/het_belebele_sg.safety_rerun.log 2>&1 < /dev/null &
  DRIVER_PID=$!
  disown -a
  log "launched, driver pid=$DRIVER_PID"

  ( while kill -0 "$DRIVER_PID" 2>/dev/null; do sleep 60; done
    log "driver pid=$DRIVER_PID done -- refreshing safety_reparse_summary.json"
    PYTHONPATH=src python scripts/recompute_safety_rate.py --config het_belebele_sg --force \
      >> logs/bench_suite/het_belebele_sg.safety_rerun.log 2>&1
    log "safety_reparse_summary.json refreshed" ) \
    >> "$LOG" 2>&1 < /dev/null &
  disown -a

  return 0
)

log "watching for >=3 idle GPUs (<500MiB used)"
while true; do
  if flock "$LOCK" bash -c "$(declare -f try_claim_and_launch log); try_claim_and_launch"; then
    break
  fi
  sleep 300
done
