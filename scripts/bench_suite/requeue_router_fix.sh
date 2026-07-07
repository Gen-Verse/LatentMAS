#!/usr/bin/env bash
# Router-fix requeue watcher (see [[router-safety-fixes-predate-jul5-runs]]).
#
# The het/hom bench_suite instances launched 2026-07-05T15:50 hold their GPUs
# for the lifetime of the *already-running* python process, which loaded the
# pre-fix AttentionRouter (uniform routing, confidence pinned ~0.335-0.341)
# into memory before commit 3f75cfb seeded real routing keys -- a running
# process never picks up a code fix, only a fresh `python` invocation does.
# run_instance.sh's own retry-on-crash loop already re-execs python (so a
# crash mid-mode self-heals onto the fixed code for free); the only case that
# needs help is the instance finishing *cleanly* under the old code, which
# writes token_based_mas/latent_based_mas_ours result caches stamped with the
# stale router regime that `--resume` would otherwise silently keep reusing
# forever (checkpointing only checks key presence, not code_regime).
#
# This script waits for the whole driver (both configs, e.g.
# het_belebele_sg.yaml then het_mgsm.yaml) to vacate its GPUs, then for each
# config: invalidates any mode cache not stamped code_regime.router ==
# "prototype-seeded", and if anything was invalidated (or never finished),
# reruns just those two modes with --comm-modes so single_agent_baseline
# (already valid, router-independent) is skipped via the existing cache.
#
# Usage (daemonize like run_instance.sh / rerun_baselines_queue.sh):
#   setsid nohup bash scripts/bench_suite/requeue_router_fix.sh \
#     <instance_name> <gpu_ids> <driver_pid> <config1.yaml> [config2.yaml ...] \
#     >> logs/bench_suite/<instance_name>.router_fix_watcher.log 2>&1 &
#
# Idempotent: safe to launch more than once or after a restart -- configs
# whose caches are already fully prototype-seeded are skipped, and a
# mid-flight invocation of this script's own rerun is resumable via
# --resume.
set -u
INSTANCE=$1
GPUS=$2
DRIVER_PID=$3
shift 3
CONFIGS=("$@")

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
LOGDIR=logs/bench_suite
mkdir -p "$LOGDIR"

log() { echo "[$(date -u +%FT%TZ)] [$INSTANCE/router_fix] $*"; }

log "watching driver_pid=$DRIVER_PID gpus=$GPUS configs=${CONFIGS[*]}"
while kill -0 "$DRIVER_PID" 2>/dev/null; do
  sleep 60
done
log "driver_pid=$DRIVER_PID gone -- GPUs $GPUS now free, checking mode caches"

MAX_ATTEMPTS=6
for CFG in "${CONFIGS[@]}"; do
  NAME=$(basename "$CFG" .yaml)
  RESULTS_DIR=".cache/checkpoints/bench_suite/${NAME}/coordination/_results"

  mapfile -t STALE < <(python scripts/bench_suite/check_mode_regime.py "$RESULTS_DIR" 2>>"$LOGDIR/${INSTANCE}.router_fix_watcher.log")
  if [ "${#STALE[@]}" -eq 0 ]; then
    log "cfg=$NAME: no stale/missing token_based_mas or latent_based_mas_ours cache -- already fixed or never finished under old code, nothing to invalidate"
  else
    for F in "${STALE[@]}"; do
      [ -f "$F" ] || continue
      mv "$F" "${F}.stale-router-bug"
      log "cfg=$NAME: invalidated $(basename "$F") -> $(basename "$F").stale-router-bug"
    done
  fi

  # Re-check: only rerun if either mode is still missing a fixed-regime cache
  # (covers both "just invalidated" and "process never got that far").
  mapfile -t REMAINING < <(python scripts/bench_suite/check_mode_regime.py "$RESULTS_DIR" 2>>"$LOGDIR/${INSTANCE}.router_fix_watcher.log")
  if [ "${#REMAINING[@]}" -ne 0 ]; then
    log "cfg=$NAME: still missing fixed-regime cache for one or more modes -- relaunching --comm-modes token_based_mas,latent_based_mas_ours"
    RC=1
    for ATTEMPT in $(seq 1 "$MAX_ATTEMPTS"); do
      log "cfg=$NAME attempt=$ATTEMPT START"
      nohup python scripts/run_coordination_pipeline.py --config "$CFG" --resume \
        --comm-modes token_based_mas,latent_based_mas_ours \
        >> "$LOGDIR/${NAME}.router_fix.log" 2>&1
      RC=$?
      log "cfg=$NAME attempt=$ATTEMPT END exit=$RC"
      [ "$RC" -eq 0 ] && break
      sleep 60
    done
    [ "$RC" -ne 0 ] && log "cfg=$NAME: GIVING UP after $MAX_ATTEMPTS attempts"
  else
    log "cfg=$NAME: fixed-regime caches present for both modes, skipping rerun"
  fi
done

log "done"
