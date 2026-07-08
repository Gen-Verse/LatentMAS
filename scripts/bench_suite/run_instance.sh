#!/usr/bin/env bash
# Benchmark-suite instance driver: runs each given config sequentially with
# resume, retrying up to MAX_ATTEMPTS per config. benchmark_runner.py's Stage-E
# now checkpoints per (benchmark, language) chunk within a mode (not just on
# full-mode completion), so a kill mid-mode only re-does the in-flight chunk
# instead of the whole mode -- retries are genuinely cheap now, hence the
# higher attempt budget below. The python call itself is nohup'd so it
# survives a SIGHUP if this driver's controlling terminal ever goes away
# (it is already double-forked to init by the caller, but belt-and-suspenders).
# One instance owns the GPUs in $2 via CUDA_VISIBLE_DEVICES; configs address
# them as cuda:0/1/2.
#
# Usage: run_instance.sh <instance_name> <gpu_ids> <config1.yaml> [config2.yaml ...]
set -u
INSTANCE=$1
GPUS=$2
shift 2

export CUDA_VISIBLE_DEVICES=$GPUS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGDIR=logs/bench_suite
mkdir -p "$LOGDIR"
DRIVER_LOG="$LOGDIR/${INSTANCE}.driver.log"

echo "[$(date -u +%FT%TZ)] instance=$INSTANCE gpus=$GPUS configs=$*" >> "$DRIVER_LOG"

MAX_ATTEMPTS=6
OVERALL_RC=0
for CFG in "$@"; do
  NAME=$(basename "$CFG" .yaml)
  RC=1
  for ATTEMPT in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "[$(date -u +%FT%TZ)] START cfg=$NAME attempt=$ATTEMPT" >> "$DRIVER_LOG"
    nohup python scripts/run_coordination_pipeline.py --config "$CFG" --resume \
      >> "$LOGDIR/${NAME}.log" 2>&1
    RC=$?
    echo "[$(date -u +%FT%TZ)] END   cfg=$NAME attempt=$ATTEMPT exit=$RC" >> "$DRIVER_LOG"
    [ "$RC" -eq 0 ] && break
    sleep 60
  done
  if [ "$RC" -ne 0 ]; then
    echo "[$(date -u +%FT%TZ)] GIVING UP on cfg=$NAME after $MAX_ATTEMPTS attempts" >> "$DRIVER_LOG"
    OVERALL_RC=1
  fi
done

echo "[$(date -u +%FT%TZ)] instance=$INSTANCE DONE rc=$OVERALL_RC" >> "$DRIVER_LOG"
exit $OVERALL_RC
